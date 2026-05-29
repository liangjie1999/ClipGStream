#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from einops import repeat
import sys
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model_of_reference import ReferenceClipGaussianModel
from scene.gaussian_model_of_source import SourceClipGaussianModel

import tinycudann as tcnn
import torch.nn.functional as F

import os
import open3d as o3d
import torch

from time import time


def save_anchor(anchor):
    # anchor 是 torch.Tensor
    anchor_1 = anchor.detach().cpu().numpy()  # 转成 numpy

    # 创建 open3d 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(anchor_1)

    # 保存为 .ply
    o3d.io.write_point_cloud("anchor.ply", pcd)
    print("保存完成 ✅ anchor.ply")

def generate_full_neural_gaussians(viewpoint_camera, pc,  visible_mask=None, is_training=False,  timestamp = None , opt_thro = 0.0, clip_id: int = 0, render_mode=False ):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    if timestamp == None:
        if render_mode:
            # When using render.py, the timestamp needs to be converted from 'absolute time relative to the input sequence length'
            #                                                           to 'relative time within the clip'.
            # For example, when N=1400 and M=10, when accessing the first frame, the initial timestamp is 1/1400.
            #                                       After the conversion below, the timestamp will become 1/10.

            clip_size = pc.opt.clip_size
            time_interval_01 = 1 / clip_size      
            camera_frame_idx_abs = round(viewpoint_camera.time / time_interval_01)
            
            camera_frame_idx_rel = camera_frame_idx_abs - clip_size * clip_id

            timestamp = torch.tensor(camera_frame_idx_rel / clip_size)
        else:
            timestamp = torch.tensor(viewpoint_camera.time)
        
    time_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[time_mask][visible_mask]  # [N,3]   

    timestamp = timestamp.to(anchor.device).repeat(anchor.shape[0],1) 

    dy_feat, dy_factor = pc.dynamic_module(anchor, timestamp)

    sta_feat = pc.get_anchor_feat[time_mask][visible_mask]  # [N,32]          

    if pc.opt.open_feat_cat: 
        feat = torch.cat( [sta_feat, dy_feat] ,dim=-1)   
    else:
        feat = sta_feat + dy_feat 


    # feat =   sta_feat
    # get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center.cuda().unsqueeze(0)
    ob_dist = ob_view.norm(dim=1, keepdim=True)  #  distance from camera to point
    ob_view = ob_view / ob_dist

    from time import time
    
    start_time = time()
    neural_opacity = pc.get_opacity_mlp(feat)  # [N,32+3]   
    end_time = time()
    # print(f'opacity decode time: {end_time - start_time}')


    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = ( neural_opacity > opt_thro )  
    mask = mask.view(-1)

    # select opacity                               
    opacity = neural_opacity[mask]  

    # get offset's color   
    start_time = time()
    color = pc.get_color_mlp(feat)
    end_time = time()
    # print(f'color decode time: {end_time - start_time}')

    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3]) # [mask]

    # The [:,:3] controls the step size of offset. The [:,3:] serves as the base scale for neural gaussian's shape, which means the cov MLP learn a residual scales.
    
    start_time = time()
    scale_rot = pc.get_cov_mlp(feat) 
    end_time = time()
    # print(f'cov decode time: {end_time - start_time}')

    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]

    # offsets
    # grid_offsets =  pc._offset[visible_mask].view([-1,3])  #  [N,10,3] grid_offsets =  grid_offsets[pc.dynamic_mask[visible_mask]]  
    start_time = time()
    offsets =  pc.get_offset_mlp(feat).view([-1,3])   # pc._offset[pc.dynamic_mask]  #  [N,10,3]  
    end_time = time()
    # print(f'offsets decode time: {end_time - start_time}')

    grid_scaling = pc.get_scaling[time_mask][visible_mask] # [N,6]  grid_scaling = grid_scaling[pc.dynamic_mask[visible_mask]]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    
    start_time = time()
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # 复制k
    end_time = time()
    # print(f'repeat time: {end_time - start_time}')

    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets  ], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets  = masked.split([6, 3, 3, 7, 3], dim = -1 )

    # torch.cuda.empty_cache()

    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]  
    xyz = repeat_anchor + offsets


    if scaling.shape[0]==0:
        pass

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, 
    else:
        return xyz, color, opacity, scaling, rot

def generate_coarse_neural_gaussians(viewpoint_camera, pc,  visible_mask=None, is_training=False,  timestamp = None , opt_thro = 0.0 ):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    

    # cur_dy_mask = torch.logical_and(visible_mask, pc.dynamic_mask)
    # cur_sta_mask = torch.logical_and(visible_mask, ~pc.dynamic_mask)
    
    anchor = pc.get_anchor[visible_mask]  # [N,3]
    if timestamp == None:
        timestamp = torch.tensor(viewpoint_camera.time).to(anchor.device).repeat(anchor.shape[0],1)
    else :
        timestamp = torch.tensor(timestamp).to(anchor.device).repeat(anchor.shape[0],1)
    if pc.hash:
        dy_feat, dy_factor = pc.dynamic_module(anchor, timestamp)
    else:
        dy_feat, dy_factor = pc.hexplane(anchor,timestamp)
    sta_feat = pc.get_anchor_feat[visible_mask]  # [N,32]          
    # feat = dy_factor * dy_feat + ( 1 - dy_factor ) * sta_feat  
    feat =   sta_feat
    # get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center.cuda().unsqueeze(0)
    ob_dist = ob_view.norm(dim=1, keepdim=True)  # 相机到点的距离
    ob_view = ob_view / ob_dist

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)   # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)     # [N, c+3]
    cat_sta_feat_view = torch.cat([sta_feat, ob_view], dim=1)
    cat_local_time = torch.cat([feat, anchor, timestamp], dim=1)  # [N, c+3+1]
    cat_local_view_time = torch.cat([feat, ob_view, anchor, timestamp], dim=1)  # [N, c+3+1]

    # time_indices = ( timestamp * 300 ).long()
    # time_feature = pc.time_embedding( time_indices.cuda() ).squeeze()
    # neural_opacity = pc.get_opacity_mlp( torch.cat(  [ feat , time_feature ] , dim=1) )  # [N,32+3]   
    neural_opacity = pc.get_opacity_mlp(feat)  # [N,32+3]   

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = ( neural_opacity > opt_thro ) 
    mask = mask.view(-1)

    # select opacity                               
    opacity = neural_opacity[mask]  # 去掉小于0的点

    # get offset's color   
    # color = pc.get_color_mlp(cat_local_view_wodist)  #   # [N,32+3]
    color = pc.get_color_mlp(feat)   
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3]) # [mask]

    # The [:,:3] controls the step size of offset. The [:,3:] serves as the base scale for neural gaussian's shape, which means the cov MLP learn a residual scales.
    scale_rot = pc.get_cov_mlp(feat) 
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]

    # offsets
    # grid_offsets =  pc._offset[visible_mask].view([-1,3])  #  [N,10,3] grid_offsets =  grid_offsets[pc.dynamic_mask[visible_mask]]  
    offsets =  pc.get_offset_mlp(feat).view([-1,3])   # pc._offset[pc.dynamic_mask]  #  [N,10,3]  
    grid_scaling = pc.get_scaling[visible_mask] # [N,6]  grid_scaling = grid_scaling[pc.dynamic_mask[visible_mask]]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets  ], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets  = masked.split([6, 3, 3, 7, 3], dim = -1 )



    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])



    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]  
    xyz = repeat_anchor + offsets


    if scaling.shape[0]==0:
        pass

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, 
    else:
        return xyz, color, opacity, scaling, rot

def dynamic_neural_gaussians( dynamic, viewpoint_camera, pc,  visible_mask=None, is_training=False, timestamp = None , opt_thro = 0.0):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    
    if dynamic:
        cur_dy_mask = torch.logical_and(visible_mask, pc.dynamic_mask)
    else:
        cur_dy_mask = torch.logical_and(visible_mask, ~pc.dynamic_mask)
    
    anchor = pc.get_anchor[cur_dy_mask]  # [N,3]
    if timestamp == None:
        timestamp = torch.tensor(viewpoint_camera.time).to(anchor.device).repeat(anchor.shape[0],1)
    else :
        timestamp = torch.tensor(timestamp).to(anchor.device).repeat(anchor.shape[0],1)
    if pc.hash:
        dy_feat, dy_factor = pc.dynamic_module(anchor,timestamp)
    else:
        dy_feat, dy_factor = pc.hexplane(anchor,timestamp)
    sta_feat = pc.get_anchor_feat[cur_dy_mask]  # [N,32]      
    feat = dy_factor * dy_feat + ( 1 - dy_factor ) * sta_feat  
    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center.cuda().unsqueeze(0)
    ob_dist = ob_view.norm(dim=1, keepdim=True)  # 相机到点的距离
    ob_view = ob_view / ob_dist

    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)     # [N, c+3]


    # time_indices = ( timestamp * 300 ).long()
    # time_feature = pc.time_embedding( time_indices.cuda() ).squeeze()
    # neural_opacity = pc.get_opacity_mlp( torch.cat(  [ feat , time_feature ] , dim=1) )  # [N,32+3]   
    neural_opacity = pc.get_opacity_mlp(feat)  # [N,32+3]  

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = ( neural_opacity > opt_thro )  
    mask = mask.view(-1)

    # select opacity                               
    opacity = neural_opacity[mask]  # 去掉小于0的点

    # get offset's color   
    # color = pc.get_color_mlp(cat_local_view_wodist)  #   # [N,32+3]

    color = pc.get_color_mlp(feat)   
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3]) # [mask]


    # The [:,:3] controls the step size of offset. The [:,3:] serves as the base scale for neural gaussian's shape, which means the cov MLP learn a residual scales.
    scale_rot = pc.get_cov_mlp(feat) 
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]

    # offsets
    # grid_offsets =  pc._offset[visible_mask].view([-1,3])  #  [N,10,3] grid_offsets =  grid_offsets[pc.dynamic_mask[visible_mask]]  
    offsets =  pc.get_offset_mlp(feat).view([-1,3])   # pc._offset[pc.dynamic_mask]  #  [N,10,3]  
    grid_scaling = pc.get_scaling[cur_dy_mask] # [N,6]  grid_scaling = grid_scaling[pc.dynamic_mask[visible_mask]]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets  ], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets  = masked.split([6, 3, 3, 7, 3], dim = -1 )



    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])



    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]  
    xyz = repeat_anchor + offsets


    if scaling.shape[0]==0:
        pass

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, 
    else:
        return xyz, color, opacity, scaling, rot


def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, opt_thro = 0.0, stage="fine", scaling_modifier = 1.0, iteration = 30000 , retain_grad=False , render_anchor=False, visible_mask=None, clip_id=0, render_mode=False, primitive_type='3dgs'):
    if primitive_type == '3dgs':
        return render_3dgs(viewpoint_camera, pc, pipe, bg_color, opt_thro, stage, scaling_modifier, iteration, retain_grad, render_anchor, visible_mask, clip_id, render_mode=render_mode)
    elif primitive_type == '2dgs':
        return render_2dgs(viewpoint_camera, pc, pipe, bg_color, opt_thro, stage, scaling_modifier, iteration, retain_grad, render_anchor, visible_mask, clip_id, render_mode=render_mode)
    else:
        assert False, f"Current codes don't support primivie_tpye: {primitive_type} "

def prefilter_voxel(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, clip_id=None, timestamp=None, primitive_type='3dgs'):
    if primitive_type == '3dgs':
        return prefilter_voxel_3dgs(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, clip_id, timestamp)
    elif primitive_type == '2dgs':
        return prefilter_voxel_2dgs(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, clip_id, timestamp)
    else:
        assert False, f"Current codes don't support primivie_type: {primitive_type} "

def render_3dgs(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, opt_thro = 0.0, stage="coarse", scaling_modifier = 1.0, iteration = 30000 , retain_grad=False , render_anchor=False, visible_mask=None, clip_id=0, render_mode=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if iteration  >= 10000 and iteration <= 20000:
        opt_thro = (iteration / 1000) *0.001 - 0.01
    elif iteration > 20000:
        opt_thro = 0.01
        
    opt_thro = 0.0
    if stage == "fine":
        xyz, color, opacity, scaling, rot, neural_opacity, mask = \
                                generate_full_neural_gaussians(viewpoint_camera, pc, visible_mask=visible_mask, is_training=is_training, opt_thro=opt_thro, clip_id=clip_id, render_mode=render_mode)
    elif stage == "coarse":
        xyz, color, opacity, scaling, rot, neural_opacity, mask = \
                                generate_coarse_neural_gaussians(viewpoint_camera, pc, visible_mask=visible_mask, is_training=is_training, opt_thro=opt_thro)
    else:
        dy_xyz, dy_color, dy_opacity, dy_scaling, dy_rot, neural_opacity, mask = \
                                dynamic_neural_gaussians(True, viewpoint_camera, pc, visible_mask=visible_mask, is_training=is_training,  opt_thro=opt_thro)
        sta_xyz, sta_color, sta_opacity, sta_scaling, sta_rot, neural_opacity, mask = \
                                dynamic_neural_gaussians(True, viewpoint_camera, pc, visible_mask=visible_mask, is_training=is_training,  opt_thro=opt_thro)
        xyz, color, opacity, scaling, rot = torch.cat([dy_xyz,sta_xyz]), torch.cat([dy_color,sta_color]), torch.cat([dy_opacity,sta_opacity]), \
                                torch.cat([dy_scaling,sta_scaling]), torch.cat([dy_rot,sta_rot])


    if iteration % 1000 == 0 and iteration % 5000 != 0:
        print(f"Temporal Gaussian points number: {xyz.shape[0]}, total seeds: {pc.get_anchor.shape[0]}")




    if render_anchor=="all":
        
        # xyz = pc.get_anchor[visible_mask]   # + pc._offset[visible_mask] * pc._scaling[visible_mask][:,:3]
        # color =  torch.ones((xyz.shape)).cuda()
        # color[:,0] = 0.43
        # color[:,1] = 0.49
        # color[:,2] = 0.63

        opacity = 0.5 * torch.ones((xyz.shape)).cuda()
        scaling = 0.001 * torch.ones((xyz.shape)).cuda()
        # rot = rot[:xyz.shape[0]]
        # scaling_modifier = 0.3

    



    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
            # dy_dynamics.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) 
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=1,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    rendered_sta_image = None
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    from time import time

    start_time = time()
    rendered_image, radii  = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None
    )

    # if pipe.env_map_res:    # 这是在做什么？
    #     assert pc.env_map is not None
    #     R = 60
    #     rays_o, rays_d = viewpoint_camera.get_rays()
    #     delta = ((rays_o*rays_d).sum(-1))**2 - (rays_d**2).sum(-1)*((rays_o**2).sum(-1)-R**2)
    #     assert (delta > 0).all()
    #     t_inter = -(rays_o*rays_d).sum(-1)+torch.sqrt(delta)/(rays_d**2).sum(-1)
    #     xyz_inter = rays_o + rays_d * t_inter.unsqueeze(-1)
    #     tu = torch.atan2(xyz_inter[...,1:2], xyz_inter[...,0:1]) / (2 * torch.pi) + 0.5 # theta
    #     tv = torch.acos(xyz_inter[...,2:3] / R) / torch.pi
    #     texcoord = torch.cat([tu, tv], dim=-1) * 2 - 1
    #     bg_color_from_envmap = F.grid_sample(pc.env_map[None], texcoord[None])[0] # 3,H,W
    #     # mask2 = (0 < xyz_inter[...,0]) & (xyz_inter[...,1] > 0) # & (xyz_inter[...,2] > -19)
    #     rendered_image = rendered_image + (1 - alpha) * bg_color_from_envmap # * mask2[None]
    # end_time = time()
    # print(f'rasterizer time {end_time - start_time}')

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "render_sta_image": rendered_sta_image,
                "depth_map":  None,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "neural_points":xyz
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }


def prefilter_voxel_3dgs(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, clip_id=None, timestamp=None, render_mode=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """


    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=1,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:  # false
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling     # TODO: BUG 这里的scales和rotations是通过get获取的，我们并没有写这个代码
        rotations = pc.get_rotation  # [N,4]

    radii_pure = rasterizer.visible_filter(means3D = means3D,  # 判断是否可见？
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp )

    return radii_pure > 0
    


def render_2dgs(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, opt_thro = 0.0, stage="coarse", scaling_modifier = 1.0, iteration = 30000 , retain_grad=False , render_anchor=False, visible_mask=None, clip_id=0, render_mode=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    import gsplat

    from gsplat.cuda._wrapper import fully_fused_projection_2dgs
    is_training = pc.get_color_mlp.training

    if iteration  >= 10000 and iteration <= 20000:
        opt_thro = (iteration / 1000) *0.001 - 0.01
    elif iteration > 20000:
        opt_thro = 0.01
        
    opt_thro = 0.0
    if stage == "fine":
        xyz, color, opacity, scaling, rot, neural_opacity, mask = \
                                generate_full_neural_gaussians(viewpoint_camera, pc, visible_mask=visible_mask, is_training=is_training, opt_thro=opt_thro, clip_id=clip_id, render_mode=render_mode)
    elif stage == "coarse":
        xyz, color, opacity, scaling, rot, neural_opacity, mask = \
                                generate_coarse_neural_gaussians(viewpoint_camera, pc, visible_mask=visible_mask, is_training=is_training, opt_thro=opt_thro)
    else:
        dy_xyz, dy_color, dy_opacity, dy_scaling, dy_rot, neural_opacity, mask = \
                                dynamic_neural_gaussians(True, viewpoint_camera, pc, visible_mask=visible_mask, is_training=is_training,  opt_thro=opt_thro)
        sta_xyz, sta_color, sta_opacity, sta_scaling, sta_rot, neural_opacity, mask = \
                                dynamic_neural_gaussians(True, viewpoint_camera, pc, visible_mask=visible_mask, is_training=is_training,  opt_thro=opt_thro)
        xyz, color, opacity, scaling, rot = torch.cat([dy_xyz,sta_xyz]), torch.cat([dy_color,sta_color]), torch.cat([dy_opacity,sta_opacity]), \
                                torch.cat([dy_scaling,sta_scaling]), torch.cat([dy_rot,sta_rot])


    if iteration % 1000 == 0 and iteration % 5000 != 0:
        print(f"Temporal Gaussian points number: {xyz.shape[0]}, total seeds: {pc.get_anchor.shape[0]}")




    if render_anchor=="all":
        
        # xyz = pc.get_anchor[visible_mask]   # + pc._offset[visible_mask] * pc._scaling[visible_mask][:,:3]
        # color =  torch.ones((xyz.shape)).cuda()
        # color[:,0] = 0.43
        # color[:,1] = 0.49
        # color[:,2] = 0.63

        opacity = 0.5 * torch.ones((xyz.shape)).cuda()
        scaling = 0.001 * torch.ones((xyz.shape)).cuda()
        # rot = rot[:xyz.shape[0]]
        # scaling_modifier = 0.3

    



    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
            # dy_dynamics.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) 
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)

    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)

    render_colors, \
    render_alphas, \
    render_normals, \
    render_normals_from_depth, \
    render_distort, \
    render_median, \
    info = \
    gsplat.rasterization_2dgs(
        means=xyz,  # [N, 3] 
        quats=rot,  # [N, 4] 
        scales=scaling,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=color, # 这个color传入的是球谐？OctreeGS中传入的是球谐 （我们是传入颜色的）
        viewmats=viewmat[None].cuda(),  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]  
        backgrounds=bg_color[None],   #backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=None,
        render_mode="RGB+ED",
        near_plane=float(os.environ.get('near_plane', 0.01)),
        depth_mode='expected',
        absgrad=bool(os.environ.get('absgrad', False))
    )

    if render_colors.shape[-1] == 4:
        colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
        depth = depths[0].permute(2, 0, 1)
    else:
        colors = render_colors
        depth = None

    rendered_image = colors[0].permute(2, 0, 1)
    # 他这里返回的是两个值 我觉得直接取一个最大的即可
    radii = info["radii"].squeeze(0) # [N,]
    radii, _ = torch.max(radii, dim=1)
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "render_sta_image": None,
                "depth_map":  None,
                "viewspace_points": info["means2d"],
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "neural_points":xyz,
                'rend_alpha': render_alphas,
                'rend_normal': render_normals, # 高斯的normal期望（这个初始是不知道的） 用rend_normal和surf_normal，其实基本等价于surf_normal是GT
                'rend_dist': render_distort,
                'surf_depth': depth,  # depth应该是期望深度（不确定） 
                'surf_normal': render_normals_from_depth,   # 深度期望得到的法向量 （因为我们初始化点够多，所以surf_depth和surf_normal其实基本是对的  
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }
    

    
# TODO： 这里应该过一个时间mask的
def prefilter_voxel_2dgs(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, clip_id=None, timestamp=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    import gsplat

    from gsplat.cuda._wrapper import fully_fused_projection_2dgs
    means = pc.get_anchor
    scales = pc.get_scaling[:, :3]
    quats = pc.get_rotation
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)

    Ks = torch.tensor([
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],device="cuda",)[None]
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None].cuda()

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    densifications = (
        torch.zeros((C, N, 2), dtype=means.dtype, device="cuda")
    )
    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        # densifications,   # 这个OctreeGS里是有的 不知道有什么用
        Ks,
        int(viewpoint_camera.image_width),
        int(viewpoint_camera.image_height),
        0.3, # eps2d=0.3
        float(os.environ.get('near_plane', 0.01)),
        1e10, # far_plane=1e10
        0.0, # radius_clip=0.0
        False, # packed=False
        False, # sparse_grad=False
    )
    
    # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
    radii, means2d, depths, conics, compensations = proj_results
    camera_ids, gaussian_ids = None, None
    

    # QUESTION: 是否要squeeze

    mask1 = radii.squeeze(0) > 0
    mask2 = radii.squeeze(0) < 1600
    
    # print(torch.logical_and(mask1, mask2).all(dim=1).shape)
    return torch.logical_and(mask1, mask2).all(dim=1)

 