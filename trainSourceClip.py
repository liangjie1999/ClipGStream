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

import os
# os.environ['CUDA_VISIBLE_DEVICES']="7"

import numpy as np

from utils.timer import Timer

from torch.utils.data import DataLoader
from utils.loader_utils import FineSampler, get_stamp_list
import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim, LikelihoodLoss, factor_loss
from gaussian_renderer import prefilter_voxel, render , network_gui
import sys
from scene import SourceClipScene, SourceClipGaussianModel, ReferenceClipGaussianModel

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams,ModelHiddenParams
import matplotlib.pyplot as plt
from PIL import Image
#
from utils.visualize_utils import tensor2image
import cv2
# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

import math

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def draw_img(img, idx):
    ldr = img.detach().cpu()
    img_tan = ldr.reshape(-1)
    plt.hist(np.array(img_tan), bins=50)
    plt.xlabel('Max_value',fontsize=18)
    plt.ylabel('Images',fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"ldr_{idx}.png",format='png')
    plt.close()

def show_depth(image,name):
    depth_max = torch.max(image)
    depth_min = torch.min(image)
    image = (image - depth_min) / (depth_max - depth_min + 0.0001)
    show_image = Image.fromarray(np.transpose(np.array(image.tile(3,1,1).detach().clamp(0,1).cpu() * 255 ).astype(np.uint8), (1, 2, 0)))
    show_image.save(f'test_images/rgb_{name}.png')



def show_image(image,name):
    from torchvision.utils import save_image
    save_image(image.detach().clamp(0,1), f'test_images/rgb_{name}.jpg')
    
    
def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')

def save_launcher_shell_path(save_folder):
    # 在bash脚本中 export bash_path="$(realpath "${BASH_SOURCE[0]}")" 
    import os

    bash_path = os.environ.get('bash_path', '')

    if bash_path == '':
        return 
        assert False, "请从环境变量中传入bash_path（启动命令的地址） "

    os.makedirs(save_folder, exist_ok=True)
    os.system(f'cp {bash_path} {save_folder}')

    return bash_path

def save_run_codes(save_folder):
    import os
    from pathlib import Path

    os.makedirs(f'{save_folder}/code', exist_ok=True)

    # 1. 定义你想要包含的文件或文件夹模式
    include_patterns = [
        '*.py',              # 所有Python文件
        'arguments/***',              # src目录及其内容      ***是保证下面的所有内容都加入，如果只有arguements 不会复制子文件（就一个目录）
        'gaussian_renderer/***',          # configs目录及其内容
        'scene/***',            # utils目录及其内容
        'utils/***',  # 依赖文件
        # 添加其他你需要包含的文件或文件夹...
    ]

    # 2. 获取当前脚本所在目录的绝对路径
    cur_folder = Path(__file__).resolve().parent

    # 3. 构建rsync命令
    # 基本命令: -av 表示归档模式和详细输出
    cmd = f'rsync -av '
    
    # 4. 关键: 先排除所有文件 --
    cmd += f""  # 注意: 这里用单引号包裹通配符*，防止shell扩展

    # 5. 然后，为每一个包含模式添加include规则
    #    顺序很重要，通常先包含目录结构，再包含具体文件
    for pattern in include_patterns:
        cmd += f"--include='{pattern}' "  # 注意: 用单引号包裹模式

    # 6. 添加源路径和目标路径

    # 命令是按顺序匹配的，后面的不重要了，要先加入文件 最后排除所有
    cmd += f"--exclude='*' {cur_folder}/ {save_folder}/code"  # 注意源路径结尾的斜杠，它影响同步行为
    
    os.system(cmd)
    # 7. 打印命令并执行 (建议先打印检查，确认无误后再执行)
    print(f"Generated rsync command:\n{cmd}")

def training(dataset, hyper, opt, pipe, frames_start_end, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname,args, load_iteration=None):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    ref_gaussians = ReferenceClipGaussianModel(hyper, opt, dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                            dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, loaded_for_source=True, model_path=dataset.model_path)

    clip_id = frames_start_end[0] // opt.clip_size

    gaussians = SourceClipGaussianModel(ref_gaussians, hyper, opt, dataset.feat_dim, 10, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, frames_start_end=frames_start_end, clip_id=clip_id, model_path=dataset.model_path)
    
    dataset.model_path = args.model_path
    timer = Timer()
    # 这里就可以控制恢复训练 直接传入load_iteration就可以
    scene = SourceClipScene(dataset, gaussians, frames_start_end, load_iteration=load_iteration)
    timer.start()

    scene_reconstruction(dataset, hyper, opt, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer, args, load_iteration=load_iteration, frames_start_end=frames_start_end)
    

    
def scene_reconstruction(dataset, hyper, opt,  pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer, args, load_iteration=None, frames_start_end=[0, 250]):
    if load_iteration and checkpoint:
        assert False, "load_iteration 和 checkpoint只能传入一个"
    
    first_iter = 0

    gaussians.training_setup(opt,stage)

    writer_path = os.path.join(scene.model_path, 'writer')
    os.makedirs(writer_path,exist_ok=True)
    writer = SummaryWriter(log_dir = writer_path)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if load_iteration:
        first_iter = load_iteration # 模型参数应该在Scene阶段就完成注入了（和渲染走的一个流程）

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, train_iter), desc="Training progress")
    first_iter += 1
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    if not viewpoint_stack and not opt.dataloader: # opt.dataloader True
        # dnerf's branch 
        import copy
        train_cams = scene.getTrainCameras()
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)

    if opt.dataloader: # 走的是这个
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=1,sampler=sampler,num_workers=16,collate_fn=list)
            random_loader = False
        else:   # 走的是这个 
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=1,shuffle=True,num_workers=16,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)

    if stage == "coarse" and opt.zerostamp_init : #fine 不走这个
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack, 0) # 20
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False #

    last_iteration = 0
    cur_index = 0

    frame_counter = {}

    #######################
    cur_clip = 0
    clip_size = opt.clip_size # 20帧
    per_clip_iter = opt.iterations
    total_frames = frames_start_end[1] - frames_start_end[0]

    clip_nums = math.ceil(total_frames / clip_size)

    if True:
        cur_clip = frames_start_end[0] // int(opt.clip_size)

        viewpoint_stack = scene.getTrainCameras() # 这里感觉可以创建多个Dataset，直接根据id获取就行了
        viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
        random_loader = True    # 这个random_loader是用来做什么的？
        loader = iter(viewpoint_stack_loader)

        for iteration in range(0, per_clip_iter):
            iter_start.record()
            if  opt.dataloader and not load_in_memory: # 走的这个
                try:
                    viewpoint_cams = next(loader) # 这里其实就是根据iteration 修改一下获取到的viewpoint_stack就行
                except StopIteration:   # 执行时机是？最后一下？还是倒数第二下？
                    print("reset dataloader into random dataloader.")
                    if not random_loader:
                        viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                        random_loader = True
                    loader = iter(viewpoint_stack_loader) 

            
            gaussians.update_learning_rate(iteration) 

            viewpoint_cam = viewpoint_cams[0]   # 我们要先看loader的返回值 知道总共有哪些数据，然后要对齐所有数据
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background, clip_id=cur_clip)  # 判断是否可见 
            retain_grad = (iteration < opt.update_until and iteration >= 0)

            # TODO: 这里应该还要根据当前的gop去筛选出点云 时间窗口 分层结构就不需要了
            render_pkg = render( viewpoint_cam, gaussians, pipe, background, iteration=iteration, stage=stage, visible_mask=voxel_visible_mask, retain_grad=retain_grad, clip_id=cur_clip, primitive_type=opt.primitive_type)
            image, depth_map, viewspace_point_tensor, visibility_filter, offset_selection_mask, scaling, opacity, neural_points = \
                    render_pkg["render"], render_pkg["depth_map"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], \
                        render_pkg["scaling"], render_pkg["neural_opacity"], render_pkg["neural_points"]

            gt_image = viewpoint_cam.original_image.cuda()
            # TODO: 支持可视化图片（这里可视化了图片 然后再空训练几轮（从0开始）看看是不是从0开始训了相当于）
            Ll1 = l1_loss(image, gt_image)   

            ssim_loss = (1.0 - ssim(image, gt_image)[0])  

            scaling_reg =  scaling.prod(dim=1).mean()    
            # PIL 2 img
            psnr_ = psnr(image, gt_image).mean().double() 
            
            if os.environ.get('debug', True):
                if iteration % 1000 == 1:
                    # 怎么第一张上来渲染的是不对？（按理来说不可能吧，像是没有加载进来数据一样的感觉）
                    os.makedirs(f'{args.model_path}/debug/{cur_clip}', exist_ok=True)   
                    display_image = tensor2image(image)
                    display_image[:, :, :3] = display_image[:, :, [2, 1, 0]]
                    # cv2接收的是bgr （tensor2image是将输出转换成了rgb torch存的本身也是bgr）
                    cv2.imwrite(f'{args.model_path}/debug/{cur_clip}/{iteration}.jpg', display_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss  + 0.01 * scaling_reg 

            # 这个也得改改， 这里有一个 bug 
            if stage == "fine" and hyper.time_smoothness_weight != 0 and not opt.hash :  
                # tv_loss = 0
                if True:
                    tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight, viewpoint_cam.time)
                    loss += tv_loss
            
            primitive_type = opt.primitive_type
            if primitive_type == '2dgs':
                lambda_normal = opt.lambda_normal if iteration > 2000 else 0.0
                lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

                rend_dist = render_pkg["rend_dist"]
                rend_normal  = render_pkg['rend_normal']
                surf_normal = render_pkg['surf_normal']
                normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
                normal_loss = lambda_normal * (normal_error).mean()
                dist_loss = lambda_dist * (rend_dist).mean()

                loss += dist_loss
                loss += normal_loss
            
            loss.backward()

            for param in gaussians.dynamic_module.parameters():  
                if param.grad is not None: 
                    if torch.isnan(param.grad).any():
                        pass
                    param.grad.nan_to_num_()     

                    torch.clamp_(param.grad, -1000, 1000)
                    

            if torch.isnan(loss).any():
                print("loss is nan,end training, reexecv program now.")
                return 
                # os.execv(sys.executable, [sys.executable] + sys.argv)

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log  
                total_point = gaussians.get_anchor.shape[0]  
                if iteration % 10 == 0 :  
                    writer.add_scalar("Loss", loss.item(), iteration)  # add scale
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}" ,  
                                                "psnr": f"{psnr_:.{2}f}" ,  
                                                "point":f"{total_point}"})  
                    progress_bar.update(10)

                if iteration == train_iter:
                    progress_bar.close()

                if stage == "fine" :
                    training_report(args, gaussians, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), \
                                    testing_iterations, scene, render, [ pipe , background  ], stage, scene.dataset_type, logger)

                global_iteration = iteration + cur_clip * per_clip_iter

                if iteration < opt.update_until and iteration > opt.start_stat : #   and stage == "fine":
                    # add statis
                    gaussians.training_statis(  neural_points ,viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                    # densification
                    if iteration >= opt.update_from  :  # and iteration % opt.update_interval == 0 :  # 1500 开始致密化
                        gaussians.adjust_anchor(iteration, check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
                


                if iteration % 100 == 0 :
                    torch.cuda.empty_cache()

                elif iteration == opt.update_until:  # 15000 删除缓存
                    del gaussians.opacity_accum
                    del gaussians.offset_gradient_accum
                    del gaussians.offset_denom
                    del gaussians.grad_max 
                    del gaussians.grad_max_points
                    torch.cuda.empty_cache()
                        
                # Optimizer step 
                # TODO: train_iter就是训练总轮数 我们还要保证这个足够大 大于 per_clip_iteration * clip_nums
                if iteration < train_iter : 
                    gaussians.optimizer.step()
                    gaussians.dy_optimizer.step()
                    gaussians.dy_optimizer.zero_grad(set_to_none = True)
                    gaussians.optimizer.zero_grad(set_to_none = True)

        global_iteration = (cur_clip + 1) * per_clip_iter # 20000 40000
        logger.info("\n[ITER {}] Saving Gaussians".format(global_iteration))
        
        scene.save(global_iteration)


    writer.close()



def prepare_output_and_logger(expname):    
    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        os.makedirs(args.model_path+"/logs/", exist_ok = True)
        tb_writer = SummaryWriter(args.model_path +"/logs/" )
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(args , gaussians, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, \
                    scene : SourceClipScene, renderFunc, renderArgs, stage, dataset_type, logger):

    # if tb_writer:
    #     tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
    #     tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
    #     tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
    
    # Report test and samples of training set

    if iteration in testing_iterations:
        # torch.cuda.empty_cache()
        # 
        validation_configs = (
            # {'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
            {'name': 'test', 'cameras' : [scene.getTestCameras()[idx] for idx in range(len(scene.getTestCameras())) ] },
            # {'name': 'test', 'cameras' : [scene.getTestCameras()[0]  ] },
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, gaussians, renderArgs[0], renderArgs[1])  # 判断是否可见 
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, iteration = iteration, visible_mask = voxel_visible_mask, 
                                                   retain_grad = True, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if config['name'] == 'test' and iteration > 0:
                        os.makedirs(f'{gaussians.model_path}/renders_middle/{gaussians.clip_id}/{iteration}/', exist_ok=True)
                        torchvision.utils.save_image(
                            image,
                            f'{gaussians.model_path}/renders_middle/{gaussians.clip_id}/{iteration}/{idx:0>5}.jpg'
                        )
                    # try: 
                    #     if tb_writer and (idx < 5): 
                    #         tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration) 
                    #         if iteration == testing_iterations[0]: 
                    #             tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    # except: 
                    #     pass 
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    psnr_ = psnr(image, gt_image).mean().double()                    
                    psnr_test += psnr_

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                # print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                logger.info("[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, config['name'], l1_test, psnr_test))
                
                # print("sh feature",scene.gaussians.get_features.shape)
        #         if tb_writer:
        #             tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
        #             tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)

        # torch.cuda.empty_cache()


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        ref_gaussians = ReferenceClipGaussianModel(hyper, opt, dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, loaded_for_source=True)

        gaussians = SourceClipGaussianModel(ref_gaussians, hyper, opt, dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = SourceClipScene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    

def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


def warm_up(lp, opt,pp, test_iterations,save_iterations, checkpoint_iterations, start_checkpoint, debug_from, model_path, args):
    pass

# 先写仅支持训练一个clip的代码，然后再丰富成支持多个clip的代码
# 支持多个clip训练的代码直接写在__main__里就行了，多次调用即可
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False) # 250 * 30 = 7500 7500 
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000] + [5000 * i for i in range(1, 4)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--restore_iteration", type=int, default=None)

    parser.add_argument("--frames_start_end", type=int, nargs=2, default=[0, 10], help="Start and end frames")
    # parser.add_argument("--expname", type=str, default = "")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    save_launcher_shell_path(args.model_path)
    save_run_codes(args.model_path)

    if args.configs:
        import mmcv
        from utils.general_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)

    args.iterations = args.iterations + 1

    logger.info(f'args: {args}')

    # if args.gpu != '-1':
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    #     os.system("echo $CUDA_VISIBLE_DEVICES")
    #     logger.info(f'using GPU {args.gpu}')
    # try:
    #     saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    # except:
    #     logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"ClipGStream-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # total_frames = args.frames_start_end[1] - args.frames_start_end[0]

    # per_clip_iter = int(os.environ.get('per_clip_iter', 10000))

    # for i in range(1, gop_num + 1):
    #     args.save_iterations.append(per_clip_iter * i)
    #     args.save_iterations.append((per_clip_iter // 2) * i)

    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.frames_start_end, args.test_iterations, args.save_iterations, \
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.model_path, args, load_iteration=args.restore_iteration)




