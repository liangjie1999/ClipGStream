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
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import imageio
import numpy as np
import torch

import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams ,ModelHiddenParams
from scene import ReferenceClipScene, ReferenceClipGaussianModel, SourceClipScene, SourceClipGaussianModel
from gaussian_renderer import prefilter_voxel, render
from time import time
# import torch.multiprocessing as mp
import threading
import concurrent.futures
from PIL import Image

import sys

def multithread_write(image_list, path, render_view_id, global_start_frames):
    os.makedirs(os.path.join(path, str(render_view_id)), exist_ok=True)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, str(render_view_id), '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, global_start_frames + index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
import matplotlib.pyplot as plt
import numpy as np

def draw_img(img, idx, title=""):
    # Flatten and convert tensor to numpy array
    ldr = img.detach().cpu().numpy().flatten()
    ldr = ldr[ldr>0.02]
    zero_array = np.zeros(140000)  # 创建一个包含 58000 个 0 的数组
    ldr = np.concatenate((ldr, zero_array))  # 连接原始数组和 0 值数组
    # Define bins for smaller and larger values
    bins = np.concatenate(([0, 1], np.linspace(1, ldr.max(), 49)))

    plt.figure(figsize=(10, 4))  # 保持宽度为 10，减小高度为 3

    # Plot the histogram for the entire range
    counts, bins, patches = plt.hist(ldr, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

    # If the maximum count is very large, plot separately for small and large values
    if counts.max() > 1000:
        plt.figure(figsize=(10, 4))
        counts, bins, patches = plt.hist(ldr[ldr < 100], bins=50, color='skyblue', edgecolor='black', alpha=0.7)

    # Annotate counts
    i = 0
    for count, bin_edge in zip(counts, bins[:-1]):
        if i == 1 or i ==21 or i == 31:
            if count > 0:
                plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, f'{int(count)}', 
                            ha='center', va='bottom', fontsize=10, color='black')
        i+=1

    # Set labels and title
    plt.title(title, fontsize=20)
    plt.xlabel('Dynamic Residual Feature Value', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Set custom y-axis ticks
    plt.yticks(range(0, 150001, 30000))

    # Use scientific notation for y-axis
    ax = plt.gca()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.yaxis.set_tick_params(pad=15)
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(f"distribution_{idx}.png", format='png')
    plt.savefig(f"distribution_{idx}.pdf", format='pdf')
    plt.close()

def show_image(image,name):
    show_image = Image.fromarray(np.transpose(np.array(image.detach().clamp(0,1).cpu() * 255 ).astype(np.uint8), (1, 2, 0)))
    show_image.save(f'test_images/rgb_{name}.jpg')

def render_set_virtual(opt,model_path, name, iteration, views, gaussians, pipeline, background, cam_type, frames_start_end):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    render_view_ids = [0]

    global_start_frames = frames_start_end[0]
    global_end_frames = frames_start_end[1]
    for render_view_id in render_view_ids:
        render_images = []
        gt_list = []
        render_list = []
        # breakpoint()
        print("point nums:",gaussians._anchor.shape[0])
        all_time = 0
        per_view_frames = len(views)
        
        start_frame = render_view_id * per_view_frames

        # end_frame = start_frame + min(per_view_frames, clip_size * (iteration // per_clip_iter))
        end_frame = (render_view_id + 1) * per_view_frames

        # 这里的start_frame end_frame是在当前加载好数据里的索引 也就是一个局部相对索引
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            # TODO：这里的clip_id就是根据idx做的 那就是不支持不从0开始渲染的
            clip_id = idx // clip_size

            print('clip_id', clip_id)

            time1 = time()

            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, clip_id=0)  # 判断是否可见 
            retain_grad = (iteration < opt.update_until and iteration >= 0)
            # rendering = render(view, gaussians, pipeline, background, stage="fine", render_anchor="all", visible_mask=voxel_visible_mask, retain_grad=retain_grad)["render"]
            rendering = render(view, gaussians, pipeline, background, stage="fine", visible_mask = voxel_visible_mask, retain_grad=retain_grad, clip_id=0)["render"]

           
            time2 = time()
            
            all_time += (time2-time1)

            render_images.append(to8b(rendering).transpose(1,2,0))

            render_list.append(rendering)
            if name in ["train", "test"]:
                if cam_type != "PanopticSports":
                    gt = view.original_image[0:3, :, :]
                else:
                    gt  = view['image'].cuda()
                gt_list.append(gt)

        # TODO：这里的输出路径也需要根据视点的不同去保存一下吧
        time2=time()
        print("FPS:",(end_frame - start_frame)/all_time)

        print("writing training images.") 
        multithread_write(gt_list, gts_path, render_view_id, global_start_frames) 
        print("writing rendering images.") 
        multithread_write(render_list, render_path, render_view_id, global_start_frames)   # TODO： 这里保存图片的时候也要注意图片名称的问题，（视角/时间）
                                                      # 这里的start_frame end_frame 不要这么写 因为这不是时序上的名称 正确的是 视角号 + start_frame

        os.makedirs(os.path.join(model_path, name, "ours_{}".format(iteration), 'videos', os.environ.get('virtual_frame_interval', '10'), os.environ.get('render_kind', '3d'), str(render_view_id)), exist_ok=True) 
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'videos', os.environ.get('virtual_frame_interval', '10'), os.environ.get('render_kind', '3d'), str(render_view_id), f'{global_start_frames}_{global_end_frames}.mp4'), render_images,fps=25)


def render_set(opt,model_path, name, iteration, views, gaussians, pipeline, background, cam_type, frames_start_end):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    render_view_ids = [0, 1, 2, 3]
    render_view_ids = [0]

    global_start_frames = frames_start_end[0]
    for render_view_id in render_view_ids:
        render_images = []
        gt_list = []
        render_list = []
        # breakpoint()
        print("point nums:",gaussians._anchor.shape[0])
        all_time = 0
        per_view_frames = frames_start_end[1] - frames_start_end[0]
        
        start_frame = render_view_id * per_view_frames

        end_frame = start_frame + min(per_view_frames, opt.project_total_frames)

        for idx, view in enumerate(tqdm(views[start_frame:end_frame], desc="Rendering progress")):
            clip_id = idx // clip_size

            print('clip_id', clip_id)

            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, clip_id=clip_id)  # 判断是否可见 
            retain_grad = (iteration < opt.update_until and iteration >= 0)
            # rendering = render(view, gaussians, pipeline, background, stage="fine", render_anchor="all", visible_mask=voxel_visible_mask, retain_grad=retain_grad)["render"]
            time1 = time()

            rendering = render(view, gaussians, pipeline, background, stage="fine", visible_mask = voxel_visible_mask, retain_grad=retain_grad, clip_id=clip_id, render_mode=True)["render"]
            time2 = time()

            print(f"render time: {time2 - time1}")
            all_time += (time2-time1)

            render_images.append(to8b(rendering).transpose(1,2,0))

            if idx == 1:
                cv2.imwrite(f'./show/debug.jpg', to8b(rendering).transpose(1,2,0))

            render_list.append(rendering)
            if name in ["train", "test"]:
                if cam_type != "PanopticSports":
                    gt = view.original_image[0:3, :, :]
                else:
                    gt  = view['image'].cuda()
                gt_list.append(gt)

        time2=time()
        print("FPS:",(end_frame - start_frame)/all_time)

        print("writing training images.") 
        multithread_write(gt_list, gts_path, render_view_id, global_start_frames) 
        print("writing rendering images.") 
        multithread_write(render_list, render_path, render_view_id, global_start_frames)   # TODO： 这里保存图片的时候也要注意图片名称的问题，（视角/时间）
                                                      # 这里的start_frame end_frame 不要这么写 因为这不是时序上的名称 正确的是 视角号 + start_frame

        

def render_sets( opt , hyper, dataset : ModelParams, frames_start_end, iteration : int, pipeline : PipelineParams,  skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        if frames_start_end[0] == 0:
            # reference clip
        
            gaussians = ReferenceClipGaussianModel(hyper, opt,dataset.feat_dim, 10, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                                dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, frames_start_end=frames_start_end, iteration=iteration, model_path=dataset.model_path)
            # iteration = 7000
            scene = ReferenceClipScene(dataset, gaussians, frames_start_end = frames_start_end, load_iteration=iteration, shuffle=False)
        else:
            ref_gaussians = ReferenceClipGaussianModel(hyper, opt, dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                                    dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, loaded_for_source=True, model_path=dataset.model_path)

            clip_id = frames_start_end[0] // opt.clip_size

            gaussians = SourceClipGaussianModel(ref_gaussians, hyper, opt, dataset.feat_dim, 10, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                                    dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, frames_start_end=frames_start_end, clip_id=clip_id, model_path=dataset.model_path)
                        
            scene = SourceClipScene(dataset, gaussians, frames_start_end, load_iteration=iteration, clip_id=clip_id)

        cam_type=scene.dataset_type
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]

        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(opt,dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type, frames_start_end=frames_start_end)
        if not skip_test:
            render_set(opt,dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type, frames_start_end=frames_start_end)
        if not skip_video:
            render_set_virtual(opt,dataset.model_path,"video2",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type, frames_start_end=frames_start_end)
        
        if not os.environ.get('skip_blender', True):
            output_name = os.environ.get('blender_name', 'main')
            output_name = f'video_blender/{output_name}'

            render_set_virtual(opt,dataset.model_path,output_name,scene.loaded_iter,scene.getBlenderCameras(), gaussians,pipeline,background,cam_type, frames_start_end=frames_start_end)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, nargs='+')
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--frames_start_end", type=int, nargs=2, default=[0, 300], help="Start and end frames")

    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.general_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # enable logging
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # total_frames = args.frames_start_end[1] - args.frames_start_end[0]

    start = args.frames_start_end[0]
    end = args.frames_start_end[1]
    clip_size = args.clip_size

    import math
    clip_nums = math.ceil((end - start) / clip_size)

    start_clip_id = start // clip_size

    for clip_id in range(start_clip_id, start_clip_id + clip_nums):
        start_end = (clip_id * clip_size, (clip_id + 1) * clip_size)
        args.frames_start_end = start_end

        for i_iteration in args.iteration:
            render_sets(op.extract(args), hp.extract(args),   model.extract(args), start_end, i_iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)
        