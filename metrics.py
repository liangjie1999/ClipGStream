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

from pathlib import Path
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'



from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim, msssim
import lpipsPyTorch as lp
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import sys
from arguments import ModelParams, PipelineParams, OptimizationParams
import imageio
import os
import numpy as np
import cv2
import csv
import torch  
from utils.image_utils import psnr
from utils.loss_utils import ssim
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'



tonemap = lambda x : (np.log(np.clip(x, 0, 1) * 5000 + 1 ) / np.log(5000 + 1)).astype(np.float32)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
# 展示LDR
def show_image(image,idx):
    show_image = Image.fromarray(to8b(image))
    show_image.save(f'rgb_{idx}.png')



def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open( os.path.join(renders_dir , fname))
        gt = Image.open(os.path.join(gt_dir , fname))
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate_(test_path, render_view_id=0):
    test_path = f'{test_path}/test/ours_315520/'
    # test_path_list = [os.path.join(test_path, "test", f) for f in sorted(os.listdir(os.path.join(test_path, "test"))) if 'ours' in f]
    # test_path = test_path_list[-1]
    print(f"Selected test model is: {test_path}")

    dssim2s = []
    dssim1s = []
    psnrs = []
    ssims = []
    lpipss = []
    gt_dir = os.path.join(test_path, "gt", str(render_view_id))
    renders_dir =os.path.join(test_path , "renders", str(render_view_id))



    renders, gts, image_names = readImages(renders_dir, gt_dir)
    csvfile = open(os.path.join(test_path, f'eval_ldr_{render_view_id}.csv'),"w") 
    writer = csv.writer(csvfile)



    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):

        dssim1 =  (1 - ssim(renders[idx], gts[idx])[0]) / 2
        dssim2  =  (1-  msssim(renders[idx], gts[idx])) / 2
        psnr_ = psnr(renders[idx], gts[idx])
        lpips_ =  psnr_ * 0 #  lpips(renders[idx], gts[idx], net_type='alex')

        dssim1s.append(dssim1)
        dssim2s.append(dssim2)
        psnrs.append(psnr_)
        lpipss.append(lpips_)  # this used vgg model


        writer.writerow([ idx , psnr_.item(), dssim1.item(), dssim2.item(), lpips_.item()])
    
    avg_ssim = torch.tensor(dssim1s).mean().item()
    avg_msssim = torch.tensor(dssim2s).mean().item()
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_lpips = torch.tensor(lpipss).mean().item()
    print(" Avg PSNR : {:>12.7f}".format(avg_psnr, ".5"))
    print(" Avg DSSIM1 : {:>12.7f}".format(avg_ssim, ".5"))
    print(" Avg DSSIM2 : {:>12.7f}".format(avg_msssim, ".5"))
    print(" Avg LPIPS: {:>12.7f}".format(avg_lpips, ".5"))
    print("")

    writer.writerow([ "avg" , avg_psnr, avg_ssim, avg_msssim, avg_lpips])

def evaluate_memory_efficient(test_path, iteration, render_view_id=0, batch_size=8, gt_root_dir=None):
    """
    分块加载计算图像质量指标，避免OOM
    
    参数:
        test_path: 测试结果路径
        render_view_id: 渲染视角ID
        batch_size: 批处理大小，根据GPU内存调整
    """
    test_path = f'{test_path}/test/ours_{iteration}/'
    print(f"Selected test model is: {test_path}")

    gt_dir = None

    if gt_root_dir == None:
        gt_dir = os.path.join(test_path, "gt", str(render_view_id))
    else:
        gt_dir = os.path.join(gt_root_dir, str(render_view_id))
    
    renders_dir = os.path.join(test_path, "renders", str(render_view_id))
    
    # 获取所有图像文件
    image_files = sorted([f for f in os.listdir(renders_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # print(image_files)
    image_idxs = [int(item.split('.')[0]) for item in image_files]
    
    # for idx in range(1, len(image_idxs)):
    #     if image_idxs[idx] - image_idxs[idx - 1] == 1:
    #         continue

    #     print(idx, image_idxs[idx], image_idxs[idx - 1])    
    # print(image_idxs)
    # assert(False)
    image_files = image_files[:330] + image_files[360:]
    print(image_files)
    total_images = len(image_files)
    
    print(f"找到 {total_images} 张图像，使用批处理大小: {batch_size}")
    
    # 初始化结果列表
    all_dssim1 = []
    all_dssim2 = []
    all_psnr = []
    all_lpips = []
    
    # 初始化LPIPS模型（如果需要）
    lpips_model = lp.LPIPS('vgg', '0.1').cuda()  # 或 'vgg'
    
    # 创建CSV文件
    csv_path = os.path.join(test_path, f'eval_ldr_{render_view_id}_all_batch.csv')
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(["index", "psnr", "dssim1", "dssim2", "lpips"])
        
        # 分批处理图像
        for batch_start in tqdm(range(0, total_images, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, total_images)
            batch_files = image_files[batch_start:batch_end]
            
            # 加载当前批次的渲染图和GT图
            renders_batch = []
            gts_batch = []
            
            for fname in batch_files:
                render_path = os.path.join(renders_dir, fname)
                gt_path = os.path.join(gt_dir, fname)
                
                # 加载图像并转换为Tensor
                render = Image.open(render_path)
                gt = Image.open(gt_path)
                target_size = gt.size  # 返回一个元组 (width, height)

                # 将render图片调整为gt的尺寸
                # 使用高质量的重采样算法（如LANCZOS）
                render = render.resize(target_size, Image.Resampling.LANCZOS)
                
                render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
                gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
                
                renders_batch.append(render_tensor)
                gts_batch.append(gt_tensor)
            
            # 计算当前批次的指标
            for idx, (render, gt) in enumerate(zip(renders_batch, gts_batch)):
                global_idx = batch_start + idx
                
                try:
                    # 计算各项指标
                    dssim1 = (1 - ssim(render, gt)[0]) / 2
                    dssim2 = (1 - msssim(render, gt)) / 2
                    psnr_val = psnr(render, gt)
                    
                    # LPIPS计算较慢且耗内存，可根据需要启用
                    # lpips_val = 0  # 默认不计算LPIPS
                    lpips_val = lpips_model(render, gt).item()  # 如果需要计算LPIPS
                    
                    # 保存结果
                    all_dssim1.append(dssim1.item())
                    all_dssim2.append(dssim2.item())
                    all_psnr.append(psnr_val.item())
                    all_lpips.append(lpips_val)
                    
                    # writing to csv
                    writer.writerow([global_idx, psnr_val.item(), dssim1.item(), dssim2.item(), lpips_val])
                    
                except Exception as e:
                    print(f"计算图像 {global_idx} ({batch_files[idx]}) 时出错: {e}")
                    # 写入错误标记
                    writer.writerow([global_idx, "ERROR", "ERROR", "ERROR", "ERROR"])
                
                finally:
                    # 释放当前图像占用的内存
                    del render, gt
                    torch.cuda.empty_cache()  # 清理GPU缓存[9,10](@ref)
            
            # 释放整个批次的内存
            del renders_batch, gts_batch
            torch.cuda.empty_cache()
    
    # 计算平均指标
    avg_psnr = np.mean(all_psnr) if all_psnr else 0
    avg_dssim1 = np.mean(all_dssim1) if all_dssim1 else 0
    avg_dssim2 = np.mean(all_dssim2) if all_dssim2 else 0
    avg_lpips = np.mean(all_lpips) if all_lpips else 0
    
    # 输出结果
    print("\n===== 最终结果 =====")
    print(f"评估图像数量: {len(all_psnr)}/{total_images}")
    print(" Avg PSNR : {:>12.7f}".format(avg_psnr))
    print(" Avg DSSIM1 : {:>12.7f}".format(avg_dssim1))
    print(" Avg DSSIM2 : {:>12.7f}".format(avg_dssim2))
    print(" Avg LPIPS : {:>12.7f}".format(avg_lpips))
    
    # 将平均值写入CSV
    with open(csv_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["avg", avg_psnr, avg_dssim1, avg_dssim2, avg_lpips])
    
    return {
        "psnr": avg_psnr,
        "dssim1": avg_dssim1,
        "dssim2": avg_dssim2,
        "lpips": avg_lpips
    }

def evaluate_memory_efficientTemporal(test_path, iteration, render_view_id=0, batch_size=8, gt_root_dir=None):
    """
    分块加载计算图像质量指标，避免OOM
    
    参数:
        test_path: 测试结果路径
        render_view_id: 渲染视角ID
        batch_size: 批处理大小，根据GPU内存调整
    """
    test_path = f'{test_path}/test/ours_{iteration}/'
    print(f"Selected test model is: {test_path}")

    gt_dir = None

    if gt_root_dir == None:
        gt_dir = os.path.join(test_path, "gt", str(render_view_id))
    else:
        gt_dir = os.path.join(gt_root_dir, str(render_view_id))
    
    renders_dir = os.path.join(test_path, "renders", str(render_view_id))
    
    # 获取所有图像文件
    image_files = sorted([f for f in os.listdir(renders_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    image_files = image_files[:50]

    # print(image_files)
    image_idxs = [int(item.split('.')[0]) for item in image_files]
    
    # for idx in range(1, len(image_idxs)):
    #     if image_idxs[idx] - image_idxs[idx - 1] == 1:
    #         continue

    #     print(idx, image_idxs[idx], image_idxs[idx - 1])    
    # print(image_idxs)
    # assert(False)
    total_images = len(image_files)
    
    print(f"找到 {total_images} 张图像，使用批处理大小: {batch_size}")
    
    # 初始化结果列表
    all_dssim1 = []
    all_dssim2 = []
    all_psnr = []
    all_lpips = []
    
    # 初始化LPIPS模型（如果需要）
    lpips_model = lp.LPIPS('vgg', '0.1').cuda()  # 或 'vgg'
    
    # 创建CSV文件
    csv_path = os.path.join(test_path, f'eval_ldr_{render_view_id}_all_batch_Temporal.csv')
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(["index", "psnr", "dssim1", "dssim2", "lpips"])
        
        # 分批处理图像
        for batch_start in tqdm(range(0, total_images, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, total_images)
            batch_files = image_files[batch_start:batch_end]
            
            # 加载当前批次的渲染图和GT图
            renders_batch = []
            gts_batch = []
            
            for fname in batch_files:
                render_path = os.path.join(renders_dir, fname)
                idx = int(fname.split('.')[0])
                gt_path = os.path.join(renders_dir, f'{idx+1:0>5}.png')
                
                # 加载图像并转换为Tensor
                render = Image.open(render_path)
                gt = Image.open(gt_path)
                target_size = gt.size  # 返回一个元组 (width, height)

                # 将render图片调整为gt的尺寸
                # 使用高质量的重采样算法（如LANCZOS）
                render = render.resize(target_size, Image.Resampling.LANCZOS)
                
                render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
                gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
                
                renders_batch.append(render_tensor)
                gts_batch.append(gt_tensor)
            
            # 计算当前批次的指标
            for idx, (render, gt) in enumerate(zip(renders_batch, gts_batch)):
                global_idx = batch_start + idx
                
                try:
                    # 计算各项指标
                    dssim1 = (1 - ssim(render, gt)[0]) / 2
                    dssim2 = (1 - msssim(render, gt)) / 2
                    psnr_val = psnr(render, gt)
                    
                    # LPIPS计算较慢且耗内存，可根据需要启用
                    lpips_val = 0  # 默认不计算LPIPS
                    # lpips_val = lpips_model(render, gt).item()  # 如果需要计算LPIPS
                    
                    # 保存结果
                    all_dssim1.append(dssim1.item())
                    all_dssim2.append(dssim2.item())
                    all_psnr.append(psnr_val.item())
                    all_lpips.append(lpips_val)
                    
                    # 写入CSV
                    writer.writerow([global_idx, psnr_val.item(), dssim1.item(), dssim2.item(), lpips_val])
                    
                except Exception as e:
                    print(f"计算图像 {global_idx} ({batch_files[idx]}) 时出错: {e}")
                    # 写入错误标记
                    writer.writerow([global_idx, "ERROR", "ERROR", "ERROR", "ERROR"])
                
                finally:
                    # 释放当前图像占用的内存
                    del render, gt
                    torch.cuda.empty_cache()  # 清理GPU缓存[9,10](@ref)
            
            # 释放整个批次的内存
            del renders_batch, gts_batch
            torch.cuda.empty_cache()
    
    # 计算平均指标
    avg_psnr = np.mean(all_psnr) if all_psnr else 0
    avg_dssim1 = np.mean(all_dssim1) if all_dssim1 else 0
    avg_dssim2 = np.mean(all_dssim2) if all_dssim2 else 0
    avg_lpips = np.mean(all_lpips) if all_lpips else 0
    
    # 输出结果
    print("\n===== 最终结果 =====")
    print(f"评估图像数量: {len(all_psnr)}/{total_images}")
    print(" Avg PSNR : {:>12.7f}".format(avg_psnr))
    print(" Avg DSSIM1 : {:>12.7f}".format(avg_dssim1))
    print(" Avg DSSIM2 : {:>12.7f}".format(avg_dssim2))
    print(" Avg LPIPS : {:>12.7f}".format(avg_lpips))
    
    # 将平均值写入CSV
    with open(csv_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["avg", avg_psnr, avg_dssim1, avg_dssim2, avg_lpips])
    
    return {
        "psnr": avg_psnr,
        "dssim1": avg_dssim1,
        "dssim2": avg_dssim2,
        "lpips": avg_lpips
    }


def evaluate_memory_efficient_static(test_path, iteration, render_view_id=0, batch_size=8, gt_root_dir=None):
    """
    分块加载计算图像质量指标，避免OOM
    
    参数:
        test_path: 测试结果路径
        render_view_id: 渲染视角ID
        batch_size: 批处理大小，根据GPU内存调整
    """
    test_path = f'{test_path}/test/ours_{iteration}/'
    print(f"Selected test model is: {test_path}")

    gt_dir = None

    if gt_root_dir == None:
        gt_dir = os.path.join(test_path, "gt", str(render_view_id))
    else:
        gt_dir = os.path.join(gt_root_dir)
    
    renders_dir = os.path.join(test_path, "renders", str(render_view_id))
    
    # 获取所有图像文件
    image_files = sorted([f for f in os.listdir(renders_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # print(image_files)
    image_idxs = [int(item.split('.')[0]) for item in image_files]
    
    # for idx in range(1, len(image_idxs)):
    #     if image_idxs[idx] - image_idxs[idx - 1] == 1:
    #         continue

    #     print(idx, image_idxs[idx], image_idxs[idx - 1])    
    # print(image_idxs)
    # assert(False)
    image_files = image_files[:330] + image_files[360:]
    print(image_files)
    total_images = len(image_files)
    
    print(f"找到 {total_images} 张图像，使用批处理大小: {batch_size}")
    
    # 初始化结果列表
    all_dssim1 = []
    all_dssim2 = []
    all_psnr = []
    all_lpips = []
    
    # 初始化LPIPS模型（如果需要）
    lpips_model = lp.LPIPS('vgg', '0.1').cuda()  # 或 'vgg'
    
    # 创建CSV文件
    csv_path = os.path.join(test_path, f'{str(render_view_id)}.csv')
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(["index", "psnr", "dssim1", "dssim2", "lpips"])
        
        # 分批处理图像
        for batch_start in tqdm(range(0, total_images, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, total_images)
            batch_files = image_files[batch_start:batch_end]
            
            # 加载当前批次的渲染图和GT图
            renders_batch = []
            gts_batch = []
            
            for fname in batch_files:
                render_path = os.path.join(renders_dir, fname)
                gt_path = os.path.join(gt_dir, fname)
                
                # 加载图像并转换为Tensor
                render = Image.open(render_path)
                gt = Image.open(gt_path)
                target_size = gt.size  # 返回一个元组 (width, height)

                # 将render图片调整为gt的尺寸
                # 使用高质量的重采样算法（如LANCZOS）
                render = render.resize(target_size, Image.Resampling.LANCZOS)
                
                render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
                gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
                
                renders_batch.append(render_tensor)
                gts_batch.append(gt_tensor)
            
            # 计算当前批次的指标
            for idx, (render, gt) in enumerate(zip(renders_batch, gts_batch)):
                global_idx = batch_start + idx
                
                try:
                    # 计算各项指标
                    dssim1 = (1 - ssim(render, gt)[0]) / 2
                    dssim2 = (1 - msssim(render, gt)) / 2
                    psnr_val = psnr(render, gt)
                    
                    # LPIPS计算较慢且耗内存，可根据需要启用
                    # lpips_val = 0  # 默认不计算LPIPS
                    lpips_val = lpips_model(render, gt).item()  # 如果需要计算LPIPS
                    
                    # 保存结果
                    all_dssim1.append(dssim1.item())
                    all_dssim2.append(dssim2.item())
                    all_psnr.append(psnr_val.item())
                    all_lpips.append(lpips_val)
                    
                    # 写入CSV
                    writer.writerow([global_idx, psnr_val.item(), dssim1.item(), dssim2.item(), lpips_val])
                    
                except Exception as e:
                    print(f"计算图像 {global_idx} ({batch_files[idx]}) 时出错: {e}")
                    # 写入错误标记
                    writer.writerow([global_idx, "ERROR", "ERROR", "ERROR", "ERROR"])
                
                finally:
                    # 释放当前图像占用的内存
                    del render, gt
                    torch.cuda.empty_cache()  # 清理GPU缓存[9,10](@ref)
            
            # 释放整个批次的内存
            del renders_batch, gts_batch
            torch.cuda.empty_cache()
    
    # 计算平均指标
    avg_psnr = np.mean(all_psnr) if all_psnr else 0
    avg_dssim1 = np.mean(all_dssim1) if all_dssim1 else 0
    avg_dssim2 = np.mean(all_dssim2) if all_dssim2 else 0
    avg_lpips = np.mean(all_lpips) if all_lpips else 0
    
    # 输出结果
    print("\n===== 最终结果 =====")
    print(f"评估图像数量: {len(all_psnr)}/{total_images}")
    print(" Avg PSNR : {:>12.7f}".format(avg_psnr))
    print(" Avg DSSIM1 : {:>12.7f}".format(avg_dssim1))
    print(" Avg DSSIM2 : {:>12.7f}".format(avg_dssim2))
    print(" Avg LPIPS : {:>12.7f}".format(avg_lpips))
    
    # 将平均值写入CSV
    with open(csv_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["avg", avg_psnr, avg_dssim1, avg_dssim2, avg_lpips])
    
    return {
        "psnr": avg_psnr,
        "dssim1": avg_dssim1,
        "dssim2": avg_dssim2,
        "lpips": avg_lpips
    }


def process_n3dv():
    # device = torch.device("cuda:2")
    # torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    Mlp = ModelParams(parser)
    
    args = parser.parse_args(sys.argv[1:])

    iteration = 100000
    
    render_view_ids = [0]
    gt_root_dir = '/data6/liangjie/experiment_result/localdy_n3dv/hash_1200_30/voxel_de/test/ours_32640/gt/' # 包含1400帧的gt 每次就不用重复保存gt了
    # gt_root_dir = None
    for render_view_id in render_view_ids:
        evaluate_memory_efficient(args.model_path, iteration, render_view_id, batch_size=100, gt_root_dir=gt_root_dir)

def process_n3dv_short():
    # device = torch.device("cuda:2")
    # torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    Mlp = ModelParams(parser)
    
    args = parser.parse_args(sys.argv[1:])

    iteration = 32640
    render_view_ids = [0, 1, 2, 3]
    render_view_ids = [0]
    # gt_root_dir = '/data6/liangjie/experiment_result/localdy_1400_exp/hash/main/test/ours_1523200/gt' # 包含1400帧的gt 每次就不用重复保存gt了
    gt_root_dir = None

    items = os.listdir('/data6/liangjie/experiment_result/localdy_n3dv_short/hash_1200_30/')
    items = ['cook_spinach']

    for item in items:
        args.model_path = '/data6/liangjie/experiment_result/localdy_n3dv_short/hash_1200_30/' + item + '/voxel0.16'
        for render_view_id in render_view_ids:
            evaluate_memory_efficient(args.model_path, iteration, render_view_id, batch_size=100, gt_root_dir=gt_root_dir)


if __name__ == "__main__":
    # device = torch.device("cuda:2")
    # torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--iteration", default=-1, type=int, nargs='+')
    Mlp = ModelParams(parser)
    
    args = parser.parse_args(sys.argv[1:])

    render_view_ids = [0, 1, 2, 3]
    render_view_ids = [0]
    # gt_root_dir = '/data6/liangjie/experiment_result/localdy_1400_exp/hash/main/test/ours_1523200/gt' # 包含1400帧的gt 每次就不用重复保存gt了
    gt_root_dir = None

    for i_iteration in args.iteration:
    # evaluate_memory_efficient_static(args.model_path, iteration, 0, batch_size=100, gt_root_dir=gt_root_dir)
        for render_view_id in render_view_ids:
            evaluate_memory_efficient_static(args.model_path, i_iteration, render_view_id, batch_size=100, gt_root_dir=gt_root_dir)