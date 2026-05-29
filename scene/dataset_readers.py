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
import glob
import sys
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple, Optional
from colorama import Fore, init, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, read_extrinsics_text_list
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
import torchvision.transforms as transforms
import torch
import torch.nn as nn

from scene.get_render_poses import get_render_poses,interpolation_frames

from utils.sh_utils import SH2RGB
from scene.gaussian_model_of_reference import BasicPointCloud
import cv2

import math

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int
    train_camera_clips: Optional[list]
    test_camera_clips: Optional[list]
    blender_cameras: Optional[list]


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        # print(f'FovX: {FovX}, FovY: {FovY}')

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # print(f'image: {image.size}')

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])

    open_time_inject = bool(os.environ.get('open_time_inject', False))
    open_time_list_inject = bool(os.environ.get('open_time_list_inject', False))

    if open_time_inject:
        point_times = np.array(vertices['time'])[..., None]
    else:
        point_times = None

    if open_time_list_inject:
        point_times_list = np.array(vertices['time_list'])
    else:
        point_times_list = None


    return BasicPointCloud(points=positions, colors=colors, normals=normals, point_times=point_times, point_times_list=point_times_list)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    return ply_data

def colmap_format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[idx//dataset.N_frames][0], image.shape[1])
            FovY = focal2fov(dataset.focal[idx//dataset.N_frames][1], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))


    return cameras


def generateCamerasFromBlender(poses, width, height, FovX, FovY, cam):
    w2cs = np.linalg.inv(poses)
    cam_infos = []
    times = [i/10 for i in range(10)]
    # times = [0.1]
    for idx, w2c in enumerate(w2cs):
        R = w2c[:3, :3].transpose()
        T = w2c[:3, 3]

        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY , FovX=FovX , image=cam.image, time= times[idx%10], mask = None,
                              image_path=None, image_name=None, width=width, height=height)
        cam_infos.append(cam_info)
    return cam_infos

def readCameraFromBlender(path):
    cameras_extrinsic_file = os.path.join(path, "images.txt")
    cameras_intrinsic_file = os.path.join(path, "cameras.txt")
    cam_extrinsics = read_extrinsics_text_list(cameras_extrinsic_file)  #
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    return cam_extrinsics, cam_intrinsics


def readColmapSceneInfo(path, images, eval, lod, frames_start_end=[0,300],llffhold=8, downsample=1.0, clip_size=10, ply_path='plys'):
    
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        
    # TODO: 这里创建一个相机轨迹的读取

    from scene.colmap_dataset import Colmap_Dataset, Colmap_Dataset_from_Blender

    test_num = [i for i in range(len(cam_extrinsics)) if i % llffhold == 0]
    train_num = [i for i in range(len(cam_extrinsics)) if i % llffhold != 0]

    train_exr = [cam_extrinsics[i] for i in range(len(cam_extrinsics)) if i not in test_num] 
    test_exr = [cam_extrinsics[i] for i in range(len(cam_extrinsics)) if i in test_num] 


    train_dataset = Colmap_Dataset(  path, train_exr, cam_intrinsics, downsample = downsample, split = train_num ,frames_start_end = frames_start_end)
    test_dataset =  Colmap_Dataset(  path, test_exr,  cam_intrinsics,  downsample =downsample, split = test_num , frames_start_end = frames_start_end )

    if not os.environ.get('skip_blender', True):
        # using for render blender cams. (you can ignore this code)
        blender_path = os.environ.get('blender_path', '')
        cam_extrinsics_blender, cam_intrinsics_blender = readCameraFromBlender(blender_path)

        if bool(os.environ.get('open_single_view', False)):
            total = (min(frames_start_end[1], len(cam_extrinsics_blender)) - frames_start_end[0])

            single_view_index = int(os.environ.get('single_view_idx', 730))
            cam_extrinsics_blender = [cam_extrinsics_blender[single_view_index]] * total
        else:
            blender_offset = int(os.environ.get('blender_offset', 0))
            cam_extrinsics_blender = cam_extrinsics_blender[frames_start_end[0] + blender_offset:min(frames_start_end[1] + blender_offset, len(cam_extrinsics_blender))]
         
        blender_cam_dataset = Colmap_Dataset_from_Blender(path, cam_extrinsics_blender, cam_intrinsics_blender, downsample=downsample,
                                            split=list(range(0, len(cam_extrinsics_blender))),
                                            frames_start_end=frames_start_end)
    else:
        blender_cam_dataset = None

    if True:
        train_camera_clips = []
        test_camera_clips = []
        
        total_frames = frames_start_end[1] - frames_start_end[0]    
        clip_nums = math.ceil(total_frames / clip_size)

        
        for cur_clip in range(clip_nums):
            gop_dataset = Colmap_Dataset(path, train_exr, cam_intrinsics, downsample = downsample, split = train_num ,frames_start_end = frames_start_end, clip_id=cur_clip, clip_size=clip_size)   

            train_camera_clips.append(gop_dataset)
            ### test_camera ### 
            gop_dataset = Colmap_Dataset(path, train_exr, cam_intrinsics, downsample = downsample, split = test_num ,frames_start_end = frames_start_end, clip_id=cur_clip, clip_size=clip_size)   

            test_camera_clips.append(gop_dataset)


    train_cam_infos = colmap_format_infos(train_dataset, "train")

    val_cam_infos = interpolation_frames(train_cam_infos, frames_start_end, train_dataset)
    # val_cam_infos = get_render_poses(train_cam_infos)


    nerf_normalization = getNerfppNorm(train_cam_infos)


    # base_path = os.path.join(path, "plys")
    base_path = os.path.join(path, ply_path)
    cur_clip_id = frames_start_end[0] // clip_size

    ply_path = os.path.join(base_path, f'{cur_clip_id}.ply')
        

    pcd = fetchPly(ply_path)
    
    print("origin points,",pcd.points.shape[0])



    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos, 
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=300,  
                           train_camera_clips=train_camera_clips,
                           test_camera_clips=test_camera_clips,
                           blender_cameras=blender_cam_dataset
                           )
    
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", is_debug=False, undistorted=False):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]
        # check if filename already contain postfix
        if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
            extension = ""

        c2ws = np.array([frame["transform_matrix"] for frame in frames])
        
        Ts = c2ws[:,:3,3]

        ct = 0

        progress_bar = tqdm(frames, desc="Loading dataset")

        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            if not os.path.exists(cam_name):
                continue
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            
            if idx % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(frames)}"+Style.RESET_ALL})
                progress_bar.update(10)
            if idx == len(frames) - 1:
                progress_bar.close()
            
            ct += 1
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # 这里只改变旋转轴吗？坐标原点的坐标不需要改变吗
            # c2w == pose 是 相机在世界坐标系下的朝向 t是相机在世界坐标系下的绝对位置 我们只是要改相机的朝向为COLMAP格式 因此只动左上角
            c2w[:3, 1:3] *= -1
            if "small_city_img" in path:
                c2w[-1,-1] = 1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if undistorted:
                mtx = np.array(
                    [
                        [frame["fl_x"], 0, frame["cx"]],
                        [0, frame["fl_y"], frame["cy"]],
                        [0, 0, 1.0],
                    ],
                    dtype=np.float32,
                )
                # 这几个是畸变参数
                dist = np.array([frame["k1"], frame["k2"], frame["p1"], frame["p2"], frame["k3"]], dtype=np.float32)
                im_data = np.array(image.convert("RGB"))
                arr = cv2.undistort(im_data / 255.0, mtx, dist, None, mtx)
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            else:
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
            if is_debug and idx > 50:
                break
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", ply_path=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if ply_path is None:
        ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))

    return cameras

def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        # R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            time = time, mask=None))
    return cameras

def load_ply( path) : 
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)


    colors = np.random.rand(xyz.shape[0], xyz.shape[1])
    normals = np.random.rand(xyz.shape[0], xyz.shape[1])
    return BasicPointCloud(points=xyz, colors=colors, normals=normals)

def readdynerfInfo(datadir,use_bg_points,eval):
    # loading all the data follow hexplane format
    # ply_path = os.path.join(datadir, "points3D_dense.ply")
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
    datadir,
    "train",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,  
    )
        
    test_dataset = Neural3D_NDC_Dataset(
    datadir,
    "test",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
    )
    
    train_cam_infos = format_infos(train_dataset,"train")
    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # xyz = np.load
    # ply_path = os.path.join(datadir, "points3D_downsample2.ply") # 第0帧colmap点云  
    ply_path = os.path.join(datadir, "downsampled_points_50.ply")

    
    pcd = load_ply(ply_path)




    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=300
                           )
    return scene_info


def readHyperDataInfos(datadir,use_bg_points,eval):
    from scene.hyper_loader import Load_hyper_data, format_hyper_data
    import copy
    train_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split ="train")
    test_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split="test")
    print("load finished")
    train_cam = format_hyper_data(train_cam_infos,"train")
    print("format finished")
    max_time = train_cam_infos.max_time
    video_cam_infos = copy.deepcopy(test_cam_infos)
    video_cam_infos.split="video"


    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    pcd = fetchPly(ply_path)
    xyz = np.array(pcd.points)

    pcd = pcd._replace(points=xyz)
    nerf_normalization = getNerfppNorm(train_cam)
    # plot_camera_orientations(train_cam_infos, pcd.points)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )

    return scene_info
def plot_camera_orientations(cam_list, xyz):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    # xyz = xyz[xyz[:,0]<1]
    threshold=2
    xyz = xyz[(xyz[:, 0] >= -threshold) & (xyz[:, 0] <= threshold) &
                         (xyz[:, 1] >= -threshold) & (xyz[:, 1] <= threshold) &
                         (xyz[:, 2] >= -threshold) & (xyz[:, 2] <= threshold)]

    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='r',s=0.1)
    for cam in tqdm(cam_list):
        # 提取 R 和 T
        R = cam.R
        T = cam.T

        direction = R @ np.array([0, 0, 1])

        ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig("output.png")

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "dynerf" : readdynerfInfo,
    "nerfies": readHyperDataInfos,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
}