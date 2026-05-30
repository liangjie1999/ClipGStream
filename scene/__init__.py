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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model_of_reference import ReferenceClipGaussianModel
from scene.gaussian_model_of_source import SourceClipGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.spacetime_dataset import FourDGSdataset, FourDGSdatasetFromBlender
import math
import copy

class ReferenceClipScene:

    gaussians : ReferenceClipGaussianModel

    def __init__(self, args : ModelParams, gaussians : ReferenceClipGaussianModel, frames_start_end=[0,300], load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None):
        """b
        :param path: Path to colmap scene main folder.fv
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod, frames_start_end, downsample=args.downsample, llffhold=args.llffhold, ply_path=args.ply_path, clip_size=self.gaussians.opt.clip_size )
            dataset_type = "colmap"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, ply_path=ply_path)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
            dataset_type="dynerf"
        elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
            dataset_type="nerfies"
        else:
            assert False, "Could not recognize scene type!"
            

        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # scene_info.test_cameras is Colmap_Dataset（inherited from Dataset）
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)    
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)
        print("Loading blender Cameras")
        self.blender_camera = FourDGSdatasetFromBlender(scene_info.blender_cameras, args, dataset_type)

        if True:
            self.train_camera_clips = []
            
            for t_dataset in scene_info.train_camera_clips:
                self.train_camera_clips.append(FourDGSdataset(t_dataset, args, dataset_type))
            
            self.test_camera_clips = []
            
            for t_dataset in scene_info.test_camera_clips:
                self.test_camera_clips.append(FourDGSdataset(t_dataset, args, dataset_type))
            
        



        if self.loaded_iter:
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                           "history",
                                                           "point",
                                                           "0.ply"), scene_info.point_cloud)
            self.gaussians.load_model(os.path.join(self.model_path))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime, frames_start_end)




    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_camera
    
    def getTrainCamerasByCipId(self, clip):
        return self.train_camera_clips[clip]

    def getTestCameras(self, scale=1.0):
        return self.test_camera
    
    def getTestCamerasByClipId(self, clip):
        return self.test_camera_clips[clip]

    def getVideoCameras(self, scale=1.0):
        return self.video_camera
    
    def getBlenderCameras(self):
        return self.blender_camera


class SourceClipScene:
    # 这里要有一个支持load0的函数

    gaussians : SourceClipGaussianModel

    def __init__(self, args : ModelParams, gaussians : SourceClipGaussianModel, frames_start_end=[0,300], load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None, clip_id = None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        self.clip_id = clip_id      

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod, frames_start_end, downsample=args.downsample, llffhold=args.llffhold, ply_path=args.ply_path )
            dataset_type = "colmap"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, ply_path=ply_path)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
            dataset_type="dynerf"
        elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
            dataset_type="nerfies"
        else:
            assert False, "Could not recognize scene type!"
            

        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # scene_info.test_cameras 是Colmap_Dataset类型（也是继承自Dataset的）
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)    
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)

        if not os.environ.get('skip_blender', True):
            print("Loading blender Cameras")
            self.blender_camera = FourDGSdatasetFromBlender(scene_info.blender_cameras, args, dataset_type)
        else:
            self.blender_camera = None

        # if clip_id == None  => invoke by trainSourceClip.py
        # else                => invoke by render.py
        if clip_id == None:
            self.gaussians.create_source_anchors_with_reference(scene_info.point_cloud) # inherited anchors(A_0) from Reference Clip
            self.gaussians.create_reference_anchors_mask()            # Creating A_0's mask. The mask will be used to freeze the training of A_0
        else:
            model_path = gaussians.model_path
            self.gaussians.load_ply_sparse_gaussian(os.path.join(model_path, 'history', 'point', f'{self.clip_id}.ply'))
            self.gaussians.load_dynamic_module()

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"),)
        self.gaussians.save_mlp_checkpoints(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_camera
    
    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        return self.video_camera
    
    def getBlenderCameras(self):
        return self.blender_camera

