
from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
from torch.utils.data import Subset

from PIL import Image


class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()
        if isinstance(index, slice):
            # 如果是切片，返回Subset对象
            indices = list(range(len(self)))[index]

            return Subset(self, indices)
        elif isinstance(index, int):
            if self.dataset_type != "PanopticSports":
                try:
                    image, w2c, time = self.dataset[index]
                    R,T = w2c
                    mask=None
                    try:
                        # 单个焦距

                        FovX = focal2fov(self.dataset.focal[0], image.shape[2]) # 这里的FovX是通过image.shape计算得来的，
                        FovY = focal2fov(self.dataset.focal[0], image.shape[1])  
                    except:  # for colmap 走的是这里 我感觉咱们直接把focal信息也copy好吧 要不然太麻烦了 然后直接按index取
                        FovX = focal2fov(self.dataset.focal_cps[index][0], image.shape[2])
                        FovY = focal2fov(self.dataset.focal_cps[index][1], image.shape[1])  #  colmap 

                except:
                    caminfo = self.dataset[index]
                    image = caminfo.image
                    R = caminfo.R
                    T = caminfo.T
                    FovX = caminfo.FovX
                    FovY = caminfo.FovY
                    time = caminfo.time
                    mask = caminfo.mask
                return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None, 
                                image_name=f"{index}",uid=index, data_device=torch.device("cuda"),time=time,
                                mask=mask)
            else:
                return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)


class FourDGSdatasetFromBlender(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()
        if isinstance(index, slice):
            # 如果是切片，返回Subset对象
            indices = list(range(len(self)))[index]

            return Subset(self, indices)
        elif isinstance(index, int):
            if self.dataset_type != "PanopticSports":

                img, image_wh, w2c, time = self.dataset[index]
                R,T = w2c
                mask=None
                try:
                    FovX = focal2fov(self.dataset.focal[0], image_wh[0]) # 这里的FovX是通过image.shape计算得来的，
                    FovY = focal2fov(self.dataset.focal[0], image_wh[1])  
                except:  # for colmap 走的是这里 我感觉咱们直接把focal信息也copy好吧 要不然太麻烦了 然后直接按index取
                    FovX = focal2fov(self.dataset.focal_cps[index][0], image_wh[0])
                    FovY = focal2fov(self.dataset.focal_cps[index][1], image_wh[1])  #  colmap 

                # 创建一个尺寸为image_wh的纯黑图片 from PIL import Image 格式的
                
                return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=img,gt_alpha_mask=None, 
                                image_name=f"{index}",uid=index, data_device=torch.device("cuda"),time=time,
                                mask=mask)
            else:
                return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)