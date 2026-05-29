
import os
import cv2
import random
import numpy as np
from PIL import Image
 
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms, utils
import random
def get_stamp_list(dataset, timestamp):
    frame_length = int(len(dataset)/len(dataset.dataset.poses))
    # print(frame_length)
    if timestamp > frame_length:
        raise IndexError("input timestamp bigger than total timestamp.")
    print("select index:",[i*frame_length+timestamp for i in range(len(dataset.dataset.poses))])
    return [dataset[i*frame_length+timestamp] for i in range(len(dataset.dataset.poses))]
class FineSampler(Sampler):
    def __init__(self, dataset):
        self.len_dataset = len(dataset) 
        self.len_pose = len(dataset.dataset.poses)
        self.frame_length = int(self.len_dataset/ self.len_pose)

        sample_list = []
        for i in range(self.frame_length):
            for j in range(4):
                idx = torch.randperm(self.len_pose) *self.frame_length + i
                # print(idx)
                # breakpoint()
                now_list = []
                cnt = 0
                for item in idx.tolist():
                    now_list.append(item)
                    cnt+=1
                    if cnt % 2 == 0 and len(sample_list)>2:    
                        select_element = [x for x in random.sample(sample_list,2)]
                        now_list += select_element
            
            sample_list += now_list
            
        self.sample_list = sample_list
        # print(self.sample_list)
        # breakpoint()
        print("one epoch containing:",len(self.sample_list))
    def __iter__(self):

        return iter(self.sample_list)
    
    def __len__(self):
        return len(self.sample_list)



class FineSampler(Sampler):
    def __init__(self, dataset, clip_size = 20, per_clip_iter = 10000, total_frames = 250):
        self.dataset = dataset
        self.cur_clip = 0
        self.clip_size = clip_size # 20帧
        self.per_clip_iter = per_clip_iter
        self.total_frames = total_frames

    def __iter__(self):
        # 根据当前的clip_id，对time_list属性进行筛选

        # dataset下面的点云的time_list属性 筛选出idx在 [cur_clip, cur_clip * clip_size]的点


        return iter(self.sample_list)
    
    def __len__(self):
        return len(self.sample_list)

    def set_cur_clip(self, cur_clip):
        self.cur_clip = cur_clip