import torch
import tinycudann as tcnn
import torch.nn as nn
import os
from scene.spacetime_hexplane import HexPlaneField


# 1 + 1400 / 20 = 71
class SpaceTimePlaneField(torch.nn.Module):
    def __init__(self, args, xyz_bound_max, xyz_bound_min , feat_dim, clip_id=0):
        super(SpaceTimePlaneField, self).__init__()

        self.feat_dim = feat_dim
        self.enc_models = nn.ModuleList()
        self.levels = int(os.environ.get('levels', 12))
        levels = self.levels

        # 表明不使用双层结构
        if levels == 0:
            enc_model = HexPlaneField(args.bounds, args.kplanes_config, args.multires, 64, clip_id=clip_id ).to("cuda") 
            enc_model.set_aabb(xyz_bound_max, xyz_bound_min )

            self.enc_models.append(enc_model)
        else:
            for level in range(levels+1):
                if level == 0:
                    enc_model = HexPlaneField(args.bounds, args.kplanes_config, args.multires, int(os.environ.get('level_0_dim', 56)), clip_id=clip_id ).to("cuda") 
                    enc_model.set_aabb(xyz_bound_max, xyz_bound_min )
                else:  
                    enc_model = HexPlaneField(args.bounds, args.kplanes_config, args.multires, 64 - int(os.environ.get('level_0_dim', 56)), clip_id=clip_id ).to("cuda") 
                    enc_model.set_aabb(xyz_bound_max, xyz_bound_min )

                self.enc_models.append(enc_model)


        self.register_buffer('xyz_bound_min',xyz_bound_min)
        self.register_buffer('xyz_bound_max',xyz_bound_max)

    def dump(self, path):
        torch.save(self.state_dict(),path)
        

    def get_contracted_xyz(self, xyz):  # 远离中心的一些浮点可以不要
        with torch.no_grad():
            contracted_xyz=(xyz-self.xyz_bound_min)/(self.xyz_bound_max-self.xyz_bound_min)
            return contracted_xyz

    # point time 应该是 N * 1的tensor
    def forward(self, xyz:torch.Tensor, time):
        time_scalar = float(time[0].item())  # 假设 time 是标量 tensor
        dynamic_feature_0 = self.enc_models[0](xyz, time) 

        if self.levels != 0:
            time_i = time * self.levels - int(time_scalar * self.levels ) 
            dynamic_feature_1 = self.enc_models[ int(time_scalar * self.levels ) + 1 ](xyz,  time_i ) # 相对时间（第几个） 这里hash_inputs里面有归一化（内部时间）
            dynamic_feature_out = torch.cat([dynamic_feature_0[0], dynamic_feature_1[0]], dim=-1)
        else:
            dynamic_feature_out = dynamic_feature_0[0]
        dynmiac_mask =  torch.zeros((xyz.shape[0],1),  device="cuda")

        return  dynamic_feature_out, dynmiac_mask



    def get_params (self):
        parameter_list = []
        for name, param in self.named_parameters():
            parameter_list.append(param)
        return parameter_list
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)  
        return parameter_list