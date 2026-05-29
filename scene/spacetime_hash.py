import torch
import tinycudann as tcnn
import torch.nn as nn
import os

# 1 + 1400 / 20 = 71
class SpaceTimeHashingField(torch.nn.Module):
    def __init__(self, xyz_bound_min, xyz_bound_max , hashmap_size=20, activation = "ReLU", n_levels = 16,
        n_features_per_level=4, base_resolution = 16 ,n_neurons = 128 , feat_dim = 64 , levels = 0, open_feat_cat=True):
        super(SpaceTimeHashingField, self).__init__()
        self.open_feat_cat = open_feat_cat
        self.feat_dim = feat_dim
        self.enc_models = nn.ModuleList()
        self.levels = levels

        if self.levels == 0:
            enc_model = tcnn.NetworkWithInputEncoding(
            n_input_dims = 4,
            n_output_dims = 64 if self.open_feat_cat else self.feat_dim,    
            encoding_config={
                "otype": "HashGrid" ,
                "n_levels": 16 , # 16   16个不同分辨率的网格
                "n_features_per_level": 4,  # 每个节点4个特征
                "log2_hashmap_size": hashmap_size, # 2的20次方 hashmap最大尺寸
                "base_resolution": 16, # 最初的网格是16
                "per_level_scale": 2.0 , # 每个网格是上一个网格的2倍 16 * 2**15 = 最大网格为2 ** 19
            },                           # 所有网格总共 2 ** 4 + 2 ** 5 + .... 2** 19 (差不多是2 ** 20个节点)
                                            # 网格共有参数 n_features_per_level (4) * 2 ** 20 = 400多万参数  4M * 4B(一个float) 16MB? 
                                            # 真实参数量：2的20次方 百万个 * 4B * 4 = 16MB 这里每个分辨率的网格都是一个单独的hash 还要再乘16 存储大小是看hashmapsize的（不是网格size）
                                            # 100M
            # 第一层 64 * 128(n_neurons)
            # 第二层 128 * 56(n_output_dims) 5万才50多k
            network_config={
                "otype": "FullyFusedMLP",
                "activation": activation,   
                "output_activation": "ReLU",  
                "n_neurons": n_neurons, # 这个神经元个数和输出的维度有关系吗：没有这是hidden layer
                "n_hidden_layers": 1 ,
            },
            )
            self.enc_models.append(enc_model)
        else:
            # no use
            for level in range(levels+1):
                if level == 0:
                    enc_model = tcnn.NetworkWithInputEncoding(
                        n_input_dims = 4,
                        n_output_dims = int(os.environ.get('level_0_dim', 56)),    # 16    64  as same as 4dgs 动态特征是由大grid和小grid拼接而来的 因此是56 + 8  =64
                        # 这个encoding的输出 是 16 * 4 = 64维吗？
                        # 一个hash的参数量计算：
                        # 16 [16个hash] * (2 ** 20) [每个Hash的特征量] * 4（每个特征的参数量）= 64M参数
                        # 一个hash的字节数：64M * 4 = 256M
                        # * 70就是13G了
                        encoding_config={
                            "otype": "HashGrid" ,
                            "n_levels": 16 , # 16   16个不同分辨率的网格
                            "n_features_per_level": 4,  # 每个节点4个特征
                            "log2_hashmap_size": hashmap_size , # 2的20次方 hashmap最大尺寸
                            "base_resolution": 16, # 最初的网格是16
                            "per_level_scale": 2.0 , # 每个网格是上一个网格的2倍 16 * 2**15 = 最大网格为2 ** 19
                        },                           # 所有网格总共 2 ** 4 + 2 ** 5 + .... 2** 19 (差不多是2 ** 20个节点)
                                                    # 网格共有参数 n_features_per_level (4) * 2 ** 20 = 400多万参数  4M * 4B(一个float) 16MB? 
                                                    # 真实参数量：2的20次方 百万个 * 4B * 4 = 16MB 这里每个分辨率的网格都是一个单独的hash 还要再乘16 存储大小是看hashmapsize的（不是网格size）
                                                    # 100M
                        # 第一层 64 * 128(n_neurons)
                        # 第二层 128 * 56(n_output_dims) 5万才50多k
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": activation,   
                            "output_activation": "ReLU",  
                            "n_neurons": n_neurons, # 这个神经元个数和输出的维度有关系吗：没有这是hidden layer
                            "n_hidden_layers": 1 ,
                        },
                    )
                else:
                    enc_model = tcnn.NetworkWithInputEncoding(
                        n_input_dims = 4,
                        n_output_dims = 64 - int(os.environ.get('level_0_dim', 56)),    # 16    64  as same as 4dgs
                        encoding_config={
                            "otype": "HashGrid" ,
                            "n_levels": 16 , # 16
                            "n_features_per_level": 8,  # 8
                            "log2_hashmap_size": 19 , # 20
                            "base_resolution": 16, # 
                            "per_level_scale": 2.0 ,
                        },

                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": activation,
                            "output_activation": "ReLU",  
                            "n_neurons": n_neurons,
                            "n_hidden_layers": 1 ,  # 这个是hidden layer 为1 真实的网络应该就是两层了128 8
                        },
                    )
                self.enc_models.append(enc_model)


        self.mlp_mask = tcnn.Network(
            n_input_dims= 4,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 1,
            },
        )

        self.register_buffer('xyz_bound_min',xyz_bound_min)
        self.register_buffer('xyz_bound_max',xyz_bound_max)

    def dump(self, path):
        torch.save(self.state_dict(),path)
        
    # TODO： 这个xyz_bound在训练的时候 是根据gop0 和 curgop的 我们这里最好也按这种pair的方式实现
    def get_contracted_xyz(self, xyz):  # 远离中心的一些浮点可以不要
        with torch.no_grad():
            contracted_xyz=(xyz-self.xyz_bound_min)/(self.xyz_bound_max-self.xyz_bound_min)
            return contracted_xyz

    # point time 应该是 N * 1的tensor
    def forward(self, xyz:torch.Tensor, time):
        contracted_xyz=self.get_contracted_xyz(xyz)                          # Shape: [N, 3]
        
        mask = ( contracted_xyz >= 0 ) & ( contracted_xyz <= 1 )
        mask = mask.all(dim=1)

        dynamic_features = []
        time_scalar = float(time[0].item())  # 假设 time 是标量 tensor

        # for i in range(self.levels):
        #     idx = (2**i - 1) + int(time_scalar * 2**i)

        #     time_i = (time[mask] * (2**i) - int(time_scalar * 2**i) )
        #     hash_inputs = torch.cat([contracted_xyz[mask], time_i], dim=-1) # time_i

        #     dynamic_feature_level = self.enc_models[idx](hash_inputs)  # [M, feat_dim]
        #     dynamic_features.append(dynamic_feature_level) # 为啥动态特征 倾向于变成 0 ？

        # # 拼接所有层的特征：[M, feat_dim * levels]
        # dynamic_feature_out = torch.cat(dynamic_features, dim=-1)
        # dynamic_feature_out = sum(dynamic_features)
        hash_inputs = torch.cat([contracted_xyz[mask], time[mask]], dim=-1) # time_i

        from time import time
        start_time = time()
        dynamic_feature_0 = self.enc_models[0](hash_inputs) 
        end_time = time()

        # print(f"STF sample time: {end_time - start_time}")

        if self.levels != 0:
            time_i = time[mask] * self.levels - int(time_scalar * self.levels ) 
            hash_inputs = torch.cat([contracted_xyz[mask], time_i], dim=-1)
            dynamic_feature_1 = self.enc_models[ int(time_scalar * self.levels ) + 1 ](hash_inputs) # 相对时间（第几个） 这里hash_inputs里面有归一化（内部时间）
            dynamic_feature_out = torch.cat([dynamic_feature_0,dynamic_feature_1],dim=-1)
        else:
            dynamic_feature_out = dynamic_feature_0

        # 输出总维度也变了
        dynamic_feature = None
        if self.open_feat_cat:
            dynamic_feature = torch.zeros((xyz.shape[0], self.feat_dim // 2), device="cuda") # cat模式需要一半
        else:
            dynamic_feature = torch.zeros((xyz.shape[0], self.feat_dim), device="cuda") # cat模式需要一半        
        
        dynamic_feature[mask] = dynamic_feature_out.float()



        temp_dynamics = self.mlp_mask( hash_inputs )  
        dynmiac_mask =  torch.zeros((xyz.shape[0],1),  device="cuda")
        dynmiac_mask[mask] = torch.sigmoid(temp_dynamics.float())
        
        return  dynamic_feature, dynmiac_mask
    


    def get_params (self):

        parameter_list = []
        for name, param in self.named_parameters():
            parameter_list.append(param)
        return parameter_list
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters(): # enc_model.para
            if  "enc_model" not in name:
                parameter_list.append(param)
        return parameter_list


    def get_hash_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "enc_model" in name:
                parameter_list.append(param)  
        return parameter_list
