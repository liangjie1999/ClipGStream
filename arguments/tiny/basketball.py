_base_ = './default.py'

import os
# 改进策略  
OptimizationParams = dict(
    dataloader=True ,
    zerostamp_init = True ,
    # hash = False ,
    hash_init_lr = 0.0002 ,
    hash_final_lr = 0.000002 ,
    hashmap_size = 17 ,  # 17   这个可以小一点
    activation = "ReLU" ,
    n_levels = 16 ,  # 16   这个完全可以调小一点, 改成三四个都可以， 然后 n_features_per_level 调大一点，改成 16 
    n_features_per_level = 8 ,  #  4  
    base_resolution = 16 ,  #  16   调大一点
    n_neurons = 128 ,
    opacity_factor = 2 ,
    cov_factor = 2 ,
    color_factor = 2 , 
    offset_factor = 4  ,

    start_stat = 1500000,
    update_from = 1600,
    update_interval = 100,
    update_until = 30000,
    success_threshold = 0.8,
    densify_grad_threshold = 0.0006,
    percentile = 100,



    position_lr_max_steps = 200_000,
    offset_lr_max_steps = 200_000,
    mlp_opacity_lr_max_steps = 200_000,
    mlp_cov_lr_max_steps = 200_000,
    mlp_offset_lr_max_steps = 200_000,
    mlp_color_lr_max_steps = 200_000,
    mlp_featurebank_lr_max_steps = 200_000,
    appearance_lr_max_steps = 200_000,

)

ModelParams = dict(
    llffhold = 10,
)