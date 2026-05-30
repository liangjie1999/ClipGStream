ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 80]
    },
    multires = [1,2]
)

OptimizationParams = dict(
    # position_lr_init = 0.00016,   # 试试移动 seed point
    # position_lr_final = 0.0000016,
    # deformation_lr_init = 0.00016,
    # deformation_lr_final = 0.000016,
    # deformation_lr_delay_mult = 0.01,
    # grid_lr_init = 0.004,
    # grid_lr_final = 0.00016,

    # self.deformation_lr_init = 0.00016
    # self.deformation_lr_final = 0.000016
    # self.deformation_lr_delay_mult = 0.01
    # self.grid_lr_init = 0.0016
    # self.grid_lr_final = 0.00016
    hash = False ,
    dataloader = False ,
    iterations = 40000 ,
    hash_init_lr = 0.0002 ,
    hash_final_lr = 0.000002 ,
    hashmap_size = 15 ,  # 17
    activation = "ReLU" ,
    n_levels = 16 ,  # 16 
    n_features_per_level = 4 ,  #  4 
    base_resolution = 16 ,  #  16 
    n_neurons = 64 ,
    opacity_factor = 1 ,
    cov_factor = 1 ,
    color_factor = 1 , 
    offset_factor = 2  ,

    start_stat = 1500,
    update_from = 1600,
    update_interval = 100,
    update_until = 15000,
    success_threshold = 1.0,
    densify_grad_threshold = 0.001,
    percentile = 100
)



ModelParams = dict(
    sh_degree = 3,
    feat_dim = 64,
    n_offsets = 10,
    voxel_size =  0.001 ,# if voxel_size<=0, using 1nn dist
    update_depth = 3,
    update_init_factor = 16,
    update_hierachy_factor = 4,

)
