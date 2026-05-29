_base_="default.py"
ModelParams=dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 100]
    },
)

OptimizationParams = dict(

    hash = False ,
    dataloader = False ,
    iterations = 30000 ,
    hash_init_lr = 0.0002 ,
    hash_final_lr = 0.000002 ,
    hashmap_size = 17 ,  # 17
    activation = "ReLU",
    n_levels = 16,  # 16 
    n_features_per_level = 8,  #  4
    base_resolution = 16,  #  16 
    n_neurons = 128,
    opacity_factor = 2 ,
    cov_factor = 2 ,
    color_factor = 2 , 
    offset_factor = 4  ,

    start_stat = 1500,
    update_from = 1600,
    update_interval = 100,
    update_until = 15000,
    success_threshold = 1.0,
    densify_grad_threshold = 0.001,
    percentile = 100
)


