import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import os

def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0

# aabb 是 2 * 3
def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0
def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    # coords 1 N 2 (为什么第二个维度是 [1] * (grid_dim - 1))
    # 如果平面是两维的(x y) 就是 @ 1 @ @ 
    # 如果平面是三维的(x y z) @ 1 1 @ @
    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    # 这个应该是把grid看成是坐标点，然后coords是我们的采样点，每个采样点做bilinear采样
    # 这个grid_sampler接收的数据是要求什么格式的
    interp = grid_sampler(  # 输出是 1 16 1 n 这个grid要求的输入维度/输出维度是怎样的
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    # interp的输出是 B C(feature_dim=16) h_out(1) w_out(数量n) 1 16 1 n
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

# ParameterList
# 1 * 16 * 64 * 64
# 1 * 16 * 64 * 64
# 1 * 16 * 150 * 64
def init_grid_param(
        grid_nd: int, # 2
        in_dim: int, # 4 x y z t
        out_dim: int,
        reso: Sequence[int], # 360中 [64, 64, 64, 150] xyz是64 时间是150
        a: float = -0.01, # 0.1
        b: float = 0.01): # 0.5
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd)) 
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]  # 这是因为 PyTorch Tensor 的存储顺序 与我们习惯的坐标系维度顺序常常相反。
                                                                # Python list 里的维度一般 [dim_x, dim_y]
                                                                # 但在 Tensor 里，最后一个维度通常对应 “最内层/最快变化的维度”。
        ))

        # new_grid_coef [1, out, y, x]
        # coo_comb 有time的 x, time  y, time z, time的 参数都设成了0
        # if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
        #     nn.init.ones_(new_grid_coef)
        # else:
        nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs

def save_grid_param_from_gop0(grid_coefs, res):
    model_path = os.environ['model_path']

    os.makedirs(os.path.join(model_path, 'history'), exist_ok=True)
    history_path = os.path.join(model_path, 'history', f'hex_plane_init_weight_{res}.pth')

    torch.save(grid_coefs.state_dict(), history_path)

def init_grid_param_from_gop0(grid_nd: int, # 2
        in_dim: int, # 4 x y z t
        out_dim: int,
        reso: Sequence[int], # 360中 [64, 64, 64, 150] xyz是64 时间是150
        res: int,
        a: float = 0.1,
        b: float = 0.5,
        ):
    
    model_path = os.environ['model_path']

    hisitory_path = os.path.join(model_path, 'history', f'hex_plane_init_weight_{res}.pth')
    
    gop_0_weight = torch.load(hisitory_path)

    grid_coefs_reloaded = init_grid_param(grid_nd=grid_nd, in_dim=in_dim, out_dim=out_dim, reso=reso, a=a, b=b)  
    grid_coefs_reloaded.load_state_dict(gop_0_weight)   # 一个nn.Paramer是有load_state_dict的能力的
    
    return grid_coefs_reloaded

# 4 * 16 * 64 * 150 * 6 一个平面（一个分辨率下）3M多
# 4 * 16 * 128 * 300 * 6 14M
# yige
def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],  # 
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id,  grid in enumerate(ms_grids[:num_levels]): # 这里的num_level指的是多平面的分辨率有几个
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane           # 注释里也可以写语法 *reso 解构 shape[1]是output dim
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso 
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])   # pts本身是 N*4的，对后一个取了C 所以是N * 2
                .view(-1, feature_dim) # 这个好像没必要 这个是转换成 N 16了？(输出就已经是这样了)
            )
            # compute product over planes
            # 采样结果乘起来（就是yx zx ... )所有平面采样的结果
            # 这个采样还能训练吗（双线性应该是可微的）
            # 多分辨率平面 是平面的分辨率不同，但其实输出的特征形状是相同的 都是 N * 16
            # interp_space = interp_space * interp_out_plane

            # combine over scales
            if concat_features:
                multi_scale_interp.append(interp_out_plane)
            else:
                multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp  # 分辨率之间特征concat，分辨率内特征乘

# QUESTION: 为什么hash网格要做16个不同分辨率的网格，但是hexPlane就不需要呢？
class HexPlaneField(nn.Module):
    def __init__(
        self,
        bounds,
        planeconfig, # args.kplanes_config
        multires,   # args.multires
        feature_dim,
        clip_id = 0
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        # self.feature_out = nn.Sequential(nn.Linear(32 ,feature_dim))
        # self.mlp_mask = nn.Sequential( nn.ReLU(), nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, 1),nn.Sigmoid())
        
        self.mlp_mask = tcnn.Network(
            n_input_dims= 4,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 128,
                "n_hidden_layers": 1,
            },
        )


        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        # 平面也是多分辨率的 args.multires控制 360默认的
        # 一个plane的参数量
        # 2（两个分辨率) * 6(6个平面) * （64 * 150 * 5)[5是因为精细分辨率是粗的4倍] * 16(每个特征的参数量)
        # 这个分辨率是有多个的
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            # 为什么这里只在空域上做多分辨率?时域没有这个说法吗？
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            # 一个gp包含 xyzt（C42 共6个）参数 每个参数的形状是 1 outputdim=16 y x
            if clip_id == 0:
                gp = init_grid_param(
                    grid_nd=config["grid_dimensions"], # 2
                    in_dim=config["input_coordinate_dim"], # 4
                    out_dim=config["output_coordinate_dim"],
                    reso=config["resolution"],
                )

                save_grid_param_from_gop0(gp, res)
            else:
                gp = init_grid_param_from_gop0(
                    grid_nd=config["grid_dimensions"], # 2
                    in_dim=config["input_coordinate_dim"], # 4
                    out_dim=config["output_coordinate_dim"],
                    reso=config["resolution"],
                    res=res
                )

            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]    # 这个是固定的config["output_coordinate_dim"] 16
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp) # self.grids里保存的是多个分辨率的网格
        # print(f"Initialized model grids: {self.grids}")
        print("feature_dim:",self.feat_dim)
        # self.feat_dim中存的是 Σ多分辨率 output_dim 
        # 比如两层多分辨率 就是 16 + 16 = 32
        
        self.feature_out = nn.Sequential(nn.Linear( 288,feature_dim)) # self.feat_dim
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.cat([
            xyz_max,
            xyz_min
        ]).view(2,3)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        # pts: N * 3
        pts = normalize_aabb(pts, self.aabb)    # 变换到[-1, 1]
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)


        return features  # [N,32]

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        hidden = self.get_density(pts, timestamps)
        dynamic_feature = self.feature_out(hidden)

        # dynmiac_mask = self.mlp_mask (torch.cat([pts,timestamps], dim = -1))
        dynmiac_mask = 0.5
        return dynamic_feature, dynmiac_mask  # [N,32]

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