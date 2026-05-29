import numpy as np
import glob

import open3d as o3d

import os

from plyfile import PlyData, PlyElement

import open3d as o3d
from plyfile import PlyData
import numpy as np
from pathlib import Path

import torch

def voxel_down(input_path, voxel_size):
    pcd = o3d.io.read_point_cloud(input_path)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    input_path = Path(input_path)

    print(f"{input_path} {voxel_size} {len(pcd.points)}")
    
    return pcd, len(pcd.points)   

def inc_merge_plydata(
    base_ply: PlyData, 
    inc_ply: PlyData, 
    merge_voxel_size: float,
    get_res: bool = False,
) -> PlyData:
    base_data = base_ply.elements[0].data
    inc_data = inc_ply.elements[0].data

    base_points = np.stack((base_data['x'], base_data['y'], base_data['z']), axis=1)
    inc_points = np.stack((inc_data['x'], inc_data['y'], inc_data['z']), axis=1)

    # 创建基础点云的体素索引
    base_voxel_indices = set()
    for point in base_points:
        # 计算点所在的体素中心坐标(作为唯一索引)[9](@ref)
        voxel_coord = tuple(np.floor(point / merge_voxel_size).astype(int))
        base_voxel_indices.add(voxel_coord)

    new_points = []
    for i, point in enumerate(inc_points):
        voxel_coord = tuple(np.floor(point / merge_voxel_size).astype(int))
        
        if voxel_coord not in base_voxel_indices:
            new_points.append(inc_data[i])
                        
            # 更新索引(避免增量点云内部重复)
            base_voxel_indices.add(voxel_coord)

    if len(new_points) == 0:
        merged_points = base_data
    else:
        merged_points = np.concatenate([base_data, np.array(new_points)], dtype=base_data.dtype)

    if get_res:
        return PlyData([PlyElement.describe(merged_points, 'vertex')]), \
               PlyData([PlyElement.describe(np.array(new_points, dtype=base_data.dtype), 'vertex')])
    else:
        return PlyData([PlyElement.describe(merged_points, 'vertex')])
    


def getNumpyPoints(path):
    points = PlyData.read(path)

    x = points.elements[0].data['x']
    y = points.elements[0].data['y']
    z = points.elements[0].data['z']

    return np.stack([x, y, z], axis=1)

def InnerClipDeduplication(ply_paths, voxel_size=0.008):
    point = PlyData.read(ply_paths[0])

    for ply_path in ply_paths[1:]:
        point_residual = PlyData.read(ply_path)
        
        point = inc_merge_plydata(point, point_residual, merge_voxel_size=voxel_size)
    
    return point

def createSphericalCoverageField(reference_clip_ply_path):
    def o3d_knn(pcd, num_knn):
        indices = []
        sq_dists = []

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        for p in pcd.points:
            [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
            indices.append(i[1:])
            sq_dists.append(d[1:])
        return np.array(sq_dists), pcd_tree

    # 使用 open3d 读取 PLY 文件
    pcd = o3d.io.read_point_cloud(reference_clip_ply_path)
    
    sq_dist, field = o3d_knn(pcd, 3)
    dist2 = sq_dist.mean(-1).clip(min=0.0000001)
    # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(positions).float().cuda()), 0.0000001)
    field_radius = np.sqrt(dist2)  # 形状: (N, 1)
    
    sphericalCoverageField = (field, field_radius)
    return sphericalCoverageField

def batch_point_to_spheres_distance(kdtree, gs_radii, query_points, k=10):
    """
    输入: query_points (M,3) 
    输出: distances (M,) 每个点到最近球体表面的距离
    """
    query_points = np.asarray(query_points)
    M = query_points.shape[0]
    distances = np.empty(M, dtype=np.float32)

    for qi in range(M):
        q = query_points[qi]
        [_, idxs, dists2] = kdtree.search_knn_vector_3d(q, k)
        
        min_dist = float("inf")
        for i, d2 in zip(idxs, dists2):
            r = gs_radii[i]
            d = np.sqrt(d2)          # 点到球心距离
            dist_to_sphere = max(d - r, 0.0)
            if dist_to_sphere < min_dist:
                min_dist = dist_to_sphere
        distances[qi] = min_dist

    return distances


def GeometryAwareDeduplication(SCF, source_clip_ply_path, distance_threshold=0.005, add_index = 1):
    field = SCF[0]
    field_radius = SCF[1]

    distances = batch_point_to_spheres_distance(field, field_radius, getNumpyPoints(source_clip_ply_path), k=1)
    mask = distances >= distance_threshold

    # 过滤source clip点云
    plydata = PlyData.read(source_clip_ply_path)
    vertex_data = plydata['vertex'].data  # 原始结构化数组
    filtered_vertex = vertex_data[mask]
    filtered_el = PlyElement.describe(filtered_vertex, 'vertex')

    return PlyData([filtered_el])


from argparse import ArgumentParser
if __name__ == '__main__':
    # python GADedu.py --clip_size 10 --source_path /data2/dataset/xiaochou
    parser = ArgumentParser("gaussianDeduce")
    
    parser.add_argument("--skip_Downsample", action="store_true")
    parser.add_argument("--skip_ICDedup", action="store_true")
    parser.add_argument("--skip_GADedup", action="store_true")

    parser.add_argument("--clip_size", type=int, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--voxel_size", type=float, default=0.012)

    args = parser.parse_args()


    # TODO: 这里应该增加一个读取点云并做下采样的过程
    # 给每一帧做下采样后 把点云集中到一起再

    clip_size = args.clip_size
    ply_paths = glob.glob(f'{args.source_path}/frame*/stereo/fused.ply') # 这个点云本身应该也是有间隔的 
    ply_paths = sorted(ply_paths, key=lambda item: int(item.split('/frame')[-1].split('/')[0]))

    if not args.skip_Downsample:
        # 下采样
        for path in ply_paths:
            p = Path(path)
            folder = p.parent
            stem = p.stem
            suffix = p.suffix
            pcd, pcd_len = voxel_down(path, args.voxel_size)
            o3d.io.write_point_cloud(f'{folder}/{stem}_{args.voxel_size}{suffix}', pcd)
        
        ply_paths = glob.glob(f'{args.source_path}/frame*/stereo/fused_{args.voxel_size}.ply') # 这个点云本身应该也是有间隔的 
        ply_paths = sorted(ply_paths, key=lambda item: int(item.split('/frame')[-1].split('/')[0]))

    output_root_dir = f'{args.source_path}/plys/{clip_size}'

    innerClipPly_path = os.path.join(output_root_dir, 'innerClipPly')
    residualClipPly_path = os.path.join(output_root_dir, 'residualClipPly')
    os.makedirs(innerClipPly_path, exist_ok=True)
    os.makedirs(residualClipPly_path, exist_ok=True)
    
    if not args.skip_ICDedup:
        # InnerClipDeduplication
        voxel_size = args.voxel_size
        clip_nums = len(ply_paths) // clip_size
        for clip_id in range(0, clip_nums):
            cur_clip_paths = ply_paths[clip_id * clip_size:min(len(ply_paths), (clip_id + 1) * clip_size)]

            clip_points = InnerClipDeduplication(cur_clip_paths, voxel_size=voxel_size)

            print(f"Inner Clip Deduplicate {clip_id} finish.")
            clip_points.write(f'{innerClipPly_path}/{clip_id}.ply')

    if not args.skip_GADedup:
        # GeometryAwareDeduplication
        # 创建高斯场  spherical coverage field
        # 传入Reference Clip的点云
        SCF = createSphericalCoverageField(f'{innerClipPly_path}/0.ply')
        # 基于高斯场去重

        for clip_id in range(1, clip_nums):
            residual_ply = GeometryAwareDeduplication(SCF, f'{innerClipPly_path}/{clip_id}.ply')

            print(f"Geometry Aware Deduplicate {clip_id} finish.")
            residual_ply.write(f'{residualClipPly_path}/{clip_id}.ply')
        
        os.system(f'cp {innerClipPly_path}/0.ply {residualClipPly_path}/')

    os.system(f'cp {residualClipPly_path}/* {args.source_path}/plys/')