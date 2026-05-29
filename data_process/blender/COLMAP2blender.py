import numpy as np
import json
from scipy.spatial.transform import Rotation

# === 1. 读取 cameras.txt ===
def read_cameras_txt(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line and not line.startswith('#')]
    # 假设只有一个相机
    parts = lines[0].split()
    camera_id = int(parts[0])
    model = parts[1]
    width = int(parts[2])
    height = int(parts[3])
    if model == "PINHOLE":
        fx, fy, cx, cy = map(float, parts[4:8])
    elif model == "SIMPLE_PINHOLE":
        f, cx, cy = map(float, parts[4:7])
        fx = fy = f
    else:
        raise ValueError(f"Unsupported camera model: {model}")
    
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    return {
        "width": width,
        "height": height,
        "K": K,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy
    }

# === 2. 读取 images.txt (只取第一张图) ===
def read_images_txt(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line and not line.startswith('#')]
    # 每张图占两行，取第一张
    image_line = lines[0]
    parts = image_line.split()
    qvec = np.array([float(x) for x in parts[1:5]])  # qw, qx, qy, qz
    tvec = np.array([float(x) for x in parts[5:8]])  # tx, ty, tz
    camera_id = int(parts[8])
    image_name = parts[9]
    return qvec, tvec, image_name

# === 3. 主函数 ===
def colmap_to_blender_json(cameras_txt, images_txt, output_json):
    # 读取内参
    cam_info = read_cameras_txt(cameras_txt)
    qvec, tvec, image_name = read_images_txt(images_txt)

    # 四元数转旋转矩阵 (注意：scipy 默认是 [x,y,z,w]，但 COLMAP 是 [w,x,y,z])
    # 所以我们传入 [qx, qy, qz, qw]
    rot = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])
    R = rot.as_matrix()  # 3x3

    # 构建 COLMAP 的 W2C 矩阵
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R
    T_w2c[:3, 3] = tvec

    # 取逆得到 COLMAP 坐标系下的 C2W
    T_c2w_colmap = np.linalg.inv(T_w2c)

    # 坐标系变换矩阵 T: Blender <-> COLMAP
    T = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])

    # 将 COLMAP C2W 转为 Blender C2W
    # 因为 T 是对称正交矩阵，T^{-1} = T
    T_c2w_blender = T @ T_c2w_colmap @ T

    # 提取位置和旋转（欧拉角）
    location = T_c2w_blender[:3, 3]
    rotation_matrix = T_c2w_blender[:3, :3]
    euler = Rotation.from_matrix(rotation_matrix).as_euler('xyz')

    # 构建 JSON
    data = {
        "camera_name": "Camera",
        "resolution": {
            "width": cam_info["width"],
            "height": cam_info["height"]
        },
        "intrinsics": {
            "focal_length_mm": None,  # COLMAP 不提供 mm，可留空或估算
            "sensor_width_mm": None,
            "sensor_height_mm": None,
            "focal_length_px": float(cam_info["fx"]),  # 假设 fx ≈ fy
            "cx": float(cam_info["cx"]),
            "cy": float(cam_info["cy"]),
            "K": cam_info["K"].tolist()
        },
        "extrinsics": {
            "location": location.tolist(),
            "rotation_euler": euler.tolist(),
            "rotation_matrix": rotation_matrix.tolist(),
            "world_matrix": T_c2w_blender.tolist()
        }
    }

    # 保存
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✅ 已生成 {output_json}")

# === 运行 ===
if __name__ == "__main__":
    colmap_to_blender_json("cameras.txt", "images.txt", "camera_params_from_colmap.json")