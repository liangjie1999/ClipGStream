import json
import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import viser.transforms as tf

input_root_path = '/data5/liangjie/dynamic/ClipGStream/data_process/double_round_70_s'


with open(f'{input_root_path}/camera_path_blender.json', 'r') as f:
    data = json.load(f)
# # --- 输入你的 JSON 数据 ---
# data = {
#     "camera_type": "perspective",
#     "render_height": 1080,
#     "render_width": 1920,
#     "camera_path": [
#         {
#             "camera_to_world": [
#         0.1021125391125679,
#         -0.8083203434944153,
#         0.5798200368881226,
#         0.44067883491516113,
#         -0.21579712629318237,
#         -0.5869864225387573,
#         -0.7803066968917847,
#         -0.08004200458526611,
#         0.9710842370986938,
#         -0.04544439911842346,
#         -0.23437188565731049,
#         -0.12031060457229614,
#         0.0,
#         0.0,
#         0.0,
#         1.0
#             ],
#             "fov": 22.895194130645738,
#             "aspect": 1
#         },
#         # 重复帧...
#     ]
# }

# --- 计算内参 ---
w = data["render_width"]
h = data["render_height"]
fov_y_deg = data["camera_path"][0]["fov"]
fov_y_rad = math.radians(fov_y_deg)

fy = h / (2 * math.tan(fov_y_rad / 2))
fx = fy * (w / h)
cx = w / 2
cy = h / 2

import ast

with open(f"{input_root_path}/cameras.txt", "w+") as f_cam: 
    f_cam.write("# Camera list with one line of data per camera: \n")
    f_cam.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[] \n")
    f_cam.write("# Number of cameras: 1 \n")
    f_cam.write(f"1 PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}")

with open(f'{input_root_path}/images.txt', "w+") as f_cam:
    f_cam.write("# Image list with two lines of data per image: \n")
    f_cam.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME \n")
    f_cam.write("#   POINTS2D[] as (X, Y, POINT3D_ID) \n")

    keyframes = []
    # print(data['camera_path'][0]['camera_to_world'])

    # print(tf.SE3.from_matrix(np.array(data['camera_path'][0]['camera_to_world']).reshape(4, 4)))
    
    # print(tf.SE3.from_matrix(np.array(data['camera_path'][0]['camera_to_world'])))
    for i, frame in enumerate(data["camera_path"]):
        pose = tf.SE3.from_matrix(np.array(data['camera_path'][i]['camera_to_world']).reshape(4, 4))

        # apply the x rotation by 180 deg
        pose = tf.SE3.from_rotation_and_translation(
            pose.rotation() @ tf.SO3.from_x_radians(np.pi), # 右乘表示在原有变换基础上 叠加 （想想欧拉角那个公式 就是拆分完以后 你为了在当前相机状态下变换 就得往后乘变换）
            pose.translation(),                             # 存储得是P_w坐标，不用关心旧点的相机坐标（这是计算出来的）
        )

        keyframe = {
            "position": pose.translation(),
            "wxyz": pose.rotation().wxyz
        }
        
        keyframes.append(keyframe)

    for i, keyframe in enumerate(keyframes):
        pose = tf.SE3.from_rotation_and_translation(
            tf.SO3(keyframe['wxyz']),
            keyframe['position'],
        )        

        # TODO: 将pose转换为 w2c 并提取qw qx qy qz，t_w2c[0], t_w2c[1], t_w2c[2] 保存到文件中
        q_w2c  = pose.inverse().rotation().wxyz
        qw, qx, qy, qz = q_w2c
        t_w2c = pose.inverse().translation()

        # print(t_w2c)

        image_id = i + 1
        camera_id = 1
        name = f"frame_{i+1:05d}.png"

        # f_cam.write(f"{image_id} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
        #             f"{t_w2c[0] + 1:.12f} {t_w2c[1] - 2:.12f} {t_w2c[2] + 1:.12f} "
        #             f"{camera_id} {name}\n")
        f_cam.write(f"{image_id} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
                    f"{t_w2c[0]:.12f} {t_w2c[1]:.12f} {t_w2c[2]:.12f} "
                    f"{camera_id} {name}\n")
        f_cam.write("\n")

