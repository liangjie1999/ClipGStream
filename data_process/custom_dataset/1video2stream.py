import cv2
import os
from argparse import ArgumentParser
from PIL import Image

# parser = ArgumentParser("Video converter")
# parser.add_argument("--source_path", "-s", required=True, type=str)
# args = parser.parse_args()
def extract_frames_to_folder_jpg(video_path, output_folder, video_label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 构建输出路径
        frame_folder = f"{output_folder}{frame_idx:06d}"
        os.makedirs(frame_folder, exist_ok=True)

        output_filename = f"{video_label}.jpg"
        output_path = os.path.join(frame_folder, output_filename)

        # 保存帧为 JPEG，设置压缩质量
        cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        print(f"Saved {output_path}")
        frame_idx += 1

    cap.release()

def extract_frames_to_folders(video_path, base_folder, video_label):
    """从视频中提取所有帧并根据帧号将它们保存到相应的文件夹中"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 为当前帧创建目录
        frame_folder = f"{base_folder}{frame_idx:06d}"
        os.makedirs(frame_folder, exist_ok=True)

        # 格式化文件名并保存图片
        output_filename = f"{video_label}.png"

        os.makedirs(os.path.join(frame_folder, 'images'), exist_ok=True)
        output_path = os.path.join(frame_folder, 'images', output_filename) # 这里就是去畸变完的（gt是相机）
        
        
        img_wh = (1352, 1014)
        frame = cv2.resize(frame, img_wh, cv2.INTER_LANCZOS4)
        
        cv2.imwrite(output_path, frame)

        print(f"Saved {output_path}")
        frame_idx += 1

    # 关闭视频文件
    cap.release()


import argparse

if __name__ == "__main__":
    # python copy_cams.py --source /data8/dataset/longvideos/jpg/360/frame000000 --scene /data8/dataset/longvideos/jpg/360/

    parser = argparse.ArgumentParser(description='Copy directories to specified locations.')
    parser.add_argument('--source', type=str, help='The source directory containing sparse and distorted folders.')

    args = parser.parse_args()
    source_path = args.source

    import glob

    items = glob.glob(f"{source_path}/*.mp4")

    for video_filename in items:
        i = int(video_filename.split('/view_')[-1].split('.mp4')[0])
        video_filename = os.path.join(source_path, video_filename)
        base_folder = os.path.join(source_path) + "/frame"
        video_label = f"cam{i:02d}"
        extract_frames_to_folder_jpg(video_filename, base_folder, video_label)

# import glob

# items = glob.glob('/data6/wujiahao/dataset/n3dv/flame_salmon_1_colmap/frame*')

# for item in items:
#     os.system(f'rm -rf {item}')
# print(items)