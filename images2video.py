def images2video(image_paths, output_path, fps):
    import imageio

    images = []
    for image_path in image_paths:
        image = imageio.imread(image_path)
        images.append(image)

    # imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'videos', str(render_view_id), f'{global_start_frames}_{global_end_frames}.mp4'), render_images,fps=25)
    imageio.mimwrite(output_path, images, fps=fps)

def imageSortedFn(item):
    return int(item.split('/')[-1].split('.png')[0])

import glob
import os
import cv2

def mergeImages(root_path):
    render_view_ids = [0, 1, 2, 3]
    input_root_path = f'{root_path}/renders'
    output_root_path = f'{root_path}/videos'
    fps = 25

    for render_view_id in render_view_ids:
        image_dir_path = f'{input_root_path}/{render_view_id}'

        image_paths = glob.glob(f'{image_dir_path}/*.png')

        image_paths = sorted(image_paths, key=imageSortedFn)
        output_dir_path = f'{output_root_path}/{render_view_id}'
        os.makedirs(output_dir_path, exist_ok=True)

        image_paths = image_paths[:]
        output_path = f'{output_dir_path}/0_{len(image_paths)}.mp4'
        print(len(image_paths))
        images2video(image_paths, output_path, fps)

import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理输入路径')
    
    # 必需参数：输入路径
    parser.add_argument('--model_path', '-m', type=str, help='输入文件或目录的路径')
    parser.add_argument("--iteration", default=-1, type=int, nargs='+')
    args = parser.parse_args()
    
    for i_iteration in args.iteration:
        input_path = os.path.join(args.model_path, "test", f"ours_{i_iteration}")
        mergeImages(input_path)
