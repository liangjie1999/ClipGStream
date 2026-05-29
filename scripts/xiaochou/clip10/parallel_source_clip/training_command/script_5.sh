#!/bin/bash

# 自动生成的并行脚本 - GPU 5
# 包含 1 个命令

export bash_path="$(realpath "${BASH_SOURCE[0]}")" 
echo '执行命令 1/1 - GOP 6'
CUDA_VISIBLE_DEVICES=5 python trainSourceClip.py --project_total_frames 1500 --clip_size 10 --iterations 20000 -s /data2/dataset/xiaochou -m ./output/experiment/xiaochou/clip_size/10/ --frames_start_end 60 70 --configs arguments/xiaochou/basketball.py --downsample 2.0 --ply_path plys/10/residualClipPly
