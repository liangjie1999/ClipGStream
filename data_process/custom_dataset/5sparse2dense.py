#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil
import glob

# python convert_frames.py --no_gpu -s /data8/dataset/longvideos/jpg/360_2/frame000000 --last_frame_id 19
# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--skip_undistortion", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--last_frame_id", default=-1, type=int)
parser.add_argument("--start_frame_id", default=0, type=int)

args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

if args.last_frame_id == -1:
    args.last_frame_id = len(glob.glob(f"{args.source_path}/frame*")) - 1

for id in range(args.start_frame_id,args.last_frame_id+1):
    input_path = f"{args.source_path}/frame{id:0>6}"
    
    os.system(f'cp -r {input_path}/sparse/0/* {input_path}/sparse/')
    
    patch_match_cmd = (colmap_command + " patch_match_stereo \
        --workspace_path " + input_path + " \
        --workspace_format " + "COLMAP \
        --PatchMatchStereo.geom_consistency " + "true \
        --PatchMatchStereo.max_image_size 1600")

    exit_code = os.system(patch_match_cmd)
    if exit_code != 0:
        logging.error(f"patch match failed with code {exit_code}. Exiting.")
        exit(exit_code)

    fuse_cmd = (colmap_command + " stereo_fusion \
        --workspace_path " + input_path + " \
        --workspace_format " + "COLMAP \
        --input_type " + "geometric \
        --output_path " + input_path + "/stereo/fused.ply")

    exit_code = os.system(fuse_cmd)
    if exit_code != 0:
        logging.error(f"fuse failed with code {exit_code}. Exiting.")
        exit(exit_code)