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

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0


import time

sequence = args.source
if not args.skip_matching:
    os.makedirs(sequence + "/distorted/sparse", exist_ok=True)

        # --image_path " + sequence + "/input \
    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + sequence + "/distorted/database.db \
        --image_path " + sequence + "/ \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    start = time.perf_counter()
    total_start = start
    exit_code = os.system(feat_extracton_cmd)
    feat_time = time.perf_counter() - start
    print(f">>> Feature extraction finished in {feat_time:.2f} seconds.")

    # assert(False)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + sequence + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    start = time.perf_counter()
    exit_code = os.system(feat_matching_cmd)
    match_time = time.perf_counter() - start

    print(f">>> Feature matching finished in {match_time:.2f} seconds.")
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + sequence + "/distorted/database.db \
        --image_path "  + sequence + "/ \
        --output_path "  + sequence + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    start = time.perf_counter()
    exit_code = os.system(mapper_cmd)
    mapper_time = time.perf_counter() - start
    print(f">>> Mapper finished in {mapper_time:.2f} seconds.")
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    total_time = time.perf_counter() - total_start
    print(f">>> Total COLMAP reconstruction time: {total_time:.2f} seconds.")




### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + sequence + "/ \
    --input_path " + sequence + "/distorted/sparse/0 \
    --output_path " + sequence + "/ \
    --output_type COLMAP ") # --min_scale 1 --max_scale 1
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(sequence + "/sparse")
os.makedirs(sequence + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(sequence, "sparse", file)
    destination_file = os.path.join(sequence, "sparse", "0", file)
    shutil.move(source_file, destination_file)


print("Done.")
