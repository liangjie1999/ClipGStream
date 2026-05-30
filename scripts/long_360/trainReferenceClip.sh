
### You need set ###
export CUDA_VISIBLE_DEVICES=3 
export source_path="/data8/dataset/longvideos/full_court/" # the path of dataset

export bash_path="$(realpath "${BASH_SOURCE[0]}")"

python trainReferenceClip.py --iterations 10880 --clip_size 10 --project_total_frames 700 -s $source_path -m ./output/long_360 --frames_start_end 0 10 --configs arguments/long360/basketball.py
