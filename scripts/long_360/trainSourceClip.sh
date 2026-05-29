### If you have multiple GPU cards, you can run ./parralel_source_clip/generateTrainingCmd.py.
###                                 Then you will get ./training_command/run.sh


### You need set ###
export CUDA_VISIBLE_DEVICES=0 
export source_path="/data8/dataset/longvideos/full_court/" # the path of dataset

### Fixed ###
export bash_path="$(realpath "${BASH_SOURCE[0]}")"

### Single GPU card Running
for i in {1..69}; do
    start=$((10*i))
    end=$((10 + 10*i))

    python trainSourceClip.py --iterations 10880 --clip_size 10 --project_total_frames 700 -s $source_path -m ./output/long_360 --frames_start_end $start $end --configs arguments/long360/basketball.py
done
