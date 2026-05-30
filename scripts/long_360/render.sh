### If you have multiple GPU cards, you can run ./parralel_source_clip/generateRenderCmd.py.
###                                 Then you will get ./render_command/run.sh


### You need set ###
export CUDA_VISIBLE_DEVICES=1
export source_path="/data8/dataset/longvideos/full_court/"

### Fixed ###
export bash_path="$(realpath "${BASH_SOURCE[0]}")"
### Single GPU card Running
python render.py --iterations 10880 --clip_size 10 --project_total_frames 700 -s $source_path --iteration $per_clip_iter -m ./output/long_360 --frames_start_end 0 700  --configs arguments/long360/basketball.py --skip_video --skip_train

python metrics.py -m ./output/long_360

python images2video.py -m ./output/long_360/ --iteration 10880