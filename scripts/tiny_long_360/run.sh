source_path=/data8/dataset/longvideos/jpg/long_360_tiny_dataset
model_path=./output/tiny_long_360
iteration=5000

clip_size=10                    # M: frame count of single clip
project_total_frames=20         # N: input video frame count

CUDA_VISIBLE_DEVICES=1 python trainReferenceClip.py --project_total_frames $project_total_frames \
                                                    --clip_size $clip_size \
                                                    --frames_start_end 0 10 \
                                                    --iterations $iteration \
                                                    -s $source_path \
                                                    -m $model_path \
                                                    --configs arguments/tiny/basketball.py

CUDA_VISIBLE_DEVICES=1 python trainSourceClip.py    --project_total_frames $project_total_frames \
                                                    --clip_size $clip_size \
                                                    --frames_start_end 10 20 \
                                                    --iterations $iteration \
                                                    -s $source_path \
                                                    -m $model_path \
                                                    --configs arguments/tiny/basketball.py

CUDA_VISIBLE_DEVICES=4 python render.py --project_total_frames $project_total_frames \
                                        --clip_size $clip_size \
                                        --frames_start_end 0 20 \
                                        --iteration $iteration  \
                                        -s $source_path  \
                                        -m $model_path   \
                                        --configs arguments/tiny/basketball.py  \
                                        --skip_video  \
                                        --skip_train  \
python metrics.py -m $model_path --iteration $iteration
python images2video.py -m $model_path/ --iteration $iteration