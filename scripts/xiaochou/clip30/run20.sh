source_path=/data2/dataset/xiaochou
model_path=./output/experiment/xiaochou/clip_size/30/
iteration=50000

clip_size=30                    # M: frame count of single clip
project_total_frames=30         # N: input video frame count

CUDA_VISIBLE_DEVICES=0 python trainReferenceClip.py --project_total_frames $project_total_frames \
                                                    --clip_size $clip_size \
                                                    --frames_start_end 0 $clip_size \
                                                    --iterations $iteration \
                                                    -s $source_path \
                                                    -m $model_path \
                                                    --configs arguments/xiaochou/basketball.py \
                                                    --downsample 2.0 \
                                                    --ply_path plys/30

# CUDA_VISIBLE_DEVICES=1 python trainSourceClip.py    --project_total_frames $project_total_frames \
#                                                     --clip_size $clip_size \
#                                                     --frames_start_end 10 20 \
#                                                     --iterations $iteration \
#                                                     -s $source_path \
#                                                     -m $model_path \
#                                                     --configs arguments/tiny/basketball.py \
#                                                     --downsample 2.0

CUDA_VISIBLE_DEVICES=4 python render.py --project_total_frames $project_total_frames \
                                        --clip_size $clip_size \
                                        --frames_start_end 0 $project_total_frames \
                                        --iteration $iteration  \
                                        -s $source_path  \
                                        -m $model_path   \
                                        --configs arguments/xiaochou/basketball.py  \
                                        --skip_video  \
                                        --skip_train  \
                                        --downsample 2.0 \
                                        --llffhold 10 \
                                        --ply_path plys/30

# echo CUDA_VISIBLE_DEVICES=4 python render.py --project_total_frames $project_total_frames \
#                                         --clip_size $clip_size \
#                                         --frames_start_end 0 $project_total_frames \
#                                         --iteration $iteration  \
#                                         -s $source_path  \
#                                         -m $model_path   \
#                                         --configs arguments/xiaochou/basketball.py  \
#                                         --skip_video  \
#                                         --skip_train  \
#                                         --downsample 2.0 \
#                                         --llffhold 30 \
#                                         --ply_path plys/30

                                        
python metrics.py -m $model_path --iteration $iteration
python images2video.py -m $model_path/ --iteration $iteration