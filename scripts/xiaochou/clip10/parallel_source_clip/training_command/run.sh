#!/bin/bash

# 自动生成的并行启动脚本
# 用于并行执行 8 个 GPU 任务

bash "/data2/liangjie/source_codes/ClipGStream_neuralBG/scripts/xiaochou/clip10/parallel_source_clip/training_command/script_0.sh" &
bash "/data2/liangjie/source_codes/ClipGStream_neuralBG/scripts/xiaochou/clip10/parallel_source_clip/training_command/script_1.sh" &
bash "/data2/liangjie/source_codes/ClipGStream_neuralBG/scripts/xiaochou/clip10/parallel_source_clip/training_command/script_2.sh" &
bash "/data2/liangjie/source_codes/ClipGStream_neuralBG/scripts/xiaochou/clip10/parallel_source_clip/training_command/script_3.sh" &
bash "/data2/liangjie/source_codes/ClipGStream_neuralBG/scripts/xiaochou/clip10/parallel_source_clip/training_command/script_4.sh" &
bash "/data2/liangjie/source_codes/ClipGStream_neuralBG/scripts/xiaochou/clip10/parallel_source_clip/training_command/script_5.sh" &
bash "/data2/liangjie/source_codes/ClipGStream_neuralBG/scripts/xiaochou/clip10/parallel_source_clip/training_command/script_6.sh" &
bash "/data2/liangjie/source_codes/ClipGStream_neuralBG/scripts/xiaochou/clip10/parallel_source_clip/training_command/script_7.sh" &
# 可选：等待所有后台任务完成
# wait
echo "All scripts launched."
