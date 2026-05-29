import math

### You need set ###
cuda_idxs = [0, 1, 3, 4, 5, 6]
source_path = "/data8/dataset/longvideos/full_court/"

model_path = "./output/long_360"       
project_total_frames = 700
clip_size = 10
clip_nums = math.ceil(project_total_frames / clip_size)
iterations = 10880


parallell_interval = len(cuda_idxs)
cmds = [[] for _ in range(parallell_interval)]


start_gop = 1
for i in range(start_gop, clip_nums):
    start_idx = i * clip_size
    end_idx = start_idx + clip_size

    cuda_idx = cuda_idxs[(i - start_gop) % parallell_interval]
    cmd = f"CUDA_VISIBLE_DEVICES={cuda_idx} python render.py --project_total_frames {project_total_frames} --clip_size {clip_size} --iterations {iterations} -s {source_path} -m {model_path} --frames_start_end {start_idx} {end_idx} --configs arguments/tiny/basketball.py --skip_train --skip_video"

    cmd_idx = (i - start_gop) % parallell_interval
    cmds[cmd_idx].append(cmd)


output_root = './render_command'
import os
os.makedirs(output_root, exist_ok=True)
for i, cmd_list in enumerate(cmds):
    filename = f'{output_root}/script_{i}.sh'
    
    with open(filename, 'w') as f:
        # 写入shebang和文件头[6](@ref)
        f.write("#!/bin/bash\n\n")
        f.write(f"# 自动生成的并行脚本 - GPU {i}\n")
        f.write(f"# 包含 {len(cmd_list)} 个命令\n\n")

        f.write('export bash_path="$(realpath "${BASH_SOURCE[0]}")" \n')
        
        # 逐行写入每个命令[3,4](@ref)
        for j, cmd in enumerate(cmd_list, 1):
            f.write(f"echo '执行命令 {j}/{len(cmd_list)} - GOP {start_gop + (i + (j-1)*parallell_interval)}'\n")
            f.write(f"{cmd}\n")
            
            # # 添加执行状态检查（可选）
            # f.write("if [ $? -eq 0 ]; then\n")
            # f.write(f"    echo '命令 {j} 执行成功'\n")
            # f.write("else\n")
            # f.write(f"    echo '命令 {j} 执行失败'\n")
            # f.write("    exit 1\n")
            # f.write("fi\n\n")
    
    print(f"已生成文件: {filename} 包含 {len(cmd_list)} 个命令")    


import glob


# 获取当前工作目录的绝对路径，用于构建脚本的绝对路径
abs_output_root = os.path.abspath(output_root)
run_script_path = f"{abs_output_root}/run.sh"

# 查找所有 script_*.sh 文件
script_files = sorted(glob.glob(os.path.join(abs_output_root, "script_*.sh")))

if not script_files:
    print(f"警告: 在 {abs_output_root} 中未找到任何 script_*.sh 文件")
else:
    with open(run_script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# 自动生成的并行启动脚本\n")
        f.write(f"# 用于并行执行 {len(script_files)} 个 GPU 任务\n\n")

        for script in script_files:
            # 确保路径是绝对路径，避免执行时找不到
            f.write(f'bash "{script}" &\n')

        f.write("# 可选：等待所有后台任务完成\n")
        f.write("# wait\n")
        f.write('echo "All scripts launched."\n')

    # 添加可执行权限
    os.chmod(run_script_path, 0o755)
    print(f"✅ 已生成并赋予执行权限: {os.path.abspath(run_script_path)}")