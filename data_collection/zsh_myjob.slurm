#!/bin/bash
#SBATCH --job-name=judge_job          # 作业名称
#SBATCH --partition=hpg-b200          # 分区
#SBATCH --nodes=1                     # 节点数
#SBATCH --ntasks=1                    # 总任务数
#SBATCH --cpus-per-task=6             # 每个任务的 CPU 核数
#SBATCH --mem=64G                     # 内存
#SBATCH --time=12:00:00               # 最长运行时间
#SBATCH --gres=gpu:2                  # 申请 2 张 GPU
#SBATCH --output=slurm-%j.out         # 标准输出文件
#SBATCH --error=slurm-%j.err          # 错误输出文件

cd /home/ji757406.ucf/router

bash zsh_dataset_1.sh
bash zsh_judge_1.sh

# bash zsh_dataset_2.sh
# bash zsh_judge_2.sh