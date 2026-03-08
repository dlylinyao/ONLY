#!/bin/bash -l
#SBATCH --job-name=llm_judge           # Job name
#SBATCH --output=logs/llm_%j.out       # Stdout output file
#SBATCH --error=logs/llm_%j.err        # Stderr output file
#SBATCH --partition=gpu                # Puhti 的 GPU 队列
#SBATCH --ntasks=1                     # One task
#SBATCH --cpus-per-task=8              # Number of cores
#SBATCH --time=02:00:00                # Run time (放宽到2小时确保跑完)
#SBATCH --account=project_2017554      # 你的 Puhti 项目号
#SBATCH --gres=gpu:v100:1              # 申请 1 张 V100 显卡
#SBATCH --mem=64G                      # 内存 (稍微给大点，防止大模型加载时卡住)


module load pytorch


export HF_HOME="/scratch/project_2017554/linyaodu_cache"

modelname=$1


python llm-as-a-judge.py $modelname