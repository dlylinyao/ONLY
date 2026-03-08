#!/bin/bash -l
#SBATCH --job-name=llm_judge
#SBATCH --output=logs/llm_%j.out
#SBATCH --error=logs/llm_%j.err
#SBATCH --account=project_2017554
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1


mkdir -p logs


module purge
module load pytorch/2.4
export HF_HOME="/scratch/project_2017554/linyaodu_cache"

modelname=$1


python llm-as-a-judge.py $modelname