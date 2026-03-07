#!/bin/bash -l
#SBATCH --job-name=llm_      # Job name
#SBATCH --output=logs/llm_%j.out      # Stdout output file
#SBATCH --error=logs/llm_%j.err     # Stderr output file
#SBATCH --partition=small-g              # Partition name
#SBATCH --ntasks=1                       # One task
#SBATCH --cpus-per-task=8                # Number of cores
#SBATCH --time=00:30:00                   # Run time
#SBATCH --account=project_465001864      # Billing project
#SBATCH --gpus-per-node=1                # Request 1 GPU
#SBATCH --mem=120G

module load cray-python
module load LUMI/22.08 partition/G rocm/5.2.3
source /scratch/project_462000764/dayyan/model_training/venv/bin/activate
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

modelname=$1

python llm-as-a-judge.py $modelname