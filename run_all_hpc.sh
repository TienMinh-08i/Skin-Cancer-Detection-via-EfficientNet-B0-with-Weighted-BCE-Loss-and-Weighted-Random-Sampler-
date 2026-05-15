#!/bin/bash
# HPC sbatch script to run all experiments on a GPU node
#SBATCH --job-name=skin_cancer_ablation
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --nodelist=hpc23
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/ablation_%j.log

module load cuda/11.8
cd "$(dirname "$0")/.."

# Generate experiment configs
python scripts/generate_experiments.py \
    --base-config configs/config_hpc.yaml \
    --output-dir configs/experiments

# Run all experiments
python scripts/run_all_experiments.py \
    --device cuda \
    --resume

echo "All experiments completed!"
