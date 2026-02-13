#!/bin/bash
#SBATCH --job-name=knn-20pct
#SBATCH --output=logs/sweep_20pct_4model_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=01:00:00

cd /home/ah872032.ucf/jiaqi/router
source scripts/env.sh
export PYTHONUNBUFFERED=1

echo "=== 20% train: 235b + 80b + gpt4o + gemini-flash ==="

.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b 80b gpt4o gemini-flash \
    --exclude-budgets budget_unlimited budget_1500 \
    --betas 0.05 0.1 19.0 \
    --lambdas 0.9 0.95 0.98 0.99 0.995 0.999 0.9995 0.9999 1.0 \
    --train-size 0.2

echo "Done: $(date)"
