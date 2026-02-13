#!/bin/bash
#SBATCH --job-name=knn-pool-new
#SBATCH --output=logs/sweep_pool_new_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00

echo "=== Global KNN: New Pool Sweep ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd /home/ah872032.ucf/jiaqi/router
source scripts/env.sh
export PYTHONUNBUFFERED=1

BETAS="0.05 0.1 19.0"
EXCL="budget_unlimited budget_1500"
LAMBDAS="0.8 0.85 0.9 0.92 0.94 0.95 0.96 0.97 0.98 0.99 0.995 0.999 1.0"

echo "========== Pool: 235b + 80b + gemini-flash + gpt-5.2 + coder-next =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b 80b gemini-flash gpt-5.2 coder-next \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "=== Done ==="
echo "End: $(date)"
