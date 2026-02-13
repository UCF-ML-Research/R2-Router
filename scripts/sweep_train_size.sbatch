#!/bin/bash
#SBATCH --job-name=train-size-sweep
#SBATCH --output=logs/sweep_train_size_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=04:00:00

echo "=== Train Size Sweep (10% vs 20%) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd /home/ah872032.ucf/jiaqi/router
source scripts/env.sh
export PYTHONUNBUFFERED=1

BETAS="0.05 0.1 19.0"
EXCL="budget_unlimited budget_1500"
LAMBDAS="0.9 0.95 0.98 0.99 0.995 0.999 0.9995 0.9999 1.0"

echo "========== 20% train: Pool 12 (235b + gpt4o + gemini-flash) =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gpt4o gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS --train-size 0.2

echo ""
echo "========== 20% train: 4-model (235b + 80b + flash + haiku) =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b 80b gemini-flash haiku \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS --train-size 0.2

echo ""
echo "========== 20% train: 235b only =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS --train-size 0.2

echo ""
echo "========== 20% train: 235b + coder-next + gemini-flash =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b coder-next gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS --train-size 0.2

echo ""
echo "========== 20% train: all 12 models =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS --train-size 0.2

echo ""
echo "=== For comparison: 10% train Pool 12 (should match prior results) ==="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gpt4o gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS --train-size 0.1

echo ""
echo "=== All sweeps done ==="
echo "End: $(date)"
