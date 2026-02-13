#!/bin/bash
#SBATCH --job-name=knn-sub10
#SBATCH --output=logs/sweep_sub10_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00

echo "=== Global KNN Sweep with sub_10 training set ==="
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

echo "========== sub_10: Pool 12 (235b + gpt4o + gemini-flash) =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gpt4o gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS --use-sub10

echo ""
echo "========== sub_10: 235b + coder-next + gemini-flash =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b coder-next gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS --use-sub10

echo ""
echo "========== sub_10: 4-model (235b + 80b + flash + haiku) =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b 80b gemini-flash haiku \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS --use-sub10

echo ""
echo "========== sub_10: all models =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS --use-sub10

echo ""
echo "========== Comparison: StratifiedShuffleSplit 10% Pool 12 =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gpt4o gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS --train-size 0.1

echo ""
echo "=== All sweeps done ==="
echo "End: $(date)"
