#!/bin/bash
#SBATCH --job-name=pool-sweep
#SBATCH --output=logs/sweep_pools_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=04:00:00

echo "=== LLM Pool Sweep (K=28) ==="
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

echo "========== Pool 1: 235b only =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool 2: 235b + coder-next =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b coder-next \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool 3: 235b + gemini-flash =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool 4: 235b + gpt4o =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gpt4o \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool 5: 235b + coder-next + gemini-flash =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b coder-next gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool 6: 235b + coder-next + gpt4o =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b coder-next gpt4o \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool 7: 235b + 80b + coder-next =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b 80b coder-next \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool 8: 235b + 80b + coder-next + gemini-flash =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b 80b coder-next gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool 9: 235b + coder-next + gemini-flash + haiku =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b coder-next gemini-flash haiku \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool 10: 235b + 80b + coder-next + gemini-flash + haiku =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b 80b coder-next gemini-flash haiku \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool 11: all 12 models =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool 12: 235b + gpt4o + gemini-flash =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gpt4o gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "=== All pool sweeps done ==="
echo "End: $(date)"
