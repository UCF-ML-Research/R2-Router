#!/bin/bash
#SBATCH --job-name=gknn-sweep
#SBATCH --output=logs/sweep_global_knn_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=04:00:00

echo "=== Global KNN Lambda Sweep (4 configs) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd /home/ah872032.ucf/jiaqi/router
source scripts/env.sh
export PYTHONUNBUFFERED=1

BETAS="0.05 0.1 19.0"
EXCL="budget_unlimited budget_1500"

echo "========== 4-model (235b, 80b, flash, haiku) =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b 80b gemini-flash haiku \
    --exclude-budgets $EXCL --betas $BETAS

echo ""
echo "========== 5-model (+ gpt-5.2) =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b 80b gemini-flash haiku gpt-5.2 \
    --exclude-budgets $EXCL --betas $BETAS

echo ""
echo "========== 9-model (no gpt4o/haiku) =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b 30b 80b coder-30b coder-next gemini-flash ministral-14b ministral-3b ministral-8b \
    --exclude-budgets $EXCL --betas $BETAS

echo ""
echo "========== 11-model (all except gemma-3n) =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b 30b 80b coder-30b coder-next gemini-flash gpt4o haiku ministral-14b ministral-3b ministral-8b \
    --exclude-budgets $EXCL --betas $BETAS

echo ""
echo "=== All sweeps done ==="
echo "End: $(date)"
