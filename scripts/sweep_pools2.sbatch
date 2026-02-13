#!/bin/bash
#SBATCH --job-name=knn-pools2
#SBATCH --output=logs/sweep_pools2_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=04:00:00

echo "=== Global KNN Pool+Lambda Sweep (round 2) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd /home/ah872032.ucf/jiaqi/router
source scripts/env.sh
export PYTHONUNBUFFERED=1

BETAS="0.05 0.1 19.0"
EXCL="budget_unlimited budget_1500"
# Finer lambda grid around the sweet spot (0.8~0.999)
LAMBDAS="0.8 0.85 0.9 0.92 0.94 0.95 0.96 0.97 0.98 0.99 0.995 0.999 1.0"

echo "========== Pool A: 235b + gpt4o + coder-next + gemini-flash =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gpt4o coder-next gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool B: 235b + gpt4o + coder-next =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gpt4o coder-next \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool C: 235b + gpt4o =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gpt4o \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool D: 235b + gpt-5.2 + gemini-flash =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gpt-5.2 gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool E: 235b + gpt-5.2 + coder-next + gemini-flash =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gpt-5.2 coder-next gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool F: 235b + gpt4o + gpt-5.2 + gemini-flash =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gpt4o gpt-5.2 gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool G: 235b + coder-next + gpt4o =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b coder-next gpt4o \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool H: 235b + gemini-3-pro + gemini-flash =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gemini-3-pro gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool I: 235b + glm-5 + gemini-flash =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b glm-5 gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool J: 235b + gpt4o + gemini-flash (finer lambda, reference) =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b gpt4o gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool K: coder-next + gpt4o + gemini-flash =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models coder-next gpt4o gemini-flash \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "========== Pool L: 235b + coder-next + gpt4o + gemini-3-pro =========="
.venv/bin/python scripts/sweep_lambda_global_knn.py \
    --models 235b coder-next gpt4o gemini-3-pro \
    --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS

echo ""
echo "=== All sweeps done ==="
echo "End: $(date)"
