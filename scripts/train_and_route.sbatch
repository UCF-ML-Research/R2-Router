#!/bin/bash
#SBATCH --job-name=r2-train-route
#SBATCH --output=logs/train_and_route_%j.log
#SBATCH --error=logs/train_and_route_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf

echo "=========================================="
echo "R2-Router: FULL-DATA Retrain Ridge(alpha=1.0) + Route (λ=0.999, sk=4.0, 4-model)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

export PYTHONUNBUFFERED=1
export PATH="/home/ah872032.ucf/jiaqi/router/.venv/bin:$PATH"

cd /home/ah872032.ucf/jiaqi/router
mkdir -p logs

echo ""
echo "=== Step 1: FULL-DATA train predictors (Ridge alpha=1.0, no PCA, ridge-only) ==="
.venv/bin/python scripts/train_category_predictors.py \
    --alpha 1.0 \
    --ridge-only \
    --pca 0 \
    --full

echo ""
echo "=== Step 2: Route RouterArena queries (λ=0.999, sk=4.0, 4-model) ==="
.venv/bin/python scripts/route_routerarena.py \
    --lambda_val 0.999 \
    --shrinkage_k 4.0 \
    --skip_embedding \
    --exclude_models gpt4o coder-next coder-30b ministral-3b ministral-8b ministral-14b 30b \
    --exclude_budgets budget_unlimited budget_1500

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
