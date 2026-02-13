#!/bin/bash
#SBATCH --job-name=knn-k-sweep
#SBATCH --output=logs/sweep_knn_k_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=04:00:00

echo "=== KNN K Value Sweep ==="
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

# 4-model pool — sweep K values
for K in 3 5 10 15 20 28 50 100 200; do
    echo ""
    echo "========== 4-model, K=$K =========="
    .venv/bin/python scripts/sweep_lambda_global_knn.py \
        --models 235b 80b gemini-flash haiku \
        --exclude-budgets $EXCL --betas $BETAS --lambdas $LAMBDAS --k $K
done

echo ""
echo "=== All K sweeps done ==="
echo "End: $(date)"
