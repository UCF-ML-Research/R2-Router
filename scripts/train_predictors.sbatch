#!/bin/bash
#SBATCH --job-name=cat-predict
#SBATCH --output=logs/predictors_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00

echo "=== Category Predictor Training (KNN, K-fold CV) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd /home/ah872032.ucf/jiaqi/router

source scripts/env.sh
export PYTHONUNBUFFERED=1

# Clear stale JIT caches
rm -rf ${CACHE_ROOT}/flashinfer/*
rm -rf ${CACHE_ROOT}/vllm/torch_compile_cache/*

.venv/bin/python scripts/train_category_predictors.py --k 5 --full

echo ""
echo "=== Done ==="
echo "End: $(date)"
