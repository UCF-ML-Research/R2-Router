#!/bin/bash
#SBATCH --job-name=cat-classifier
#SBATCH --output=logs/classifier_%j.log
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --exclude=c1000a-s15

echo "=== Category Classifier Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd /home/ah872032.ucf/jiaqi/router

# Environment
source scripts/env.sh
export PYTHONUNBUFFERED=1

# Run classifier training (CPU-only sklearn, embeddings already cached)
.venv/bin/python scripts/train_category_classifier.py --k 5

echo ""
echo "=== Done ==="
echo "End: $(date)"
