#!/bin/bash
#SBATCH --job-name=build-data
#SBATCH --output=logs/build_data_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:00:00

echo "=== Build Category Training Data ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd /home/ah872032.ucf/jiaqi/router
source scripts/env.sh
export PYTHONUNBUFFERED=1

.venv/bin/python scripts/build_category_training_data.py

echo ""
echo "=== Done ==="
echo "End: $(date)"
