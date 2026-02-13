#!/bin/bash
# =============================================================================
# Download HuggingFace model to Orange cache (CPU-only)
#
# Usage:
#   sbatch --export=MODEL="moonshotai/Kimi-K2.5" scripts/download_model.sbatch
# =============================================================================

#SBATCH --job-name=dl-model
#SBATCH --output=logs/download_%j.log
#SBATCH --error=logs/download_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf

echo "=========================================="
echo "Download Model → ${MODEL}"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

cd /home/ah872032.ucf/jiaqi/router

source scripts/env.sh
export PYTHONUNBUFFERED=1

echo "HF_HOME: $HF_HOME"
echo ""

.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
print(f'Downloading: ${MODEL}')
print(f'Cache dir: {os.environ[\"HF_HOME\"]}')
path = snapshot_download('${MODEL}', cache_dir=os.environ['HF_HOME'])
print(f'Downloaded to: {path}')
print('Done!')
"

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
