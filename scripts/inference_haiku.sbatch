#!/bin/bash
#SBATCH --job-name=r2-haiku
#SBATCH --output=logs/inference_haiku_%j.log
#SBATCH --error=logs/inference_haiku_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=12:00:00
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf

echo "=========================================="
echo "R2-Router → Claude Haiku 4.5 (OpenRouter)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

export PYTHONUNBUFFERED=1
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY}"
export PATH="/home/ah872032.ucf/jiaqi/router/.venv/bin:$PATH"

cd /home/ah872032.ucf/jiaqi/router
mkdir -p logs

SEPARATE_FILE="/home/ah872032.ucf/jiaqi/RouterArena/router_inference/predictions/haiku-all.json"
# Only copy clean template if file doesn't exist (preserve checkpoint for resume)
[ ! -f "$SEPARATE_FILE" ] && cp /home/ah872032.ucf/jiaqi/RouterArena/router_inference/predictions/r2-router-clean.json "$SEPARATE_FILE"
.venv/bin/python scripts/inference_api.py \
    --model claude-haiku-4-5 \
    --prediction-file "$SEPARATE_FILE" \
    --all-queries \
    --save-every 50

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
