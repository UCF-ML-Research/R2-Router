#!/bin/bash
# =============================================================================
# Budget sweep via OpenRouter API (CPU-only, no GPU needed)
#
# Usage:
#   sbatch --export=MODEL="z-ai/glm-5",MODEL_SHORT=glm-5 scripts/sweep_api.sbatch
#   sbatch --export=MODEL="z-ai/glm-5",MODEL_SHORT=glm-5,BUDGETS=concise scripts/sweep_api.sbatch
#
# Required env vars:
#   MODEL       — OpenRouter model ID
#   MODEL_SHORT — Short name for output folder
#
# Optional env vars:
#   BUDGETS       — Comma-separated budgets (default: concise)
#   TEMPERATURE   — Sampling temperature (default: 1.0)
#   TIME_LIMIT    — Job time limit (default: 24:00:00)
# =============================================================================

#SBATCH --job-name=sweep-api
#SBATCH --output=logs/sweep_api_%j.log
#SBATCH --error=logs/sweep_api_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=24:00:00
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf

echo "=========================================="
echo "API Sweep → ${MODEL} (${MODEL_SHORT})"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

cd /home/ah872032.ucf/jiaqi/router

source scripts/env.sh
export PYTHONUNBUFFERED=1
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY}"

CMD=".venv/bin/python scripts/inference_budget_sweep.py \
    --api \
    --model \"$MODEL\" \
    --model-short \"$MODEL_SHORT\" \
    --output-dir /orange/qi855292.ucf/ah872032.ucf/budget_sweep/$MODEL_SHORT \
    --budgets \"${BUDGETS:-concise}\" \
    --save-every 1"

if [ -n "$TEMPERATURE" ]; then
    CMD="$CMD --temperature $TEMPERATURE"
fi

echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
