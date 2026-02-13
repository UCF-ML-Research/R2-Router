#!/bin/bash
#SBATCH --job-name=r2-gpt4o
#SBATCH --output=logs/inference_gpt4o_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=12:00:00
#SBATCH --qos=qi855292.ucf

# CPU-only job for GPT-4o API inference via OpenRouter

export PYTHONUNBUFFERED=1
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY}"

cd /home/ah872032.ucf/jiaqi/router

# Run GPT-4o on ALL queries
SEPARATE_FILE="/home/ah872032.ucf/jiaqi/RouterArena/router_inference/predictions/gpt4o-all.json"
cp /home/ah872032.ucf/jiaqi/RouterArena/router_inference/predictions/r2-router-clean.json "$SEPARATE_FILE"

.venv/bin/python scripts/inference_api.py \
    --model gpt-4o \
    --prediction-file "$SEPARATE_FILE" \
    --all-queries \
    --save-every 50
