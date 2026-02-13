#!/bin/bash
# =============================================================================
# Unified budget sweep template (sweep + concise in one GPU load)
#
# Usage:
#   sbatch --partition=hpg-b200 --gres=gpu:4 \
#       --export=MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507",MODEL_SHORT=235b,TP_SIZE=4 \
#       scripts/sweep_model.sbatch
#
#   sbatch --partition=hpg-b200 --gres=gpu:1 \
#       --export=MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507",MODEL_SHORT=30b \
#       scripts/sweep_model.sbatch
#
#   # With custom sampling params (e.g., Gemma 3N):
#   sbatch --partition=hpg-b200 --gres=gpu:1 --mem=64gb \
#       --export=MODEL="google/gemma-3n-E4B-it",MODEL_SHORT=gemma-3n-e4b,SAMPLING_ARGS="--temperature 1.0 --top-p 0.95 --top-k 64 --repetition-penalty 1.0 --block-size 32" \
#       scripts/sweep_model.sbatch
#
# Required env vars:
#   MODEL       — HuggingFace model ID
#   MODEL_SHORT — Short name for output folder
#
# Optional env vars:
#   TP_SIZE       — Tensor parallel size (default: 1)
#   BATCH_SIZE    — Batch size (default: 256)
#   TIME_LIMIT    — Job time limit (default: 12:00:00)
#   BUDGETS       — Comma-separated budgets (default: all)
#   SAMPLING_ARGS — Extra sampling args (e.g., "--temperature 1.0 --top-k 64")
#   EXTRA_ARGS    — Any other extra args
# =============================================================================

#SBATCH --job-name=sweep
#SBATCH --output=logs/sweep_%j.log
#SBATCH --error=logs/sweep_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf
#SBATCH --exclude=c1000a-s15

echo "=========================================="
echo "Budget Sweep → ${MODEL} (${MODEL_SHORT})"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ', ')"
echo "Start: $(date)"
echo "=========================================="

cd /home/ah872032.ucf/jiaqi/router

source scripts/env.sh
export PYTHONUNBUFFERED=1

# Clear stale JIT caches
rm -rf ${CACHE_ROOT}/flashinfer/*
rm -rf ${CACHE_ROOT}/vllm/torch_compile_cache/*

# Build command
CMD=".venv/bin/python scripts/inference_budget_sweep.py \
    --model \"$MODEL\" \
    --model-short \"$MODEL_SHORT\" \
    --output-dir /orange/qi855292.ucf/ah872032.ucf/budget_sweep/$MODEL_SHORT \
    --tp-size ${TP_SIZE:-1} \
    --batch-size ${BATCH_SIZE:-256}"

# Add budgets if specified
if [ -n "$BUDGETS" ]; then
    CMD="$CMD --budgets \"$BUDGETS\""
fi

# Add sampling args if specified
if [ -n "$SAMPLING_ARGS" ]; then
    CMD="$CMD $SAMPLING_ARGS"
fi

# Add extra args if specified
if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
