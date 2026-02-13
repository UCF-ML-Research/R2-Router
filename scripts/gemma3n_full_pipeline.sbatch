#!/bin/bash
#SBATCH --job-name=gemma3n-full
#SBATCH --output=logs/gemma3n_full_%j.log
#SBATCH --error=logs/gemma3n_full_%j.log
#SBATCH --partition=hpg-b200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96gb
#SBATCH --time=48:00:00
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf

echo "=========================================="
echo "Gemma 3N E4B IT Full Pipeline (sweep + code regen + eval)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo "=========================================="

cd /home/ah872032.ucf/jiaqi/router

source scripts/env.sh
export PYTHONUNBUFFERED=1

# Clear stale caches
rm -rf ${CACHE_ROOT}/flashinfer/*
rm -rf ${CACHE_ROOT}/vllm/torch_compile_cache/*

# Model config
MODEL="google/gemma-3n-E4B-it"
MODEL_SHORT="gemma-3n-e4b"
OUTPUT_DIR="/orange/qi855292.ucf/ah872032.ucf/budget_sweep/${MODEL_SHORT}"

echo ""
echo "=========================================="
echo "Step 1: Budget Sweep + Code Regen (single GPU load)"
echo "=========================================="
# Gemma 3n official recommended params:
# temperature=1.0, top_k=64, top_p=0.95, repetition_penalty=1.0
.venv/bin/python scripts/inference_budget_sweep.py \
    --model "$MODEL" \
    --model-short "$MODEL_SHORT" \
    --output-dir "$OUTPUT_DIR" \
    --tp-size 1 \
    --batch-size 256 \
    --code-regen \
    --temperature 1.0 \
    --top-p 0.95 \
    --top-k 64 \
    --repetition-penalty 1.0 \
    --block-size 32

if [ $? -ne 0 ]; then
    echo "ERROR: Budget sweep failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Evaluation"
echo "=========================================="
.venv/bin/python scripts/eval_sweep.py \
    --json-dir "$OUTPUT_DIR" \
    --model-cost-name "google/gemma-3n-e4b-it" \
    --include-code

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "End: $(date)"
echo "Output: $OUTPUT_DIR"
echo "Files: $(ls -1 $OUTPUT_DIR | wc -l) JSON files"
echo "=========================================="
