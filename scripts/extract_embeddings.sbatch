#!/bin/bash
#SBATCH --job-name=extract_emb
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:b200:1
#SBATCH --exclude=c1000a-s15
#SBATCH --output=logs/extract_emb_%j.out
#SBATCH --error=logs/extract_emb_%j.err

# ============================================================================
# Extract embeddings for R2-Router training data
# Uses Qwen3-0.6B via vLLM with 1x B200 GPU
#
# Usage:
#   cd ~/jiaqi/router
#   mkdir -p logs
#   sbatch scripts/extract_embeddings.sbatch
#
# Monitor:
#   tail -f logs/extract_emb_<JOBID>.out
#   squeue -u $USER
# ============================================================================

set -euo pipefail

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Start:  $(date)"
echo "=========================================="

cd /home/ah872032.ucf/jiaqi/router

# Environment setup (uv)
export PATH="$HOME/.local/bin:$PATH"

# Load CUDA runtime matching driver (570.x supports up to CUDA 12.8)
# Default "module load cuda" loads 13.0.2 which is too new for the driver
module load cuda/12.8.1

# Prevent CUDA context from being initialized before vLLM forks
export CUDA_MODULE_LOADING=LAZY

# Use /orange for data and all caches (persistent, large storage)
ORANGE_ROOT=/orange/qi855292.ucf/ah872032.ucf
DATA_DIR=${ORANGE_ROOT}/data
CACHE_ROOT=${ORANGE_ROOT}/cache
export HF_HOME=${CACHE_ROOT}/huggingface
export VLLM_CACHE_ROOT=${CACHE_ROOT}/vllm
export FLASHINFER_CACHE_DIR=${CACHE_ROOT}/flashinfer
export TORCH_HOME=${CACHE_ROOT}/torch
export TORCH_EXTENSIONS_DIR=${CACHE_ROOT}/torch/extensions

# Clear stale caches that may reference old job scratch dirs
rm -rf ${CACHE_ROOT}/flashinfer/*
rm -rf ${CACHE_ROOT}/vllm/torch_compile_cache

# Run embedding extraction
# Uses first CSV for prompts (all 9 share the same queries)
# Passes all 9 CSVs for consistency check
uv run python data_collection/extract_embeddings.py \
    --csv_files \
        ${DATA_DIR}/DeepSeek-V3.1.csv \
        ${DATA_DIR}/gemma-3-4b-it.csv \
        ${DATA_DIR}/GLM-4.5-Air.csv \
        ${DATA_DIR}/GLM-4.6.csv \
        ${DATA_DIR}/Mistral-7B-Instruct-v0.2.csv \
        ${DATA_DIR}/Qwen3-0.6B.csv \
        ${DATA_DIR}/Qwen3-235B-A22B-Instruct-2507.csv \
        ${DATA_DIR}/Qwen3-30B-A3B-Instruct-2507.csv \
        ${DATA_DIR}/Qwen3-Next-80B-A3B-Instruct.csv \
    --output_path ${DATA_DIR}/prompt_embeddings.pkl \
    --encoder_name Qwen/Qwen3-0.6B \
    --tensor_parallel_size 1 \
    --batch_size 512

echo "=========================================="
echo "End:    $(date)"
echo "=========================================="
