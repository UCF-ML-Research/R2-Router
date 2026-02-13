#!/bin/bash
#SBATCH --job-name=r2-mini14b
#SBATCH --output=logs/inference_ministral14b_%j.log
#SBATCH --error=logs/inference_ministral14b_%j.log
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=06:00:00
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf
#SBATCH --exclude=c1000a-s15

echo "=========================================="
echo "R2-Router → Ministral-3-14B"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

export PYTHONUNBUFFERED=1
export HF_HOME=/orange/qi855292.ucf/ah872032.ucf/cache/huggingface
export HF_TOKEN="${HF_TOKEN}"
export VLLM_CACHE_ROOT=/orange/qi855292.ucf/ah872032.ucf/cache/vllm
export TORCH_EXTENSIONS_DIR=$SLURM_TMPDIR/torch_extensions
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="/home/ah872032.ucf/jiaqi/router/.venv/bin:$CUDA_HOME/bin:$PATH"

cd /home/ah872032.ucf/jiaqi/router
mkdir -p logs

SEPARATE_FILE="/home/ah872032.ucf/jiaqi/RouterArena/router_inference/predictions/ministral14b-all.json"
cp /home/ah872032.ucf/jiaqi/RouterArena/router_inference/predictions/r2-router-clean.json "$SEPARATE_FILE"
.venv/bin/python scripts/inference_routerarena.py \
    --model "mistralai/Ministral-3-14B-Instruct-2512" \
    --prediction-file "$SEPARATE_FILE" \
    --all-queries \
    --batch-size 256

echo "=== End: $(date) ==="
