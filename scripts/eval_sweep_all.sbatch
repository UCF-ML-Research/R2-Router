#!/bin/bash
#SBATCH --job-name=eval-sweep
#SBATCH --output=logs/eval_sweep_%j.log
#SBATCH --error=logs/eval_sweep_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=12:00:00
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf

echo "=========================================="
echo "Eval Sweep — All models (no code_accuracy)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

export PYTHONUNBUFFERED=1
export HF_HOME=/orange/qi855292.ucf/ah872032.ucf/cache/huggingface
cd /home/ah872032.ucf/jiaqi/router

SWEEP=/orange/qi855292.ucf/ah872032.ucf/budget_sweep
PY=.venv/bin/python
SCRIPT=scripts/eval_sweep.py

echo ""
echo "=== 30b ==="
$PY $SCRIPT --json-dir $SWEEP/30b --model-cost-name "qwen/qwen3-30b-a3b-instruct-2507"

echo ""
echo "=== 80b ==="
$PY $SCRIPT --json-dir $SWEEP/80b --model-cost-name "qwen/qwen3-next-80b-a3b-instruct"

echo ""
echo "=== 235b ==="
$PY $SCRIPT --json-dir $SWEEP/235b --model-cost-name "qwen/qwen3-235b-a22b-2507"

echo ""
echo "=== ministral-3b ==="
$PY $SCRIPT --json-dir $SWEEP/ministral-3b --model-cost-name "mistralai/ministral-3b"

echo ""
echo "=== ministral-8b ==="
$PY $SCRIPT --json-dir $SWEEP/ministral-8b --model-cost-name "mistralai/ministral-8b"

echo ""
echo "=== ministral-14b ==="
$PY $SCRIPT --json-dir $SWEEP/ministral-14b --model-cost-name "mistralai/ministral-14b"

echo ""
echo "=== coder-30b ==="
$PY $SCRIPT --json-dir $SWEEP/coder-30b --model-cost-name "qwen/qwen3-coder-30b-a3b-instruct"

echo ""
echo "=== coder-next ==="
$PY $SCRIPT --json-dir $SWEEP/Qwen3-Coder-Next --model-cost-name "Qwen/Qwen3-Coder-Next"

echo ""
echo "=== gpt4o ==="
$PY $SCRIPT --json-dir $SWEEP/gpt4o --model-cost-name "gpt-4o"

echo ""
echo "=== gemini-flash ==="
$PY $SCRIPT --json-dir $SWEEP/gemini-flash --model-cost-name "gemini-2.0-flash-001"

echo ""
echo "=== haiku ==="
$PY $SCRIPT --json-dir $SWEEP/haiku --model-cost-name "claude-3-haiku-20240307"

echo ""
echo "=== All done: $(date) ==="
