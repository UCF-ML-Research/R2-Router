#!/bin/bash
#SBATCH --job-name=r2-eval-all
#SBATCH --output=logs/eval_all_%j.log
#SBATCH --error=logs/eval_all_%j.log
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --account=qi855292.ucf
#SBATCH --qos=qi855292.ucf

echo "=========================================="
echo "R2-Router → Evaluate All Models"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

export PYTHONUNBUFFERED=1
export PATH="/home/ah872032.ucf/jiaqi/router/.venv/bin:$PATH"

cd /home/ah872032.ucf/jiaqi/router

PRED_DIR="/home/ah872032.ucf/jiaqi/RouterArena/router_inference/predictions"

# Model list: name|file pairs
# Each model runs in a separate subprocess to avoid memory leak accumulation
MODELS=(
    "gemini-flash|$PRED_DIR/gemini-flash-all.json"
    "gpt4o|$PRED_DIR/gpt4o-all.json"
    "ministral-3b|$PRED_DIR/ministral3b-all.json"
    "ministral-8b|$PRED_DIR/ministral8b-all.json"
    "ministral-14b|$PRED_DIR/ministral14b-all.json"
    "235b-chat|$PRED_DIR/235b-all-chat.json"
    "haiku|$PRED_DIR/haiku-all.json"
    "coder-30b|$PRED_DIR/coder30b-all.json"
)

for entry in "${MODELS[@]}"; do
    model="${entry%%|*}"
    pred_file="${entry##*|}"
    echo ""
    echo "######################################################################"
    echo "# Evaluating: $model"
    echo "# File: $pred_file"
    echo "# Time: $(date)"
    echo "######################################################################"

    # Run in a subshell so Python process fully exits and memory is freed
    (
        exec .venv/bin/python scripts/eval_model.py \
            --pred-file "$pred_file" \
            --model-name "$model" \
            --include-code
    )
    exit_code=$?

    echo ""
    echo ">>> $model done (exit=$exit_code) at $(date)"
    echo ""
done

echo "=========================================="
echo "All evaluations complete!"
echo "End: $(date)"
echo "=========================================="
