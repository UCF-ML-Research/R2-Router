#!/bin/bash
# Complete pipeline: 1) Define LLM pool, 2) Train R2-Router, 3) Train CARROT, 4) Compare
#
# Checkpoint behavior:
# - R2-Router: Per-model checkpoints. Only retrain if model+scheme combination doesn't exist.
# - CARROT: Pool-level checkpoints. Retrain if model pool OR training scheme changes.

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# CONFIGURATION - Edit this section
# ============================================================================

# Define your LLM pool here (format: "Name|Size|CSVPath")
# Checkpoint paths are automatically generated based on training scheme
LLM_POOL=(
    "Mistral-7B-Instruct-v0.2|0.20|data/Mistral-7B-Instruct-v0.2.csv"
    "GLM-4.5-Air|0.85|data/GLM-4.5-Air.csv"
    # "GLM-4.6|1.75|data/GLM-4.6.csv"
    # "DeepSeek-3.1|0.80|data/DeepSeek-V3.1.csv"
    "gemma-3-4b-it|0.06815|data/gemma-3-4b-it.csv"
    "gemma-3-1b-it|0.0170375|data/gemma-3-1b-it.csv"
    "gemma-3-270m-it|0.004259375|data/gemma-3-270m-it.csv"
    "Llama-3.1-70B-Instruct|0.40|data/Llama-3.1-70B-Instruct.csv"
    "Llama-3.2-3B-Instruct|0.02|data/Llama-3.2-3B-Instruct.csv"
    "Qwen2.5-Math-1.5B-Instruct|0.09|data/Qwen2.5-Math-1.5B-Instruct.csv"
    "Qwen2.5-Math-7B-Instruct|0.35|data/Qwen2.5-Math-7B-Instruct.csv"
    "Qwen3-0.6B|0.0173|data/Qwen3-0.6B.csv"
    # "Qwen3-235B-A22B-Instruct-2507|0.55|data/Qwen3-235B-A22B-Instruct-2507.csv"
    # "Qwen3-Next-80B-A3B-Instruct|0.6|data/Qwen3-Next-80B-A3B-Instruct.csv"
)

# R2-Router Hyperparameters - Edit these to tune prediction precision
# Model types: "linear", "ridge", "lasso", "elasticnet", "random_forest", "gradient_boosting", "mlp", "torch_mlp"
CORE_MODEL_TYPE="ridge"        # Model type (ridge/lasso/mlp/torch_mlp)
CORE_ALPHA=10.0                # Regularization strength (for ridge/lasso/elasticnet)
CORE_L1_RATIO=0.5              # ElasticNet mixing (0=Ridge, 1=Lasso)
CORE_N_ESTIMATORS=100          # Number of trees (for tree-based models)
CORE_MAX_DEPTH=3               # Tree depth (for tree-based models)
CORE_HIDDEN_LAYERS="256, 128, 64"    # MLP hidden layer sizes (comma-separated, for mlp/torch_mlp)
CORE_MAX_ITER=1000             # Max iterations for sklearn MLP (default: 1000)
# PyTorch MLP specific parameters (only used when CORE_MODEL_TYPE="torch_mlp")
CORE_TORCH_EPOCHS=50          # Training epochs for torch_mlp (default: 100)
CORE_TORCH_LR=1e-4             # Learning rate for torch_mlp (default: 1e-4)
CORE_TORCH_DROPOUT=0.5         # Dropout rate for torch_mlp (default: 0.5)
CORE_TORCH_BATCH_SIZE=128      # Batch size for torch_mlp (default: 128)

# Lambda Distribution - Controls cost-performance tradeoff sampling
# Format: "min,max,num_points" for each segment (segments are concatenated)
# Formula: score = (1-λ)*quality - λ*cost, where λ ∈ [0,1]
# Examples:
#   "0,0.2,20;0.2,1.0,50"  - Default: denser sampling at low lambda (quality-focused)
#   "0,1.0,100"            - Uniform: evenly spaced across full range
#   "0,0.5,80;0.5,1.0,20"  - Dense low lambda, sparse high lambda
LAMBDA_DISTRIBUTION="0,0.001,100;0.001,0.01,100;0.01,0.1,100;0.1,1.0,100"

# QNC Target Accuracy Rate - Controls what percentage of best LLM to target
# 1.0 = 100% of best LLM (default), 0.9 = 90% of best LLM
TARGET_ACCURACY_RATE=0.95

# IRT Baseline Hyperparameters
IRT_LATENT_DIM=32         # Latent ability dimension (default: 32)
IRT_LR=3e-3               # Learning rate (default: 3e-3)
IRT_BATCH_SIZE=128        # Batch size (default: 128)
IRT_EPOCHS=10            # Number of epochs (default: 200)
IRT_DESCRIPTIONS=""       # Path to custom LLM descriptions JSON (optional, leave empty for auto-generation)

# Activate conda
source ~/miniconda3/bin/activate

# Generate training scheme suffix
if [ "$CORE_MODEL_TYPE" = "linear" ]; then
    SCHEME_SUFFIX="linear"
elif [ "$CORE_MODEL_TYPE" = "ridge" ]; then
    SCHEME_SUFFIX="ridge_alpha${CORE_ALPHA}"
elif [ "$CORE_MODEL_TYPE" = "lasso" ]; then
    SCHEME_SUFFIX="lasso_alpha${CORE_ALPHA}"
elif [ "$CORE_MODEL_TYPE" = "elasticnet" ]; then
    SCHEME_SUFFIX="elasticnet_alpha${CORE_ALPHA}_l1${CORE_L1_RATIO}"
elif [ "$CORE_MODEL_TYPE" = "random_forest" ]; then
    DEPTH_STR=$([ "$CORE_MAX_DEPTH" -eq 0 ] && echo "unlimited" || echo "d${CORE_MAX_DEPTH}")
    SCHEME_SUFFIX="rf_n${CORE_N_ESTIMATORS}_${DEPTH_STR}"
elif [ "$CORE_MODEL_TYPE" = "gradient_boosting" ]; then
    DEPTH_STR=$([ "$CORE_MAX_DEPTH" -eq 0 ] && echo "unlimited" || echo "d${CORE_MAX_DEPTH}")
    SCHEME_SUFFIX="gbm_n${CORE_N_ESTIMATORS}_${DEPTH_STR}"
elif [ "$CORE_MODEL_TYPE" = "mlp" ]; then
    LAYERS_STR=$(echo "$CORE_HIDDEN_LAYERS" | tr ',''_' | tr -d ' ')
    SCHEME_SUFFIX="mlp_${LAYERS_STR}"
elif [ "$CORE_MODEL_TYPE" = "torch_mlp" ]; then
    LAYERS_STR=$(echo "$CORE_HIDDEN_LAYERS" | tr ',' '_' | tr -d ' ')
    SCHEME_SUFFIX="torch_mlp_${LAYERS_STR}_ep${CORE_TORCH_EPOCHS}"
else
    SCHEME_SUFFIX="$CORE_MODEL_TYPE"
fi

# Build model arguments with auto-generated checkpoint paths
MODELS_ARG=()
for llm in "${LLM_POOL[@]}"; do
    IFS='|' read -r name size csv <<< "$llm"
    # Auto-generate checkpoint path: ./checkpoints/main/{name}_{scheme_suffix}
    checkpoint="./checkpoints/main/${name}_${SCHEME_SUFFIX}"
    MODELS_ARG+=(--model "$name" "$size" "$csv" "$checkpoint")
done

# ============================================================================
# Step 1: Train R2-Router Predictors (or load if already trained)
# ============================================================================

echo "=========================================="
echo "STEP 1: Checking R2-Router models (${#LLM_POOL[@]} models)"
echo "=========================================="
echo "Model Type: $CORE_MODEL_TYPE"
echo "Alpha: $CORE_ALPHA"
echo "Training Scheme: $SCHEME_SUFFIX"
echo "=========================================="

# Check which models need training
MODELS_TO_TRAIN=()
MODELS_ALREADY_TRAINED=()

for llm in "${LLM_POOL[@]}"; do
    IFS='|' read -r name size csv <<< "$llm"
    checkpoint="./checkpoints/main/${name}_${SCHEME_SUFFIX}"

    # Check if checkpoint exists and is complete
    # torch_mlp uses .pt files, others use .joblib
    if [ "$CORE_MODEL_TYPE" = "torch_mlp" ]; then
        if [ -d "$checkpoint" ] && \
           [ -f "$checkpoint/score_predictor.pt" ] && \
           [ -f "$checkpoint/token_predictor.joblib" ]; then
            echo "[SKIP] $name - Already trained (found at $checkpoint)"
            MODELS_ALREADY_TRAINED+=("$name")
        else
            echo "[TRAIN] $name - Will train"
            MODELS_TO_TRAIN+=(--model "$name" "$size" "$csv" "$checkpoint")
        fi
    else
        if [ -d "$checkpoint" ] && \
           [ -f "$checkpoint/limited_score_predictors.joblib" ] && \
           [ -f "$checkpoint/unlimited_score_predictor.joblib" ] && \
           [ -f "$checkpoint/unlimited_token_predictor.joblib" ]; then
            echo "[SKIP] $name - Already trained (found at $checkpoint)"
            MODELS_ALREADY_TRAINED+=("$name")
        else
            echo "[TRAIN] $name - Will train"
            MODELS_TO_TRAIN+=(--model "$name" "$size" "$csv" "$checkpoint")
        fi
    fi
done

echo ""
echo "Summary:"
echo "  Already trained: ${#MODELS_ALREADY_TRAINED[@]} models"
echo "  Need training: $((${#LLM_POOL[@]} - ${#MODELS_ALREADY_TRAINED[@]})) models"
echo ""

# Only train if there are models that need training
if [ ${#MODELS_TO_TRAIN[@]} -gt 0 ]; then
    echo "Training R2-Router models..."
    if [ "$CORE_MODEL_TYPE" = "torch_mlp" ]; then
        python -m main.r2.train_r2 \
            --model_type "$CORE_MODEL_TYPE" \
            --hidden_layers "$CORE_HIDDEN_LAYERS" \
            --torch_epochs "$CORE_TORCH_EPOCHS" \
            --torch_lr "$CORE_TORCH_LR" \
            --torch_dropout "$CORE_TORCH_DROPOUT" \
            --torch_batch_size "$CORE_TORCH_BATCH_SIZE" \
            "${MODELS_TO_TRAIN[@]}"
    else
        python -m main.r2.train_r2 \
            --model_type "$CORE_MODEL_TYPE" \
            --alpha "$CORE_ALPHA" \
            --l1_ratio "$CORE_L1_RATIO" \
            --n_estimators "$CORE_N_ESTIMATORS" \
            --max_depth "$CORE_MAX_DEPTH" \
            --hidden_layers "$CORE_HIDDEN_LAYERS" \
            --max_iter "$CORE_MAX_ITER" \
            "${MODELS_TO_TRAIN[@]}"
    fi
else
    echo "All R2-Router models already trained! Skipping training step."
fi

# ============================================================================
# Step 2: Train CARROT (or load if already trained)
# ============================================================================

echo ""
echo "=========================================="
echo "STEP 2: Training CARROT baselines"
echo "=========================================="

# CARROT models depend on the model pool and training scheme, so they need
# to be retrained whenever either changes. We store a config file to track this.

CONFIG_FILE="./checkpoints/main/carrot_config.txt"
CURRENT_CONFIG="${SCHEME_SUFFIX}|${LLM_POOL[*]}"

NEEDS_CARROT_TRAINING=true

if [ -f "$CONFIG_FILE" ]; then
    STORED_CONFIG=$(cat "$CONFIG_FILE")
    if [ "$CURRENT_CONFIG" = "$STORED_CONFIG" ]; then
        # Check if checkpoint files exist
        if [ -d "./checkpoints/main/carrot_knn" ] && \
           [ -f "./checkpoints/main/carrot_knn/knn_score.joblib" ] && \
           [ -f "./checkpoints/main/carrot_knn/knn_count.joblib" ] && \
           [ -d "./checkpoints/main/carrot_linear" ] && \
           [ -f "./checkpoints/main/carrot_linear/linear_score.joblib" ] && \
           [ -f "./checkpoints/main/carrot_linear/linear_count.joblib" ]; then
            echo "[SKIP] CARROT models already trained with current configuration"
            NEEDS_CARROT_TRAINING=false
        else
            echo "[TRAIN] CARROT checkpoints incomplete, retraining..."
        fi
    else
        echo "[TRAIN] CARROT configuration changed (model pool or training scheme), retraining..."
    fi
else
    echo "[TRAIN] No previous CARROT configuration found, training..."
fi

# Train CARROT if needed
if [ "$NEEDS_CARROT_TRAINING" = true ]; then
    echo "Training CARROT models..."
    python -m main.baselines.carrot.train_carrot "${MODELS_ARG[@]}"

    # Save configuration after successful training
    if [ $? -eq 0 ]; then
        mkdir -p ./checkpoints/main
        echo "$CURRENT_CONFIG" > "$CONFIG_FILE"
        echo "Saved CARROT configuration to $CONFIG_FILE"
    fi
fi

# ============================================================================
# Step 3: Train IRT Baselines (MIRT and NIRT, or load if already trained)
# ============================================================================

echo ""
echo "=========================================="
echo "STEP 3: Training IRT baselines (MIRT and NIRT)"
echo "=========================================="

# IRT models depend on the model pool, so they need to be retrained when pool changes

IRT_CONFIG_FILE="./checkpoints/main/irt_config.txt"
IRT_CURRENT_CONFIG="${LLM_POOL[*]}|${IRT_LATENT_DIM}|${IRT_LR}|${IRT_EPOCHS}"

NEEDS_IRT_TRAINING=true

if [ -f "$IRT_CONFIG_FILE" ]; then
    IRT_STORED_CONFIG=$(cat "$IRT_CONFIG_FILE")
    if [ "$IRT_CURRENT_CONFIG" = "$IRT_STORED_CONFIG" ]; then
        # Check if checkpoint files exist
        if [ -d "./checkpoints/main/irt_mirt" ] && \
           [ -f "./checkpoints/main/irt_mirt/mirt_model.pt" ] && \
           [ -d "./checkpoints/main/irt_nirt" ] && \
           [ -f "./checkpoints/main/irt_nirt/nirt_model.pt" ]; then
            echo "[SKIP] IRT models already trained with current configuration"
            NEEDS_IRT_TRAINING=false
        else
            echo "[TRAIN] IRT checkpoints incomplete, retraining..."
        fi
    else
        echo "[TRAIN] IRT configuration changed (model pool or hyperparameters), retraining..."
    fi
else
    echo "[TRAIN] No previous IRT configuration found, training..."
fi

# Train IRT if needed
if [ "$NEEDS_IRT_TRAINING" = true ]; then
    echo "Training IRT models (MIRT and NIRT)..."

    # Build IRT training command
    IRT_CMD="python -m main.baselines.irt.train_irt \
        --latent-dim $IRT_LATENT_DIM \
        --lr $IRT_LR \
        --batch-size $IRT_BATCH_SIZE \
        --epochs $IRT_EPOCHS"

    # Add custom descriptions if specified
    if [ -n "$IRT_DESCRIPTIONS" ]; then
        echo "Using custom LLM descriptions from: $IRT_DESCRIPTIONS"
        IRT_CMD="$IRT_CMD --llm-descriptions $IRT_DESCRIPTIONS"
    else
        echo "Using auto-generated LLM descriptions"
    fi

    # Add model arguments and execute
    eval "$IRT_CMD ${MODELS_ARG[*]}"

    # Save configuration after successful training
    if [ $? -eq 0 ]; then
        mkdir -p ./checkpoints/main
        echo "$IRT_CURRENT_CONFIG" > "$IRT_CONFIG_FILE"
        echo "Saved IRT configuration to $IRT_CONFIG_FILE"
    fi
fi

# ============================================================================
# Step 4: Compare R2-Router vs Baselines (CARROT and IRT)
# ============================================================================

echo ""
echo "=========================================="
echo "STEP 4: Comparing R2-Router vs Baselines (CARROT and IRT)"
echo "=========================================="
echo "Lambda Distribution: $LAMBDA_DISTRIBUTION"
echo "Target Accuracy Rate: $TARGET_ACCURACY_RATE"
echo "Using model arguments:"
printf '%s\n' "${MODELS_ARG[@]}" | head -20

python -m main.evaluation.compare_methods --lambda-dist "$LAMBDA_DISTRIBUTION" --target-accuracy-rate $TARGET_ACCURACY_RATE "${MODELS_ARG[@]}"

echo ""
echo "=========================================="
echo "DONE!"
echo "=========================================="
echo "Results saved to:"
echo "  - ./comparison_results/main/r2_vs_baselines_metrics.csv"
echo "  - ./comparison_results/main/r2_vs_baselines_curves.csv"
echo "  - ./comparison_results/main/r2_vs_baselines_curves.png"
echo "  - ./plots/carrot_knn/"
echo "  - ./plots/carrot_linear/"
echo "  - ./plots/irt_mirt/"
echo "  - ./plots/irt_nirt/"
echo ""
echo "Configuration used:"
echo "  R2-Router Model Type: $CORE_MODEL_TYPE"
if [ "$CORE_MODEL_TYPE" = "torch_mlp" ]; then
    echo "  R2-Router Hidden Layers: $CORE_HIDDEN_LAYERS"
    echo "  R2-Router Epochs: $CORE_TORCH_EPOCHS"
    echo "  R2-Router Learning Rate: $CORE_TORCH_LR"
    echo "  R2-Router Dropout: $CORE_TORCH_DROPOUT"
    echo "  R2-Router Batch Size: $CORE_TORCH_BATCH_SIZE"
elif [ "$CORE_MODEL_TYPE" = "mlp" ]; then
    echo "  R2-Router Hidden Layers: $CORE_HIDDEN_LAYERS"
    echo "  R2-Router Max Iterations: $CORE_MAX_ITER"
elif [ "$CORE_MODEL_TYPE" = "ridge" ] || [ "$CORE_MODEL_TYPE" = "lasso" ] || [ "$CORE_MODEL_TYPE" = "elasticnet" ]; then
    echo "  R2-Router Alpha: $CORE_ALPHA"
fi
echo "  Training Scheme: $SCHEME_SUFFIX"
echo "  Lambda Distribution: $LAMBDA_DISTRIBUTION"
echo "  Target Accuracy Rate: $TARGET_ACCURACY_RATE (QNC target)"
echo "  IRT Latent Dim: $IRT_LATENT_DIM"
echo "  IRT Epochs: $IRT_EPOCHS"
echo ""
echo "Checkpoints saved to:"
echo "  R2-Router:"
for llm in "${LLM_POOL[@]}"; do
    IFS='|' read -r name size csv <<< "$llm"
    echo "    - ./checkpoints/main/${name}_${SCHEME_SUFFIX}/"
done
echo "  CARROT:"
echo "    - ./checkpoints/main/carrot_knn/"
echo "    - ./checkpoints/main/carrot_linear/"
echo "  IRT:"
echo "    - ./checkpoints/main/irt_mirt/"
echo "    - ./checkpoints/main/irt_nirt/"
echo ""
echo "To customize lambda distribution, edit LAMBDA_DISTRIBUTION in this script."
echo "See LAMBDA_CUSTOMIZATION.md for details."
