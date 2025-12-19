# Checkpoint Management Strategy

## Overview

The experimental pipeline ([run_experiment.sh](run_experiment.sh)) implements intelligent checkpoint management to avoid unnecessary retraining while ensuring correctness.

## Two Different Checkpoint Strategies

### 1. CoRE Models: Per-Model Checkpoints ✓

**Location:** `./checkpoints/{ModelName}_{TrainingScheme}/`

**Files:**
- `limited_score_predictors.joblib` (15 predictors for limited token counts)
- `unlimited_score_predictor.joblib` (1 predictor for unlimited tokens)
- `unlimited_token_predictor.joblib` (token count predictor)

**Loading Logic:**
- Each model is checked independently
- Model is loaded if checkpoint directory exists AND all 3 files exist
- Only missing models are trained

**Why this works:**
- CoRE models are trained independently per LLM
- No dependencies between models
- Changing model pool doesn't invalidate existing models
- Can mix-and-match: load some, train others

**Example Scenario:**
```bash
# Initial run with 3 models
LLM_POOL=(
    "GLM-4.5-Air|0.85|data/GLM-4.5-Air.csv"
    "Llama-3.1-70B|0.40|data/Llama-3.1-70B.csv"
    "Qwen3-0.6B|0.0173|data/Qwen3-0.6B.csv"
)
CORE_MODEL_TYPE="ridge"
CORE_ALPHA=10.0
```
After training, you have:
- `./checkpoints/GLM-4.5-Air_ridge_alpha10.0/`
- `./checkpoints/Llama-3.1-70B_ridge_alpha10.0/`
- `./checkpoints/Qwen3-0.6B_ridge_alpha10.0/`

Now add a new model:
```bash
LLM_POOL=(
    "GLM-4.5-Air|0.85|data/GLM-4.5-Air.csv"
    "Llama-3.1-70B|0.40|data/Llama-3.1-70B.csv"
    "Qwen3-0.6B|0.0173|data/Qwen3-0.6B.csv"
    "gemma-3-4b-it|0.30|data/gemma-3-4b-it.csv"  # NEW!
)
```
Result:
- ✓ Load: GLM-4.5-Air, Llama-3.1-70B, Qwen3-0.6B (already trained)
- ✗ Train: gemma-3-4b-it (new model)

### 2. CARROT Baselines: Pool-Level Checkpoints ⚠️

**Location:** `./checkpoints/carrot_knn/` and `./checkpoints/carrot_linear/`

**Files:**
- `carrot_knn/knn_score.joblib` (KNN for score prediction)
- `carrot_knn/knn_count.joblib` (KNN for token count prediction)
- `carrot_linear/linear_score.joblib` (Linear for score prediction)
- `carrot_linear/linear_count.joblib` (Linear for token count prediction)

**Configuration Tracking:**
- `./checkpoints/carrot_config.txt` stores: `{TrainingScheme}|{ModelPool}`

**Loading Logic:**
1. Check if `carrot_config.txt` exists
2. If yes, compare stored config with current config
3. If configs match AND all checkpoint files exist → Load
4. Otherwise → Retrain

**Why this is necessary:**
- CARROT is trained on the **entire model pool**
- It learns patterns across all LLMs together
- Predictions depend on which LLMs are in the pool
- Changing the pool invalidates the trained model

**Retrain Triggers:**
1. **Model pool changes** (add/remove any LLM)
2. **Training scheme changes** (e.g., `ridge_alpha10.0` → `ridge_alpha100.0`)
   - This changes CoRE predictions
   - CARROT compares against CoRE, so must retrain

**Example Scenario 1: Adding a model**
```bash
# Before
LLM_POOL=("GLM-4.5-Air|..." "Llama-3.1-70B|...")
CARROT config: "ridge_alpha10.0|GLM-4.5-Air|... Llama-3.1-70B|..."

# After: Add gemma
LLM_POOL=("GLM-4.5-Air|..." "Llama-3.1-70B|..." "gemma-3-4b-it|...")
CARROT config: "ridge_alpha10.0|GLM-4.5-Air|... Llama-3.1-70B|... gemma-3-4b-it|..."

Result: Config mismatch → Retrain CARROT
```

**Example Scenario 2: Changing hyperparameters**
```bash
# Before
CORE_ALPHA=10.0
CARROT config: "ridge_alpha10.0|GLM-4.5-Air|..."

# After
CORE_ALPHA=100.0
CARROT config: "ridge_alpha100.0|GLM-4.5-Air|..."  # Different scheme!

Result: Config mismatch → Retrain CARROT
```

## Configuration File Format

`./checkpoints/carrot_config.txt` contains a single line:
```
{SCHEME_SUFFIX}|{LLM_NAME_1} ... {LLM_NAME_N}
```

Example:
```
ridge_alpha10.0|GLM-4.5-Air|0.85|data/GLM-4.5-Air.csv Llama-3.1-70B|0.40|data/Llama-3.1-70B.csv
```

This captures both:
- Training scheme used for CoRE
- Complete model pool (names, sizes, paths)

## Script Behavior Summary

### Step 1: Train CoRE
```bash
for each model in LLM_POOL:
    if checkpoint exists with all 3 files:
        SKIP (load existing)
    else:
        TRAIN
```

### Step 2: Train CARROT
```bash
if carrot_config.txt exists:
    if current_config == stored_config:
        if all checkpoint files exist:
            SKIP (load existing)
        else:
            TRAIN (incomplete checkpoints)
    else:
        TRAIN (config changed)
else:
    TRAIN (first time)

if training succeeded:
    save current_config to carrot_config.txt
```

## Best Practices

### When to delete checkpoints

**CoRE:**
- Delete individual model checkpoints when:
  - Changing hyperparameters for that model
  - Fixing data issues for that model
  - You want to retrain from scratch
- Keep checkpoints when:
  - Adding new models to pool
  - Removing other models from pool

**CARROT:**
- Delete CARROT checkpoints when:
  - The script will do this automatically! Just run with new config
- Manually delete if:
  - You want to force retraining
  - Checkpoints are corrupted
  - `carrot_config.txt` is out of sync

### Directory cleanup

To fully reset:
```bash
rm -rf ./checkpoints/carrot_*
rm -f ./checkpoints/carrot_config.txt
```

To reset specific CoRE model:
```bash
rm -rf ./checkpoints/GLM-4.5-Air_ridge_alpha10.0/
```

To reset all CoRE models for one scheme:
```bash
rm -rf ./checkpoints/*_ridge_alpha10.0/
```

## Why This Design?

**Efficiency:**
- Don't retrain models that haven't changed
- Especially important for large KNN models (200MB each)

**Correctness:**
- CARROT must match the current CoRE configuration
- Pool-level retraining ensures fair comparison

**Flexibility:**
- Experiment with different model pools easily
- Add new models incrementally
- Compare different training schemes side-by-side

**Transparency:**
- Clear messages: `[SKIP]` vs `[TRAIN]`
- Config file makes dependencies explicit
- Easy to debug what triggered retraining
