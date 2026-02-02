# Run Experiment: Complete Pipeline (R2-Router + CARROT)

Simple script to define your LLM pool, train R2-Router with hyperparameters, train CARROT, and compare them.

## Usage

### 1. Edit the configuration (lines 10-31 in run_experiment.sh)

```bash
# LLM Pool (checkpoints are auto-generated based on training scheme)
LLM_POOL=(
    "Name|Size|CSV"
    "GLM-4.5-Air|0.85|data/GLM-4.5-Air.csv"
    "Qwen3-0.6B|0.0173|data/Qwen3-0.6B.csv"
)

# R2-Router Hyperparameters (tune for better precision)
CORE_MODEL_TYPE="ridge"   # Try: "linear", "ridge", "lasso", "random_forest"
CORE_ALPHA=10.0           # Regularization (try: 1.0, 10.0, 100.0)
```

### 2. Run the complete pipeline

```bash
bash run_experiment.sh
```

This will:
1. **Train R2-Router** - Train your predictor models with configured hyperparameters
2. **Train CARROT** - Train KNN and Linear baselines on the same LLM pool
3. **Compare** - Generate quality-cost curves comparing both methods
4. **Save results** to `./comparison_results/`

### 3. View results

- **Plot**: `./comparison_results/core_vs_carrot_curves.png`
- **Data**: `./comparison_results/core_vs_carrot_curves.csv`
- **R2-Router plots**: `./plots/{model}_multi/`
- **CARROT plots**: `./plots/carrot_knn/` and `./plots/carrot_linear/`

## Configuration Options

### R2-Router Model Types

Edit `CORE_MODEL_TYPE` to choose the algorithm:

| Model Type | Description | When to Use |
|------------|-------------|-------------|
| `"linear"` | Linear Regression (baseline) | Fast, interpretable |
| `"ridge"` ⭐ | Ridge (L2 regularization) | **Recommended** - Better generalization |
| `"lasso"` | Lasso (L1 regularization) | Feature selection |
| `"elasticnet"` | L1 + L2 combined | Both regularization and selection |
| `"random_forest"` | Tree-based ensemble | Non-linear patterns |
| `"gradient_boosting"` | Boosted trees | Maximum accuracy (slower) |
| `"mlp"` | Neural network | Complex non-linear patterns |

### R2-Router Hyperparameters

**For Ridge/Lasso/ElasticNet:**
```bash
CORE_MODEL_TYPE="ridge"
CORE_ALPHA=10.0          # Higher = more regularization (try: 1.0, 10.0, 100.0)
CORE_L1_RATIO=0.5        # ElasticNet only: 0=Ridge, 1=Lasso
```

**For Random Forest/Gradient Boosting:**
```bash
CORE_MODEL_TYPE="random_forest"
CORE_N_ESTIMATORS=100    # Number of trees
CORE_MAX_DEPTH=3         # Tree depth (0=unlimited)
```

**For Neural Network:**
```bash
CORE_MODEL_TYPE="mlp"
CORE_HIDDEN_LAYERS="100,50"  # Layer sizes (comma-separated)
```

## Examples

### Example 1: Quick Test with Ridge

```bash
# Edit run_experiment.sh:
LLM_POOL=(
    "GLM-4.5-Air|0.85|data/GLM-4.5-Air.csv"
    "Qwen3-0.6B|0.0173|data/Qwen3-0.6B.csv"
)
CORE_MODEL_TYPE="ridge"
CORE_ALPHA=10.0

# Run:
bash run_experiment.sh
```

### Example 2: Maximum Accuracy with Gradient Boosting

```bash
# Edit run_experiment.sh:
CORE_MODEL_TYPE="gradient_boosting"
CORE_N_ESTIMATORS=200
CORE_MAX_DEPTH=5

# Run:
bash run_experiment.sh
```

### Example 3: Feature Selection with Lasso

```bash
# Edit run_experiment.sh:
CORE_MODEL_TYPE="lasso"
CORE_ALPHA=1.0

# Run:
bash run_experiment.sh
```

## Checkpoint Organization

Checkpoints are automatically organized by training scheme:

```
checkpoints/
├── GLM-4.5-Air_ridge_alpha10.0/             # Ridge with alpha=10.0
├── GLM-4.5-Air_ridge_alpha100.0/            # Ridge with alpha=100.0
├── GLM-4.5-Air_gbm_n100_d3/                 # Gradient Boosting
├── Qwen3-0.6B_ridge_alpha10.0/
└── ...

plots/
├── GLM-4.5-Air_ridge_alpha10.0/             # Plots for each scheme
├── GLM-4.5-Air_gbm_n100_d3/
└── ...
```

This makes it easy to:
- Compare different hyperparameter settings
- Keep multiple trained models
- Switch between configurations without overwriting

### Scheme Naming Convention

| Model Type | Checkpoint Name Example |
|------------|------------------------|
| Linear | `{model}_linear` |
| Ridge | `{model}_ridge_alpha10.0` |
| Lasso | `{model}_lasso_alpha1.0` |
| ElasticNet | `{model}_elasticnet_alpha1.0_l10.5` |
| Random Forest | `{model}_rf_n100_d3` |
| Gradient Boosting | `{model}_gbm_n100_d3` |
| MLP | `{model}_mlp_100_50` |

## Output

The script prints:

```
==========================================
STEP 1: Training R2-Router on 10 models
==========================================
Model Type: ridge
Alpha: 10.0
==========================================

Training model: GLM-4.5-Air
...
[Limited Score Predictors] MSE=0.0234, MAE=0.1123, R²=0.6789
[Unlimited Quality Predictor] MSE=0.0198, MAE=0.1045, R²=0.7234
[OK] GLM-4.5-Air: Prediction done.

==========================================
STEP 2: Training CARROT on 10 models
==========================================
...

==========================================
STEP 3: Comparing R2-Router vs CARROT
==========================================
...

==========================================
RESULTS
==========================================

R2-Router (Our Method):
  AUDC: 0.8456
  Peak Accuracy: 0.8723

CARROT-KNN:
  AUDC: 0.8315
  Peak Accuracy: 0.8767

R2-Router vs Best CARROT:
  AUDC improvement: +1.70%
  Peak accuracy improvement: -0.52%

==========================================
DONE!
==========================================
Results saved to:
  - ./comparison_results/core_vs_carrot_curves.csv
  - ./comparison_results/core_vs_carrot_curves.png
  - ./plots/carrot_knn/
  - ./plots/carrot_linear/

R2-Router configuration used:
  Model Type: ridge
  Alpha: 10.0
  Training Scheme: ridge_alpha10.0

R2-Router checkpoints saved to:
  - ./checkpoints/GLM-4.5-Air_ridge_alpha10.0/
  - ./checkpoints/GLM-4.6_ridge_alpha10.0/
  - ./checkpoints/gemma-3-4b-it_ridge_alpha10.0/
  ...
```

## Checkpoint Management

The script intelligently manages checkpoints to avoid unnecessary retraining:

### R2-Router Models (Per-Model Checkpoints)
Each R2-Router model is trained independently and can be loaded if:
- Checkpoint directory exists: `./checkpoints/{ModelName}_{TrainingScheme}/`
- All 3 required files exist:
  - `limited_score_predictors.joblib`
  - `unlimited_score_predictor.joblib`
  - `unlimited_token_predictor.joblib`

**Example:** If you change the model pool (add/remove LLMs) but keep the same training scheme, R2-Router models will be loaded for existing LLMs and only new LLMs will be trained.

### CARROT Baselines (Pool-Level Checkpoints)
CARROT models are trained on the **entire model pool**, so they must be retrained when:
- The model pool changes (any LLM added/removed)
- The training scheme changes (affects R2-Router predictions)

The script stores a configuration file (`./checkpoints/carrot_config.txt`) that tracks:
- Current training scheme (e.g., `ridge_alpha10.0`)
- Current model pool (list of all LLMs)

**Example:** If you change `CORE_ALPHA` from 10.0 to 100.0, all R2-Router models will be retrained in new folders (`{model}_ridge_alpha100.0`), and CARROT will automatically retrain to compare against the new R2-Router predictions.

## Understanding Metrics

**During Training (per model):**
- **MSE** (Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better
- **R²** (R-squared): Higher is better (0-1 scale)
  - R² > 0.8: Excellent
  - R² > 0.6: Good
  - R² < 0.5: Poor (try different model type)

**Final Comparison:**
- **AUDC** (Area Under Deferral Curve): Higher is better
- **Peak Accuracy**: Maximum quality achieved

## Tuning Strategy

### Step 1: Start with Ridge
```bash
CORE_MODEL_TYPE="ridge"
CORE_ALPHA=10.0
```
Run and check R² scores. If R² > 0.7, this is good enough.

### Step 2: If Ridge isn't enough
Try stronger regularization:
```bash
CORE_ALPHA=100.0  # More regularization
```
Or weaker:
```bash
CORE_ALPHA=1.0    # Less regularization
```

### Step 3: If linear models don't work
Try non-linear:
```bash
CORE_MODEL_TYPE="gradient_boosting"
CORE_N_ESTIMATORS=100
CORE_MAX_DEPTH=3
```

## Comparison with Original

**Before** (manual process):
1. Manually edit `predictor_sklearn.py` hyperparameters
2. Run `python predictor_sklearn.py` to train R2-Router
3. Run `python baselines_carrot.py` to train CARROT
4. Run `python compare_core_carrot.py` to compare
5. Check if results are good, repeat from step 1 if not

**Now** (automated):
1. Edit hyperparameters in `run_experiment.sh` (one place!)
2. Run `bash run_experiment.sh`
3. Done! Get complete comparison with configured hyperparameters

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "CSV not found" | Check paths in `LLM_POOL` |
| Low R² scores | Try `CORE_MODEL_TYPE="ridge"` with `CORE_ALPHA=10.0` |
| Training too slow | Use `CORE_MODEL_TYPE="ridge"` (fastest) |
| R2-Router worse than CARROT | Try `CORE_MODEL_TYPE="gradient_boosting"` |

## Files

- **[run_experiment.sh](run_experiment.sh)** - Main script (configure here)
- **[train_r2.py](train_r2.py)** - R2-Router training (don't edit)
- **[train_carrot.py](train_carrot.py)** - CARROT training (don't edit)
- **[compare_methods.py](compare_methods.py)** - Comparison (don't edit)

Only edit `run_experiment.sh` - it controls everything!

## Advanced: Manual Training

If you want to train models separately:

```bash
# Train only R2-Router
python train_r2.py --model_type ridge --alpha 10.0 \
    --model GLM-4.5-Air 0.85 data/GLM-4.5-Air.csv ./checkpoints/GLM-4.5-Air_multi

# Train only CARROT
python train_carrot.py --model GLM-4.5-Air 0.85 data/GLM-4.5-Air.csv ./checkpoints/GLM-4.5-Air_multi

# Compare only
python compare_methods.py --model GLM-4.5-Air 0.85 data/GLM-4.5-Air.csv ./checkpoints/GLM-4.5-Air_multi
```

But it's easier to just use `run_experiment.sh`!
