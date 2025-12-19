# Baseline Split Refactoring Summary

## Overview

Successfully split `baselines.py` into two separate files for better code organization:

1. **baselines_carrot.py** - CARROT baselines (KNN and Linear Regression)
2. **baselines_irt.py** - IRT baselines (MIRT and NIRT)

Both files now include checkpoint save/load functionality similar to `predictor_sklearn.py`.

## New File Structure

### baselines_carrot.py

Contains CARROT (Cost-Aware Routing with Regression On Tokens) baselines:

**Classes:**
- `CarrotKNNBaseline` - K-Nearest Neighbors baseline
  - Predicts quality scores using KNN on query embeddings
  - Predicts token counts using KNN on query embeddings
  - Checkpoint files: `knn_score.joblib`, `knn_count.joblib`

- `CarrotLinearBaseline` - Linear Regression baseline (vLLM-SR)
  - Predicts quality scores using Linear Regression
  - Predicts token counts using Linear Regression
  - Checkpoint files: `linear_score.joblib`, `linear_count.joblib`

**Functions:**
- `route_baseline()` - Generic routing function that selects best (model, token_limit) based on predicted scores and costs

**Backward Compatibility:**
```python
CarrotBaseline = CarrotKNNBaseline  # Old alias
LinearCarrotBaseline = CarrotLinearBaseline  # Already named correctly
```

### baselines_irt.py

Contains IRT (Item Response Theory) baselines:

**Classes:**
- `IRTBaseline` (MIRT-Router) - Multidimensional IRT
  - Uses linear IRT model with query discrimination and difficulty
  - PyTorch-based training with BCEWithLogitsLoss
  - Checkpoint files: `mirt_model.pt`, `mirt_config.pt`, `mirt_llm_embeddings.pt`

- `NIRTBaseline` (NIRT-Router) - Neural IRT
  - Neural IRT model with relevance vectors (NCDM-based)
  - Uses PCA to generate relevance vectors for interpretable abilities
  - PyTorch-based training with BCEWithLogitsLoss
  - Checkpoint files: `nirt_model.pt`, `nirt_config.pt`, `nirt_llm_embeddings.pt`, `nirt_pca.pkl`

**Neural Modules:**
- `_MIRTModule` - PyTorch module for MIRT
- `_NIRTModule` - PyTorch module for NIRT

**Backward Compatibility:**
```python
MIRTBaseline = IRTBaseline  # Old alias
```

## Key Features Added

### Checkpoint Save/Load Functionality

All baseline classes now support checkpoint saving and loading:

**CARROT baselines (sklearn models):**
```python
# Train and save
carrot_knn = CarrotKNNBaseline(llms=llms)
carrot_knn.fit(save_dir="./checkpoints/carrot_knn")

# Load from checkpoint
carrot_knn_loaded = CarrotKNNBaseline(llms=llms)
carrot_knn_loaded.load(load_dir="./checkpoints/carrot_knn")

# Or load during initialization
carrot_knn = CarrotKNNBaseline(llms=llms, load_dir="./checkpoints/carrot_knn")
```

**IRT baselines (PyTorch models):**
```python
# Train and save
irt = IRTBaseline(llms=llms, llm_texts=llm_texts)
irt.fit(epochs=200, save_dir="./checkpoints/irt_mirt")

# Load from checkpoint
irt_loaded = IRTBaseline(
    llms=llms,
    llm_texts=llm_texts,
    load_dir="./checkpoints/irt_mirt"
)

# Or load during initialization
irt = IRTBaseline(llms=llms, llm_texts=llm_texts, load_dir="./checkpoints/irt_mirt")
```

### Consistent API Design

All baseline classes follow a consistent pattern:

**Initialization:**
```python
def __init__(self, llms, ..., load_dir=None):
    # Initialize with LLM data
    # Optionally load from checkpoint if load_dir provided
```

**Training:**
```python
def fit(self, ..., save_dir=None, plot_dir=None):
    # Train the model
    # Optionally save to checkpoint if save_dir provided
    # Optionally save plots if plot_dir provided
    return self  # For method chaining
```

**Prediction:**
```python
def predict(self):
    # Predict on test set
    # Returns predictions (and optionally token counts)
```

**Loading:**
```python
def load(self, load_dir):
    # Load pre-trained model from checkpoint
    return self  # For method chaining
```

## Updated Files

### Import Updates

**results.py:**
```python
# Before:
from baselines import CarrotBaseline, LinearCarrotBaseline, route_baseline, IRTBaseline, NIRTBaseline

# After:
from baselines_carrot import CarrotKNNBaseline as CarrotBaseline, CarrotLinearBaseline as LinearCarrotBaseline, route_baseline
from baselines_irt import IRTBaseline, NIRTBaseline
```

**ood_evaluation/run_ood.py:**
```python
# Before:
from baselines import CarrotBaseline, LinearCarrotBaseline, IRTBaseline, NIRTBaseline, route_baseline

# After:
from baselines_carrot import CarrotKNNBaseline as CarrotBaseline, CarrotLinearBaseline as LinearCarrotBaseline, route_baseline
from baselines_irt import IRTBaseline, NIRTBaseline
```

**ood_evaluation/archive/eval_mmlu_pro.py:**
```python
# Before:
from baselines import (
    CarrotBaseline,
    LinearCarrotBaseline,
    IRTBaseline,
    NIRTBaseline,
    route_baseline
)

# After:
from baselines_carrot import (
    CarrotKNNBaseline as CarrotBaseline,
    CarrotLinearBaseline as LinearCarrotBaseline,
    route_baseline
)
from baselines_irt import IRTBaseline, NIRTBaseline
```

## Usage Examples

### CARROT-KNN Example

```python
from baselines_carrot import CarrotKNNBaseline

# Initialize and train
carrot_knn = CarrotKNNBaseline(
    llms=llms,
    n_neighbors_score=256,
    n_neighbors_count=256,
    metric="cosine"
)

carrot_knn.fit(save_dir="./checkpoints/carrot_knn")

# Predict
Y_hat_score, Y_hat_count = carrot_knn.predict()

# Use for routing
from baselines_carrot import route_baseline

lamb_range = np.linspace(0, 1, 21)
sizes_vec = np.array([0.85, 0.06])  # Model sizes

router_cost, router_perf = route_baseline(
    Y_hat_score=Y_hat_score,
    Y_hat_count=Y_hat_count,
    Y_score_true=Y_score_true,
    Y_count_true=Y_count_true,
    lamb_range=lamb_range,
    sizes_vec=sizes_vec
)
```

### CARROT-Linear Example

```python
from baselines_carrot import CarrotLinearBaseline

# Initialize and train
carrot_linear = CarrotLinearBaseline(
    llms=llms,
    fit_intercept=True
)

carrot_linear.fit(save_dir="./checkpoints/carrot_linear")

# Predict
Y_hat_score, Y_hat_count = carrot_linear.predict()
```

### MIRT Example

```python
from baselines_irt import IRTBaseline

# Define LLM descriptions
llm_texts = {
    "GLM-4.5-Air": "GLM-4.5-Air is a lightweight language model",
    "Llama-3.2-3B": "Llama-3.2-3B is a small instruction-tuned model"
}

# Initialize and train
irt = IRTBaseline(
    llms=llms,
    llm_texts=llm_texts,
    latent_dim=32,
    device="cuda"
)

irt.fit(
    lr=3e-3,
    batch_size=128,
    epochs=200,
    save_dir="./checkpoints/irt_mirt",
    plot_dir="./plots/irt_mirt"
)

# Predict (only quality scores, no token counts)
Y_hat_score = irt.predict()

# For routing, use unlimited token counts from LLM data
from baselines_carrot import route_baseline

Y_hat_count = np.stack([llms[name]["pred_test_unlimited_count"] for name in llm_names], axis=1)
Y_count_true = np.stack([llms[name]["true_test_unlimited_count"] for name in llm_names], axis=1)

router_cost, router_perf = route_baseline(
    Y_hat_score=Y_hat_score,
    Y_hat_count=Y_hat_count,
    Y_score_true=Y_score_true,
    Y_count_true=Y_count_true,
    lamb_range=lamb_range,
    sizes_vec=sizes_vec
)
```

### NIRT Example

```python
from baselines_irt import NIRTBaseline

# Initialize and train
nirt = NIRTBaseline(
    llms=llms,
    llm_texts=llm_texts,
    latent_dim=32,
    device="cuda"
)

nirt.fit(
    lr=3e-3,
    batch_size=128,
    epochs=200,
    save_dir="./checkpoints/irt_nirt",
    plot_dir="./plots/irt_nirt"
)

# Predict (only quality scores, no token counts)
Y_hat_score = nirt.predict()
```

## Checkpoint Directory Structure

```
checkpoints/
├── carrot_knn/
│   ├── knn_score.joblib          # KNN score predictor
│   └── knn_count.joblib          # KNN token count predictor
│
├── carrot_linear/
│   ├── linear_score.joblib       # Linear score predictor
│   └── linear_count.joblib       # Linear token count predictor
│
├── irt_mirt/
│   ├── mirt_model.pt             # PyTorch model state dict
│   ├── mirt_config.pt            # Model configuration
│   └── mirt_llm_embeddings.pt    # LLM embeddings
│
└── irt_nirt/
    ├── nirt_model.pt             # PyTorch model state dict
    ├── nirt_config.pt            # Model configuration
    ├── nirt_llm_embeddings.pt    # LLM embeddings
    └── nirt_pca.pkl              # PCA for relevance vectors
```

## Benefits of Refactoring

1. **Better Code Organization**: Related baselines grouped together
2. **Checkpoint Support**: All baselines can now save/load trained models
3. **Consistent API**: All classes follow the same fit/predict/load pattern
4. **Backward Compatibility**: Old code continues to work via aliases
5. **Easier Maintenance**: Separate files for different baseline families
6. **Reproducibility**: Can save and reload baseline models for experiments

## Testing

Run the comprehensive test script:

```bash
python test_baselines_split.py
```

This will test:
- ✅ CARROT-KNN training and checkpoint save/load
- ✅ CARROT-Linear training and checkpoint save/load
- ✅ MIRT training and checkpoint save/load
- ✅ NIRT training and checkpoint save/load
- ✅ Backward compatibility via aliases
- ✅ Prediction reproducibility from checkpoints

## Migration Guide

### For Existing Code

No changes needed! The import updates have already been made to all existing files, and backward compatibility is maintained via aliases.

### For New Code

Use the new class names directly:

```python
# CARROT baselines
from baselines_carrot import CarrotKNNBaseline, CarrotLinearBaseline, route_baseline

# IRT baselines
from baselines_irt import IRTBaseline, NIRTBaseline
```

### For Training Baselines

Add checkpoint saving to preserve trained models:

```python
# Old way (no checkpoints):
baseline.fit()

# New way (with checkpoints):
baseline.fit(save_dir="./checkpoints/baseline_name", plot_dir="./plots/baseline_name")
```

### For Loading Baselines

Load pre-trained models instead of retraining:

```python
# Option 1: Load during initialization
baseline = CarrotKNNBaseline(llms=llms, load_dir="./checkpoints/carrot_knn")

# Option 2: Load after initialization
baseline = CarrotKNNBaseline(llms=llms)
baseline.load(load_dir="./checkpoints/carrot_knn")
```

## Next Steps (Optional)

1. Train and save checkpoints for all baseline models on the full dataset
2. Update training scripts to use checkpoint saving by default
3. Consider adding progress bars to CARROT training for consistency with IRT training
4. Consider deprecating the old `baselines.py` file once all code is migrated

## Files Created/Modified

### Created:
1. ✅ `baselines_carrot.py` - CARROT baseline implementations with checkpoints
2. ✅ `baselines_irt.py` - IRT baseline implementations with checkpoints
3. ✅ `test_baselines_split.py` - Comprehensive test script
4. ✅ `BASELINE_SPLIT_SUMMARY.md` - This documentation

### Modified:
1. ✅ `results.py` - Updated imports
2. ✅ `ood_evaluation/run_ood.py` - Updated imports
3. ✅ `ood_evaluation/archive/eval_mmlu_pro.py` - Updated imports

### Unchanged (to keep):
1. 📦 `baselines.py` - Can be kept for backward compatibility or deprecated later

## Summary

The baseline splitting refactoring successfully:
- ✅ Organized code into logical, focused files
- ✅ Added checkpoint save/load functionality to all baselines
- ✅ Maintained full backward compatibility
- ✅ Updated all import statements across the codebase
- ✅ Created comprehensive tests to verify correctness
- ✅ Followed the same design pattern as `predictor_sklearn.py`
- ✅ Documented all changes and usage patterns

All baselines can now be trained once, saved to checkpoints, and reloaded for evaluation without retraining!
