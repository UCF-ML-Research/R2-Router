# Predictor Selection Guide

This guide explains how to switch between different predictor implementations in the LLM routing codebase.

## Available Predictors

### 1. PyTorch Neural Network Predictor (`predictor.py`)
- **Architecture**: Separate MLP for each token limit
- **Layers**: [256, 128, 64] with ReLU and Dropout
- **Loss**: Binary Cross Entropy with Logits
- **Training**: ~100 epochs with AdamW optimizer
- **Checkpoint format**: `.pt` files
- **Best for**: Maximum prediction accuracy

### 2. Sklearn Linear Regression Predictor (`predictor_sklearn.py`)
- **Architecture**: Linear regression per token limit
- **Training**: Fast, no hyperparameter tuning needed
- **Checkpoint format**: `.joblib` files
- **Best for**: Quick experiments, baseline comparisons

## How to Switch Predictors in results.py

### Step 1: Edit the Configuration Section

Open `results.py` and find the **PREDICTOR CONFIGURATION** section near the top:

```python
# ============================================================================
# PREDICTOR CONFIGURATION
# ============================================================================
# Choose which predictor to use by uncommenting ONE of the following:

# Option 1: PyTorch-based neural network predictor (default)
from predictor import TokenPerformancePredictor as PredictorClass

# Option 2: Sklearn-based linear regression predictor
# from predictor_sklearn import TokenPerformancePredictor as PredictorClass

# ============================================================================
```

### Step 2: Comment/Uncomment the Desired Predictor

**To use PyTorch predictor:**
```python
from predictor import TokenPerformancePredictor as PredictorClass
# from predictor_sklearn import TokenPerformancePredictor as PredictorClass
```

**To use Sklearn predictor:**
```python
# from predictor import TokenPerformancePredictor as PredictorClass
from predictor_sklearn import TokenPerformancePredictor as PredictorClass
```

### Step 3: Update Checkpoint Paths (if needed)

Make sure your `load_llm()` calls point to the correct checkpoint directories:
- PyTorch checkpoints: `./checkpoints/{model-name}_1e4/`
- Sklearn checkpoints: `./checkpoints/{model-name}_multi/`

Example:
```python
"GLM_4_5_Air": load_llm(
    name="GLM-4.5-Air",
    size=0.85,
    score_df_path="data/GLM-4.5-Air.csv",
    load_dir="./checkpoints/GLM-4.5-Air_1e4",  # Change to _multi for sklearn
    embeddings=embeddings,
    train_idx=train_idx,
    test_idx=test_idx,
    token_limits_score=token_limits_score,
    token_limits_count=token_limits_count,
    predictor_class=PredictorClass  # This line uses the configured predictor
),
```

## Training Checkpoints for Each Predictor

### Train PyTorch Predictors
```bash
python predictor.py
# Saves to: ./checkpoints/{model-name}_1e4/
```

### Train Sklearn Predictors
```bash
python predictor_sklearn.py
# Saves to: ./checkpoints/{model-name}_multi/
# Trains all models configured in the models_to_train list
```

## Implementation Details

The flexibility is achieved through:

1. **llm_loader.py** accepts an optional `predictor_class` parameter
2. It automatically detects which predictor type and instantiates it correctly:
   - PyTorch predictor needs: `hidden_dims`, `dropout`, `load_dir`, `dataset`
   - Sklearn predictor needs: `dataset`, `load_dir`
3. Both predictors have the same `.predict()` interface
4. No other code changes needed!

## Example: Complete Workflow

```bash
# 1. Train sklearn predictors for all models
python predictor_sklearn.py

# 2. Edit results.py to use sklearn predictor
#    Uncomment: from predictor_sklearn import TokenPerformancePredictor as PredictorClass

# 3. Update checkpoint paths in results.py from _1e4 to _multi

# 4. Run evaluation
python results.py
```

## Mixing Predictors

You can even mix predictors for different LLMs by passing different `predictor_class` values:

```python
from predictor import TokenPerformancePredictor as PyTorchPredictor
from predictor_sklearn import TokenPerformancePredictor as SklearnPredictor

llms = {
    "GLM_4_5_Air": load_llm(
        name="GLM-4.5-Air",
        load_dir="./checkpoints/GLM-4.5-Air_1e4",
        predictor_class=PyTorchPredictor,  # Use PyTorch
        ...
    ),
    "gemma3_4B": load_llm(
        name="gemma3-4B",
        load_dir="./checkpoints/gemma-3-4b-it_multi",
        predictor_class=SklearnPredictor,  # Use Sklearn
        ...
    ),
}
```

## Troubleshooting

**Q: Getting "Missing checkpoint" warnings?**
- Make sure you've trained the models with the correct predictor type
- Check that `load_dir` points to the right checkpoint directory

**Q: TypeError when loading predictor?**
- The code automatically handles both predictor types
- If you see errors, check that both predictor files have compatible `__init__` signatures

**Q: Different results with sklearn vs PyTorch?**
- This is expected! Linear regression is simpler than neural networks
- PyTorch predictor should generally perform better
- Use sklearn for quick experiments or as a baseline
