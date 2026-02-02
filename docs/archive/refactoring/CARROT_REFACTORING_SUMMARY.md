# CARROT Baselines Refactoring Summary

## Overview

Refactored `baselines_carrot.py` to use a unified base class architecture and added comprehensive per-LLM visualization capabilities, similar to `predictor_sklearn.py`.

## Key Changes

### 1. Unified Base Class Architecture

**Before:** Two separate classes with duplicated code
- `CarrotKNNBaseline` (150+ lines)
- `CarrotLinearBaseline` (150+ lines)

**After:** Single base class with thin child classes
- `CarrotBaseline` - Base class handling all functionality (~150 lines)
- `CarrotKNNBaseline` - Thin wrapper (~10 lines)
- `CarrotLinearBaseline` - Thin wrapper (~10 lines)

**Benefits:**
- ~40% reduction in code size
- DRY (Don't Repeat Yourself) principle
- Easier to maintain and extend
- Single source of truth for functionality

### 2. Per-LLM Visualizations

Added comprehensive visualization capabilities showing prediction accuracy for each individual LLM:

#### Quality Prediction Visualizations

**Confusion Matrix (per LLM):**
- Filename: `confmat_quality_{llm_name}.png`
- 10x10 buckets for quality scores (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
- Shows diagonal pattern indicating prediction accuracy
- Includes MSE, MAE, R² metrics in title
- Higher density on diagonal = better predictions

**Example output:**
```
confmat_quality_GLM-4.5-Air.png
confmat_quality_Llama-3.1-70B-Instruct.png
confmat_quality_Qwen3-235B-A22B-Instruct-2507.png
...
```

#### Token Count Prediction Visualizations

**Distribution + Scatter Plot (per LLM):**
- Filename: `token_count_{llm_name}.png`
- Left panel: KDE distribution comparing True vs Predicted
- Right panel: Scatter plot with perfect prediction line (red diagonal)
- Includes MSE, MAE, R² metrics in title
- Points closer to diagonal = better predictions

**Example output:**
```
token_count_GLM-4.5-Air.png
token_count_Llama-3.1-70B-Instruct.png
token_count_Qwen3-235B-A22B-Instruct-2507.png
...
```

### 3. Enhanced Training Output

**New output includes:**
```
🚀 Training CARROT-KNN...
[Quality Predictor - Overall] MSE=0.1234, MAE=0.2345, R²=0.3456
[Token Count Predictor - Overall] MSE=1234.5, MAE=567.8, R²=0.4567
📊 Generating plots for 10 LLMs...
📊 Plots saved to ./plots/carrot_knn/
   - 10 quality confusion matrices: confmat_quality_*.png
   - 10 token distribution plots: token_count_*.png
💾 Saved CARROT-KNN models to ./checkpoints/carrot_knn
```

## API Changes

### Updated `fit()` Method

**New parameter:**
```python
def fit(self,
        embedding_train: np.ndarray,
        quality_train: np.ndarray,
        token_count_train: np.ndarray,
        embedding_test: Optional[np.ndarray] = None,
        quality_test: Optional[np.ndarray] = None,
        token_count_test: Optional[np.ndarray] = None,
        llm_names: Optional[list] = None,  # NEW: List of LLM names for plots
        save_dir: Optional[str] = None,
        plot_dir: Optional[str] = None) -> 'CarrotBaseline':
```

**Usage:**
```python
carrot_knn = CarrotKNNBaseline()
carrot_knn.fit(
    embedding_train=X_train,
    quality_train=quality_train,
    token_count_train=token_count_train,
    embedding_test=X_test,
    quality_test=quality_test,
    token_count_test=token_count_test,
    llm_names=["GLM-4.5-Air", "Llama-3.1-70B", "Qwen3-235B"],  # Per-LLM plots
    save_dir="./checkpoints/carrot_knn",
    plot_dir="./plots/carrot_knn"
)
```

## Plotting Functions

### `create_confusion_matrix_plot()`

```python
def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray,
                                 llm_name: str, plot_dir: str):
    """
    Create confusion matrix for quality predictions of one LLM.

    Args:
        y_true: True quality scores for one LLM, shape (n_queries,)
        y_pred: Predicted quality scores for one LLM, shape (n_queries,)
        llm_name: Name of the LLM
        plot_dir: Directory to save plot
    """
```

**Features:**
- 10 quality buckets (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
- Heatmap with annotations showing counts
- MSE, MAE, R² in title
- 8x6 figure size, 150 DPI

### `create_token_distribution_plot()`

```python
def create_token_distribution_plot(y_true: np.ndarray, y_pred: np.ndarray,
                                   llm_name: str, plot_dir: str):
    """
    Create token count distribution comparison for one LLM.

    Args:
        y_true: True token counts for one LLM, shape (n_queries,)
        y_pred: Predicted token counts for one LLM, shape (n_queries,)
        llm_name: Name of the LLM
        plot_dir: Directory to save plot
    """
```

**Features:**
- Left: KDE distribution (True vs Predicted)
- Right: Scatter plot with perfect prediction line
- MSE, MAE, R² in title
- 12x5 figure size, 150 DPI

## Comparison with predictor_sklearn.py

### Similarities (Consistency)
✅ Per-model/per-LLM visualizations
✅ Confusion matrix for quality predictions
✅ Distribution comparison plots
✅ MSE, MAE, R² metrics reported
✅ Save to `plot_dir` with descriptive filenames
✅ Same sklearn-like API (data passed to `fit()`)

### Differences (CARROT-specific)
- **CARROT predicts 2 things per LLM:** quality + token count
- **predictor_sklearn predicts 16 things per LLM:** quality for 15 limited budgets + 1 unlimited
- **CARROT plots:** 2 plots per LLM (quality confusion + token distribution)
- **predictor_sklearn plots:** 15 plots per LLM (confusion matrices for limited budgets)

## File Structure After Training

```
./checkpoints/
├── carrot_knn/
│   ├── knn_score.joblib      # KNN quality predictor
│   └── knn_count.joblib      # KNN token count predictor
└── carrot_linear/
    ├── linear_score.joblib   # Linear quality predictor
    └── linear_count.joblib   # Linear token count predictor

./plots/
├── carrot_knn/
│   ├── confmat_quality_GLM-4.5-Air.png
│   ├── confmat_quality_Llama-3.1-70B-Instruct.png
│   ├── confmat_quality_Qwen3-235B-A22B-Instruct-2507.png
│   ├── token_count_GLM-4.5-Air.png
│   ├── token_count_Llama-3.1-70B-Instruct.png
│   └── token_count_Qwen3-235B-A22B-Instruct-2507.png
└── carrot_linear/
    ├── confmat_quality_GLM-4.5-Air.png
    ├── confmat_quality_Llama-3.1-70B-Instruct.png
    ├── confmat_quality_Qwen3-235B-A22B-Instruct-2507.png
    ├── token_count_GLM-4.5-Air.png
    ├── token_count_Llama-3.1-70B-Instruct.png
    └── token_count_Qwen3-235B-A22B-Instruct-2507.png
```

## Benefits

### 1. Code Quality
- **Cleaner:** Single base class eliminates duplication
- **Maintainable:** Changes only need to be made once
- **Extensible:** Easy to add new regressor types

### 2. Visualization Quality
- **Per-LLM insights:** See prediction accuracy for each model separately
- **Comprehensive:** Both quality and token count predictions visualized
- **Interpretable:** Confusion matrices and distributions easy to understand

### 3. Debugging & Analysis
- **Identify weak spots:** See which LLMs are harder to predict
- **Compare baselines:** Visual comparison between KNN and Linear
- **Validate predictions:** Scatter plots show over/under-prediction patterns

## Usage Example

### Training with Visualizations

```python
from baselines_carrot import CarrotKNNBaseline

# Load your data
embedding_train = ...  # shape (n_train, embedding_dim)
quality_train = ...    # shape (n_train, n_models)
token_count_train = ...  # shape (n_train, n_models)

embedding_test = ...
quality_test = ...
token_count_test = ...

llm_names = ["GLM-4.5-Air", "Llama-3.1-70B", "Qwen3-235B"]

# Train with per-LLM plots
carrot = CarrotKNNBaseline()
carrot.fit(
    embedding_train=embedding_train,
    quality_train=quality_train,
    token_count_train=token_count_train,
    embedding_test=embedding_test,
    quality_test=quality_test,
    token_count_test=token_count_test,
    llm_names=llm_names,
    save_dir="./checkpoints/carrot_knn",
    plot_dir="./plots/carrot_knn"
)

# Generates:
# - 3 quality confusion matrices
# - 3 token distribution plots
# - Overall metrics reported
```

### Loading and Using

```python
# Load pre-trained model
carrot = CarrotKNNBaseline(load_dir="./checkpoints/carrot_knn")

# Predict on new data
quality_pred, token_pred = carrot.predict(new_embeddings)
```

## Summary

✅ **Unified architecture:** Single base class with thin child classes
✅ **Per-LLM visualizations:** Separate plots for each LLM's predictions
✅ **Quality metrics:** Confusion matrices for quality predictions
✅ **Token metrics:** Distribution + scatter plots for token counts
✅ **Consistent API:** Same pattern as predictor_sklearn.py
✅ **Easy to use:** Just pass `llm_names` to `fit()`
✅ **Production ready:** Checkpoints + comprehensive evaluation

The refactored CARROT baselines now provide the same level of insight and visualization quality as the main predictor, making it easy to compare baseline performance with the R2-Router across all LLMs!
