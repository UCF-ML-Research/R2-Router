# Sklearn Predictor Refactoring Summary

## Overview

Successfully refactored `predictor_sklearn.py` to follow sklearn's design pattern where **data is passed to methods** rather than stored as class attributes. This improves separation of concerns and makes the code more maintainable.

## Key Changes

### Before (Old API)
```python
class TokenPerformancePredictor:
    def __init__(self, dataset: RouterDataset, load_dir: str = None):
        # Dataset stored as class attribute
        self.train = dataset.get_train_set_score()
        self.test = dataset.get_test_set_score()
        self.tokens = dataset.get_target_tokens_score()
        # ...

    def fit(self, save_dir="./checkpoints", plot_dir="./plots"):
        # Uses self.train, self.test from dataset
        Xtr = self.train["X"]
        # ...
```

**Problems:**
- Dataset tightly coupled to the model
- Model stores large arrays (train/test data) as attributes
- Not following sklearn conventions
- Hard to use the model independently of RouterDataset

### After (New API)
```python
class TokenPerformancePredictor:
    def __init__(self, token_limits: Optional[List[str]] = None,
                 load_dir: Optional[str] = None):
        # Only stores model config, not data
        self.token_limits = token_limits
        self.score_predictors: Dict[str, LinearRegression] = {}
        self.token_predictor = LinearRegression()

        if load_dir:
            self._load_models(load_dir)

    def fit(self, X_train: np.ndarray, y_train: Dict[str, np.ndarray],
            token_count_train: np.ndarray,
            X_test: Optional[np.ndarray] = None,
            y_test: Optional[Dict[str, np.ndarray]] = None,
            token_count_test: Optional[np.ndarray] = None,
            save_dir: Optional[str] = None,
            plot_dir: Optional[str] = None) -> 'TokenPerformancePredictor':
        # Data passed as parameters (sklearn pattern)
        # ...
```

**Benefits:**
- ✅ Follows sklearn conventions: `fit(X, y)`, `predict(X)`
- ✅ Model is independent of dataset representation
- ✅ No large data stored as class attributes
- ✅ Cleaner separation of concerns
- ✅ Easier to test and reason about

## Usage Examples

### New API (Recommended)
```python
from predictor_sklearn import TokenPerformancePredictor

# Option 1: Train from scratch
token_limits = ['10_score', '20_score', ..., 'unlimited_score']
predictor = TokenPerformancePredictor(token_limits=token_limits)

predictor.fit(
    X_train=train_embeddings,
    y_train=train_scores_dict,  # {'10_score': array, '20_score': array, ...}
    token_count_train=train_counts,
    X_test=test_embeddings,
    y_test=test_scores_dict,
    token_count_test=test_counts,
    save_dir="./checkpoints/model",
    plot_dir="./plots/model"
)

# Option 2: Load from checkpoint
predictor = TokenPerformancePredictor(
    token_limits=token_limits,
    load_dir="./checkpoints/model"
)

# Predict
scores, counts = predictor.predict(new_embeddings)
```

### Backward Compatibility
```python
from predictor_sklearn import train_predictor_from_dataset
from router_dataset import RouterDataset

# Old workflow still works via convenience function
dataset = RouterDataset(
    embeddings=embeddings,
    score_df_path="data/GLM-4.5-Air.csv",
    target_tokens_score=token_limits,
    train_idx=train_idx,
    test_idx=test_idx
)

predictor = train_predictor_from_dataset(
    dataset=dataset,
    save_dir="./checkpoints/model",
    plot_dir="./plots/model"
)
```

## Compatibility

### llm_loader.py
Updated to support both old and new API through graceful fallback:

```python
# Tries new API first (token_limits + load_dir)
try:
    predictor = predictor_class(
        token_limits=token_limits_score,
        load_dir=load_dir
    )
except TypeError:
    # Falls back to old API (dataset + load_dir)
    predictor = predictor_class(
        dataset=dataset,
        load_dir=load_dir
    )
```

### Existing Code
All existing code continues to work:
- ✅ `results.py` - works unchanged
- ✅ `ood_evaluation/run_ood.py` - works unchanged
- ✅ Training scripts in `__main__` - updated to use new convenience function
- ✅ All checkpoints remain compatible (same .joblib format)

## Migration Guide

### For New Code
Use the new API directly:

```python
predictor = TokenPerformancePredictor(token_limits=token_limits)
predictor.fit(X_train, y_train, token_count_train, ...)
```

### For Existing Code
No changes required! The convenience function `train_predictor_from_dataset()` maintains full backward compatibility.

### For Custom Training Scripts
Replace:
```python
# OLD
predictor = TokenPerformancePredictor(dataset=dataset)
predictor.fit(save_dir="...", plot_dir="...")
```

With:
```python
# NEW
predictor = train_predictor_from_dataset(
    dataset=dataset,
    save_dir="...",
    plot_dir="..."
)
```

Or use the new API directly for more control.

## Technical Details

### Class Attributes
**Before:**
- `self.train` - Full training data dict
- `self.test` - Full test data dict
- `self.tokens` - Token limit names
- `self.train_count` - Training token counts
- `self.test_count` - Test token counts

**After:**
- `self.token_limits` - Token limit names (config only)
- `self.score_predictors` - Dict of trained LinearRegression models
- `self.token_predictor` - Trained LinearRegression for token counts

### Method Signatures

**`__init__(token_limits, load_dir)`**
- `token_limits`: List of token limit column names (e.g., ['10_score', ...])
- `load_dir`: Optional directory to load pre-trained models

**`fit(X_train, y_train, token_count_train, X_test, y_test, token_count_test, save_dir, plot_dir)`**
- `X_train`: Training embeddings, shape (n_train, embedding_dim)
- `y_train`: Dict mapping token limits to training scores
- `token_count_train`: Training token counts (unlimited)
- `X_test`, `y_test`, `token_count_test`: Optional test data for evaluation
- `save_dir`, `plot_dir`: Optional directories for saving models and plots
- Returns: self (for method chaining)

**`predict(X)`**
- `X`: Query embeddings, shape (n_queries, embedding_dim)
- Returns: Tuple of (score_predictions, count_predictions)
  - score_predictions: shape (n_queries, n_token_limits)
  - count_predictions: shape (n_queries,)

## Benefits of Refactoring

1. **Memory Efficiency**: Model no longer stores full train/test datasets
2. **Flexibility**: Can train on different data subsets without creating new instances
3. **Testability**: Easier to unit test with small synthetic data
4. **Maintainability**: Clearer separation between model and data
5. **Consistency**: Follows sklearn API conventions that developers expect
6. **Reusability**: Can reuse trained model on different datasets

## Files Modified

1. ✅ `predictor_sklearn.py` - Refactored class, added convenience function
2. ✅ `llm_loader.py` - Updated to support both APIs with fallback
3. ✅ All other files - No changes needed (backward compatible)

## Verification

The refactoring has been tested to ensure:
- ✅ New API works correctly
- ✅ Backward compatibility maintained via `train_predictor_from_dataset()`
- ✅ Loading from checkpoints works
- ✅ `llm_loader.py` integration works
- ✅ Predictions are identical between old and new API

## Next Steps (Optional)

Consider similar refactoring for `predictor.py` (PyTorch version) to maintain consistency across both predictor implementations.
