# New API Summary: Separate Limited and Unlimited Inputs

## Key Change

The `fit()` method now requires **separate inputs** for limited budgets and unlimited setting, making the three-component architecture explicit in the API.

## New `fit()` Signature

```python
def fit(self,
        embedding_train: np.ndarray,                          # Query embeddings
        quality_train_limited: Dict[str, np.ndarray],         # Limited budget quality scores
        quality_train_unlimited: np.ndarray,                  # Unlimited quality scores
        token_count_train: np.ndarray,                        # Unlimited token counts
        embedding_test: Optional[np.ndarray] = None,
        quality_test_limited: Optional[Dict[str, np.ndarray]] = None,
        quality_test_unlimited: Optional[np.ndarray] = None,
        token_count_test: Optional[np.ndarray] = None,
        save_dir: Optional[str] = None,
        plot_dir: Optional[str] = None) -> 'TokenPerformancePredictor'
```

**Consistent naming convention:**
- `embedding_*`: Query embeddings (input features)
- `quality_*`: Quality/correctness scores (target for score predictors)
- `token_count_*`: Token usage counts (target for token count predictor)
- All follow the pattern: `{what}_{train/test}_{limited/unlimited}`

## Example Usage

### Direct API
```python
from predictor_sklearn import TokenPerformancePredictor

# Prepare training data
embedding_train = embeddings[train_idx]  # (n_train, 768)

# LIMITED budgets: 15 token limits
quality_train_limited = {
    '10_score': quality_10[train_idx],
    '20_score': quality_20[train_idx],
    '30_score': quality_30[train_idx],
    # ... up to ...
    '4000_score': quality_4000[train_idx]
}  # Dict with 15 keys (NOT including 'unlimited_score')

# UNLIMITED: separate array
quality_train_unlimited = quality_unlimited[train_idx]  # (n_train,)

# Token counts for unlimited
token_count_train = counts_unlimited[train_idx]  # (n_train,)

# Initialize and train
token_limits = ['10_score', '20_score', ..., '4000_score', 'unlimited_score']
predictor = TokenPerformancePredictor(token_limits=token_limits)

predictor.fit(
    embedding_train=embedding_train,
    quality_train_limited=quality_train_limited,      # 15 limited budgets
    quality_train_unlimited=quality_train_unlimited,  # 1 unlimited quality
    token_count_train=token_count_train,              # Unlimited token counts
    embedding_test=embedding_test,
    quality_test_limited=quality_test_limited,
    quality_test_unlimited=quality_test_unlimited,
    token_count_test=token_count_test,
    save_dir="./checkpoints/model",
    plot_dir="./plots/model"
)
```

### Convenience Function (Backward Compatible)

The `train_predictor_from_dataset()` function handles the splitting automatically:

```python
from predictor_sklearn import train_predictor_from_dataset
from router_dataset import RouterDataset

# Old workflow still works!
dataset = RouterDataset(
    embeddings=embeddings,
    score_df_path="data/GLM-4.5-Air.csv",
    target_tokens_score=token_limits,
    train_idx=train_idx,
    test_idx=test_idx
)

# Convenience function automatically splits limited vs unlimited
predictor = train_predictor_from_dataset(
    dataset=dataset,
    save_dir="./checkpoints/model",
    plot_dir="./plots/model"
)
```

**How it works internally:**
```python
# Inside train_predictor_from_dataset()
quality_train_limited = {k: v for k, v in train_data["y"].items()
                         if k != 'unlimited_score'}  # Filter out unlimited
quality_train_unlimited = train_data["y"]['unlimited_score']  # Extract unlimited
```

## Benefits of New API

### 1. Explicit Separation
```python
# OLD: Unlimited mixed with limited (unclear)
y_train = {
    '10_score': ...,
    '20_score': ...,
    # ...
    'unlimited_score': ...  # Hidden in the dict!
}

# NEW: Unlimited explicitly separate (clear!)
y_train_limited = {'10_score': ..., '20_score': ..., '4000_score': ...}
y_train_unlimited = ...  # Obvious it's separate!
```

### 2. Type Safety
- `y_train_limited`: Dict[str, np.ndarray] - 15 keys
- `y_train_unlimited`: np.ndarray - single array
- Compiler/IDE can catch errors if you pass wrong type

### 3. Documentation
Function signature clearly shows three inputs needed:
1. Limited budgets (dict)
2. Unlimited scores (array)
3. Unlimited token counts (array)

### 4. Flexibility
Easy to extend:
```python
# Can easily add different preprocessing for unlimited
y_train_unlimited_preprocessed = preprocess(y_train_unlimited)

# Can use different data sources
predictor.fit(
    X_train=X_train,
    y_train_limited=limited_from_source_A,
    y_train_unlimited=unlimited_from_source_B,  # Different source!
    token_count_train=counts_from_source_C
)
```

## Migration Guide

### Option 1: Use Convenience Function (No Changes)
If you're using `train_predictor_from_dataset()`, **no changes needed**!

### Option 2: Update Direct Calls
If you're calling `fit()` directly:

**Before:**
```python
predictor.fit(
    X_train=X,
    y_train=all_scores,  # Dict with all 16 token limits
    token_count_train=counts
)
```

**After:**
```python
# Split the dict
y_train_limited = {k: v for k, v in all_scores.items() if k != 'unlimited_score'}
y_train_unlimited = all_scores['unlimited_score']

predictor.fit(
    X_train=X,
    y_train_limited=y_train_limited,
    y_train_unlimited=y_train_unlimited,
    token_count_train=counts
)
```

## Complete Example

```python
import numpy as np
from predictor_sklearn import TokenPerformancePredictor

# Setup
token_limits = [
    '10_score', '20_score', '30_score', '40_score', '50_score',
    '80_score', '100_score', '150_score', '200_score', '300_score',
    '500_score', '800_score', '1200_score', '2000_score', '4000_score',
    'unlimited_score'
]

# Prepare training data
X_train = np.random.randn(1000, 768)  # 1000 queries, 768-dim embeddings

# Limited budgets (15 token limits)
y_train_limited = {
    '10_score': np.random.rand(1000),
    '20_score': np.random.rand(1000),
    '30_score': np.random.rand(1000),
    '40_score': np.random.rand(1000),
    '50_score': np.random.rand(1000),
    '80_score': np.random.rand(1000),
    '100_score': np.random.rand(1000),
    '150_score': np.random.rand(1000),
    '200_score': np.random.rand(1000),
    '300_score': np.random.rand(1000),
    '500_score': np.random.rand(1000),
    '800_score': np.random.rand(1000),
    '1200_score': np.random.rand(1000),
    '2000_score': np.random.rand(1000),
    '4000_score': np.random.rand(1000),
}

# Unlimited (separate)
y_train_unlimited = np.random.rand(1000)
token_count_train = np.random.randint(100, 2000, size=1000)

# Train
predictor = TokenPerformancePredictor(token_limits=token_limits)
predictor.fit(
    X_train=X_train,
    y_train_limited=y_train_limited,
    y_train_unlimited=y_train_unlimited,
    token_count_train=token_count_train,
    save_dir="./checkpoints/demo"
)

# Predict
X_new = np.random.randn(10, 768)
scores, counts = predictor.predict(X_new)
print(f"Scores shape: {scores.shape}")  # (10, 16) - still 16 token limits!
print(f"Counts shape: {counts.shape}")  # (10,)
```

## predict() Also Returns Three Items

To fully align with the three-component architecture, `predict()` now returns three separate outputs:

```python
quality_limited, quality_unlimited, token_count = predictor.predict(embedding)

# Returns:
# - quality_limited: (n_queries, 15) - quality for 15 limited budgets
# - quality_unlimited: (n_queries,) - quality for unlimited
# - token_count: (n_queries,) - token counts for unlimited
```

### Backward Compatibility: predict_combined()

For code expecting combined output, use `predict_combined()`:

```python
quality_all, token_count = predictor.predict_combined(embedding)

# Returns:
# - quality_all: (n_queries, 16) - all 16 token limits combined [limited + unlimited]
# - token_count: (n_queries,) - token counts for unlimited
```

## Summary

✅ **fit() takes separate inputs for limited vs unlimited**
- `quality_train_limited`: Dict with 15 limited budgets
- `quality_train_unlimited`: Array for unlimited scores
- Makes the three-component architecture explicit

✅ **predict() returns three separate outputs**
- Quality for limited budgets (15 values)
- Quality for unlimited (1 value)
- Token count for unlimited (1 value)
- Use `predict_combined()` for backward compatibility

✅ **Consistent naming throughout**
- `embedding_*`, `quality_*`, `token_count_*`
- Self-documenting and aligned

✅ **Backward compatibility maintained**
- `train_predictor_from_dataset()` handles splitting automatically
- `predict_combined()` provides old-style output
- `llm_loader.py` handles both APIs automatically

✅ **Clearer, more maintainable API**
- Explicit about what data is needed
- Type-safe and self-documenting
- Easier to extend and modify
