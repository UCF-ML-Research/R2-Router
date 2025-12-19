# Parameter Naming Improvements

## Overview

Improved parameter names in `predictor_sklearn.py` to be more descriptive and follow a consistent naming convention. All parameters now clearly indicate **what** they represent and follow the same style as `token_count_train`.

## Before vs After

### Old Names (Generic)
```python
def fit(self,
        X_train,                    # ❌ What is X?
        y_train_limited,            # ❌ What is y?
        y_train_unlimited,          # ❌ Inconsistent with token_count_train
        token_count_train,          # ✓ Good - descriptive!
        X_test,
        y_test_limited,
        y_test_unlimited,
        token_count_test,
        ...)
```

### New Names (Descriptive)
```python
def fit(self,
        embedding_train,            # ✅ Clear: query embeddings
        quality_train_limited,      # ✅ Clear: quality scores for limited budgets
        quality_train_unlimited,    # ✅ Clear: quality scores for unlimited
        token_count_train,          # ✅ Matches naming style
        embedding_test,
        quality_test_limited,
        quality_test_unlimited,
        token_count_test,
        ...)
```

## Naming Convention

All parameters now follow a consistent pattern:

```
{what}_{train/test}_{limited/unlimited}
```

### Prefixes (What it represents)
- **`embedding_`**: Query embeddings (input features, 768-dimensional vectors)
- **`quality_`**: Quality/correctness scores (targets for score predictors, [0,1])
- **`token_count_`**: Token usage counts (targets for token count predictor, integers)

### Suffixes (Train or Test)
- **`_train`**: Training data
- **`_test`**: Test/validation data

### Additional Suffix (For quality only)
- **`_limited`**: Limited token budgets (15 models)
- **`_unlimited`**: Unlimited setting (1 model)

## Complete Parameter List

### Training Data
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `embedding_train` | np.ndarray | (n_train, 768) | Training query embeddings |
| `quality_train_limited` | Dict[str, np.ndarray] | 15 × (n_train,) | Quality scores for limited budgets |
| `quality_train_unlimited` | np.ndarray | (n_train,) | Quality scores for unlimited |
| `token_count_train` | np.ndarray | (n_train,) | Token counts for unlimited |

### Test Data (Optional)
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `embedding_test` | np.ndarray | (n_test, 768) | Test query embeddings |
| `quality_test_limited` | Dict[str, np.ndarray] | 15 × (n_test,) | Quality scores for limited budgets |
| `quality_test_unlimited` | np.ndarray | (n_test,) | Quality scores for unlimited |
| `token_count_test` | np.ndarray | (n_test,) | Token counts for unlimited |

## Example with Clear Names

```python
# Old way (confusing)
predictor.fit(
    X_train=X,           # What is X?
    y_train_limited=y,   # What is y?
    ...
)

# New way (self-documenting)
predictor.fit(
    embedding_train=embeddings,              # Ah, query embeddings!
    quality_train_limited=quality_scores,    # Ah, quality scores!
    quality_train_unlimited=quality_unlimited,
    token_count_train=token_counts,
    ...
)
```

## Benefits

### 1. Self-Documenting Code
No need to check documentation - the parameter name tells you exactly what it is:
```python
embedding_train  # Obviously: embeddings for training
quality_train_limited  # Obviously: quality scores for training, limited budgets
token_count_train  # Obviously: token counts for training
```

### 2. Consistent Style
All parameters follow the same naming pattern, making the API predictable:
```python
# All follow: {what}_{train/test}_{limited/unlimited}
embedding_train
quality_train_limited
quality_train_unlimited
token_count_train

embedding_test
quality_test_limited
quality_test_unlimited
token_count_test
```

### 3. Clear Semantics
Each prefix has a clear meaning:
- `embedding_*` → Always shape (n, 768)
- `quality_*` → Always [0, 1] scores
- `token_count_*` → Always integer counts

### 4. IDE Autocomplete
When typing `quality_`, IDE shows all quality-related parameters:
```
quality_train_limited
quality_train_unlimited
quality_test_limited
quality_test_unlimited
```

### 5. Aligned with Domain
Names match the domain language:
- "embedding" is a standard ML/NLP term
- "quality" clearly indicates correctness/performance scores
- "token_count" is specific and unambiguous

## Migration Notes

### For New Code
Use the new parameter names everywhere:
```python
predictor.fit(
    embedding_train=...,
    quality_train_limited=...,
    quality_train_unlimited=...,
    token_count_train=...
)
```

### For Existing Code Using Convenience Function
**No changes needed!** The `train_predictor_from_dataset()` function handles everything:
```python
# Still works exactly the same
predictor = train_predictor_from_dataset(dataset, save_dir, plot_dir)
```

### For Existing Code Using Direct API
Update parameter names:
```python
# Before
predictor.fit(X_train=..., y_train_limited=..., ...)

# After
predictor.fit(embedding_train=..., quality_train_limited=..., ...)
```

## Comparison with sklearn Convention

### sklearn (Generic)
```python
model.fit(X, y)  # Generic: works for any ML task
```
- ✅ Good for general-purpose ML library
- ❌ Not descriptive for specific use cases

### Our API (Specific)
```python
predictor.fit(
    embedding_train,
    quality_train_limited,
    quality_train_unlimited,
    token_count_train
)
```
- ✅ Specific to our LLM routing task
- ✅ Self-documenting
- ✅ Prevents confusion about what data to pass

## Summary

✅ **Clear parameter names** that indicate what each represents
✅ **Consistent naming convention** across all parameters
✅ **Aligned with `token_count_train`** style
✅ **Self-documenting API** - no need to check docs constantly
✅ **Backward compatible** via convenience function

The API is now much more readable and maintainable!
