# Final API Summary: Complete Three-Component Architecture

## Overview

The `TokenPerformancePredictor` in `predictor_sklearn.py` now has a **complete three-component architecture** reflected in both training and prediction:

### Three Components
1. **Limited Budget Quality Predictors** (15 models)
2. **Unlimited Quality Predictor** (1 model)
3. **Unlimited Token Count Predictor** (1 model)

### API Alignment
Both `fit()` and `predict()` methods now explicitly work with these three components.

---

## Training API: fit()

### Signature
```python
def fit(self,
        embedding_train: np.ndarray,                          # (n_train, 768)
        quality_train_limited: Dict[str, np.ndarray],         # 15 limited budgets
        quality_train_unlimited: np.ndarray,                  # (n_train,) unlimited
        token_count_train: np.ndarray,                        # (n_train,) unlimited
        embedding_test: Optional[np.ndarray] = None,
        quality_test_limited: Optional[Dict[str, np.ndarray]] = None,
        quality_test_unlimited: Optional[np.ndarray] = None,
        token_count_test: Optional[np.ndarray] = None,
        save_dir: Optional[str] = None,
        plot_dir: Optional[str] = None) -> 'TokenPerformancePredictor'
```

### Three Inputs for Three Components
1. `quality_train_limited` → Trains **15 limited budget models**
2. `quality_train_unlimited` → Trains **1 unlimited quality model**
3. `token_count_train` → Trains **1 unlimited token count model**

### Example
```python
predictor.fit(
    embedding_train=embeddings[train_idx],
    quality_train_limited={
        '10_score': quality_10[train_idx],
        '20_score': quality_20[train_idx],
        # ... 15 total
        '4000_score': quality_4000[train_idx]
    },
    quality_train_unlimited=quality_unlimited[train_idx],
    token_count_train=token_counts[train_idx]
)
```

---

## Prediction API: predict()

### Signature
```python
def predict(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        Tuple of (quality_pred_limited, quality_pred_unlimited, token_count_pred):
            - quality_pred_limited: shape (n_queries, 15) - limited budgets
            - quality_pred_unlimited: shape (n_queries,) - unlimited
            - token_count_pred: shape (n_queries,) - unlimited token counts
    """
```

### Three Outputs from Three Components
1. `quality_pred_limited` (n, 15) → From **15 limited budget models**
2. `quality_pred_unlimited` (n,) → From **1 unlimited quality model**
3. `token_count_pred` (n,) → From **1 unlimited token count model**

### Example
```python
# New API - three separate outputs
quality_limited, quality_unlimited, token_count = predictor.predict(embedding)

print(quality_limited.shape)    # (100, 15) - 15 limited budgets
print(quality_unlimited.shape)  # (100,) - unlimited quality
print(token_count.shape)        # (100,) - unlimited tokens
```

---

## Backward Compatibility

### predict_combined()

For code expecting combined (n, 16) output:

```python
def predict_combined(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        Tuple of (quality_predictions, token_count_predictions):
            - quality_predictions: shape (n_queries, 16) - all 16 combined
            - token_count_predictions: shape (n_queries,) - unlimited token counts
    """
```

**Example:**
```python
# Old-style API - combined output
quality_all, token_count = predictor.predict_combined(embedding)

print(quality_all.shape)  # (100, 16) - all 16 token limits [15 limited + 1 unlimited]
print(token_count.shape)  # (100,) - unlimited tokens
```

### train_predictor_from_dataset()

Convenience function handles data splitting automatically:

```python
predictor = train_predictor_from_dataset(dataset, save_dir, plot_dir)
# Internally splits quality_train into limited and unlimited
```

### llm_loader.py

Automatically detects and handles both APIs:

```python
# In llm_loader.py
test_pred = predictor.predict(embeddings)

if len(test_pred) == 3:
    # New API: handle three outputs
    quality_limited, quality_unlimited, token_count = test_pred
    quality_all = np.column_stack([quality_limited, quality_unlimited])
else:
    # Old API: handle two outputs
    quality_all, token_count = test_pred
```

---

## Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  TokenPerformancePredictor                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Component 1: Limited Budget Quality Predictors       │  │
│  │ • limited_score_predictors: Dict[str, LinearReg]     │  │
│  │ • 15 models: 10, 20, 30, ..., 4000                   │  │
│  │ • Input:  quality_train_limited                      │  │
│  │ • Output: quality_pred_limited (n, 15)               │  │
│  │ • File:   limited_score_predictors.joblib            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Component 2: Unlimited Quality Predictor             │  │
│  │ • unlimited_score_predictor: LinearRegression        │  │
│  │ • 1 model: unlimited                                 │  │
│  │ • Input:  quality_train_unlimited                    │  │
│  │ • Output: quality_pred_unlimited (n,)                │  │
│  │ • File:   unlimited_score_predictor.joblib           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Component 3: Unlimited Token Count Predictor         │  │
│  │ • unlimited_token_predictor: LinearRegression        │  │
│  │ • 1 model: token count prediction                    │  │
│  │ • Input:  token_count_train                          │  │
│  │ • Output: token_count_pred (n,)                      │  │
│  │ • File:   unlimited_token_predictor.joblib           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Training Flow:
  fit(embedding_train, quality_train_limited, quality_train_unlimited, token_count_train)
    ↓
  [1/3] Train 15 limited budget models
  [2/3] Train 1 unlimited quality model
  [3/3] Train 1 unlimited token count model

Prediction Flow:
  predict(embedding)
    ↓
  Returns: (quality_limited, quality_unlimited, token_count)
           ↓              ↓                    ↓
        (n, 15)         (n,)                (n,)
```

---

## Naming Convention

All parameters follow a consistent pattern: **`{what}_{train/test}_{limited/unlimited}`**

| Prefix | Meaning | Shape/Type |
|--------|---------|------------|
| `embedding_*` | Query embeddings (input) | (n, 768) array |
| `quality_*` | Quality/correctness scores | [0,1] scores |
| `token_count_*` | Token usage counts | Integer counts |

| Suffix | Meaning |
|--------|---------|
| `_train` | Training data |
| `_test` | Test data |
| `_limited` | Limited token budgets (15) |
| `_unlimited` | Unlimited setting (1) |

---

## Migration Guide

### For New Code

Use the three-output API directly:

```python
from predictor_sklearn import TokenPerformancePredictor

# Train
predictor = TokenPerformancePredictor(token_limits)
predictor.fit(
    embedding_train,
    quality_train_limited,      # 15 limited budgets
    quality_train_unlimited,    # 1 unlimited
    token_count_train
)

# Predict
quality_limited, quality_unlimited, token_count = predictor.predict(embedding)

# Use separate outputs
for budget_idx in range(15):
    print(f"Budget {budget_idx}: quality = {quality_limited[:, budget_idx]}")

print(f"Unlimited: quality = {quality_unlimited}, tokens = {token_count}")
```

### For Existing Code

**Option 1:** Use convenience functions (no changes)
```python
predictor = train_predictor_from_dataset(dataset, save_dir, plot_dir)
quality_all, token_count = predictor.predict_combined(embedding)
```

**Option 2:** Adapt to new API
```python
# Old
quality_all, token_count = predictor.predict(embedding)

# New
quality_limited, quality_unlimited, token_count = predictor.predict(embedding)
quality_all = np.column_stack([quality_limited, quality_unlimited])
```

---

## Benefits

### ✅ Complete Architectural Alignment
- Training takes 3 inputs → trains 3 components
- Prediction returns 3 outputs → from 3 components
- Perfect symmetry and clarity

### ✅ Explicit Separation
- Limited budgets clearly separate from unlimited
- Each component's role is obvious
- No confusion about what's being predicted

### ✅ Flexible Usage
- Can use limited and unlimited separately
- Can combine when needed
- Backward compatibility maintained

### ✅ Self-Documenting
```python
# Just by reading the signature, you know:
quality_limited,      # 15 limited budgets
quality_unlimited,    # 1 unlimited
token_count          # Token usage
  = predictor.predict(embedding)
```

### ✅ Consistent Naming
All parameters follow the same convention throughout:
- `embedding_train`, `embedding_test`
- `quality_train_limited`, `quality_test_limited`
- `quality_train_unlimited`, `quality_test_unlimited`
- `token_count_train`, `token_count_test`

---

## Summary

The refactored API now perfectly reflects the three-component architecture:

| Component | Training Input | Prediction Output | File |
|-----------|---------------|-------------------|------|
| Limited budgets (15) | `quality_train_limited` | `quality_pred_limited` | `limited_score_predictors.joblib` |
| Unlimited quality (1) | `quality_train_unlimited` | `quality_pred_unlimited` | `unlimited_score_predictor.joblib` |
| Unlimited tokens (1) | `token_count_train` | `token_count_pred` | `unlimited_token_predictor.joblib` |

**Clean, explicit, and maintainable!** ✨
