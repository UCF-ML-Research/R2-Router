# Three-Component Architecture Refactoring

## Summary

Refactored `predictor_sklearn.py` to train **three distinct components** instead of treating unlimited as just another token limit. This provides better architectural clarity and flexibility.

## Changes

### Before: Two Components (16 + 1 = 17 models)
```python
class TokenPerformancePredictor:
    def __init__(self, ...):
        self.score_predictors = {}      # Dict with ALL 16 token limits (including unlimited)
        self.token_predictor = ...       # Token count predictor
```

**Issues:**
- Unlimited was mixed with limited budgets in same dict
- Not architecturally clear that unlimited is fundamentally different
- Hard to apply different treatment to unlimited vs limited

### After: Three Components (15 + 1 + 1 = 17 models)
```python
class TokenPerformancePredictor:
    def __init__(self, ...):
        self.limited_score_predictors = {}      # Dict with 15 limited budgets only
        self.unlimited_score_predictor = ...    # Separate unlimited quality model
        self.unlimited_token_predictor = ...    # Token count predictor
```

**Benefits:**
- ✅ Clear separation: limited budgets vs unlimited
- ✅ Architecturally explicit about the three distinct tasks
- ✅ Allows different treatment for unlimited (different model types, ensembling, etc.)
- ✅ More maintainable and easier to understand

## The Three Components

### 1. Limited Budget Score Predictors (15 models)
```python
limited_score_predictors = {
    '10_score': LinearRegression(),
    '20_score': LinearRegression(),
    ...
    '4000_score': LinearRegression()
}
```
- **Purpose**: Predict quality under token constraints
- **Input**: 768-dim embedding
- **Output**: Quality score [0,1]
- **Storage**: `limited_score_predictors.joblib`

### 2. Unlimited Quality Predictor (1 model)
```python
unlimited_score_predictor = LinearRegression()
```
- **Purpose**: Predict quality without token constraints
- **Input**: 768-dim embedding
- **Output**: Quality score [0,1]
- **Storage**: `unlimited_score_predictor.joblib`
- **Why separate?**: Unlimited is fundamentally different from constrained generation

### 3. Unlimited Token Count Predictor (1 model)
```python
unlimited_token_predictor = LinearRegression()
```
- **Purpose**: Predict how many tokens will be used
- **Input**: 768-dim embedding
- **Output**: Token count (scalar)
- **Storage**: `unlimited_token_predictor.joblib`
- **Note**: Limited budgets don't need this (count = limit)

## Training Flow

```
[1/3] Train 15 limited budget score predictors
      - Loop through ['10_score', '20_score', ..., '4000_score']
      - Train one LinearRegression per budget
      - Save as dictionary: limited_score_predictors.joblib

[2/3] Train 1 unlimited quality predictor
      - Train LinearRegression on 'unlimited_score'
      - Save as single model: unlimited_score_predictor.joblib

[3/3] Train 1 unlimited token count predictor
      - Train LinearRegression on unlimited_count
      - Save as single model: unlimited_token_predictor.joblib
```

## API Changes

### Training API

**Old:**
```python
predictor.fit(
    X_train=X_train,
    y_train=y_train,  # Dict with ALL 16 token limits including 'unlimited_score'
    token_count_train=token_counts
)
```

**New:**
```python
predictor.fit(
    X_train=X_train,
    y_train_limited=y_train_limited,      # Dict with 15 limited budgets only
    y_train_unlimited=y_train_unlimited,  # Array for unlimited scores (separate!)
    token_count_train=token_counts
)
```

### Prediction API (unchanged)

```python
# Score for limited budgets
score_100 = predictor.limited_score_predictors['100_score'].predict(X)

# Score for unlimited (separate attribute)
score_unlimited = predictor.unlimited_score_predictor.predict(X)

# Token count for unlimited
count_unlimited = predictor.unlimited_token_predictor.predict(X)
```

## File Changes

### Checkpoint Files

**Before:**
```
checkpoints/{model-name}_multi/
├── score_predictor.joblib      # 16 models (mixed limited + unlimited)
└── token_predictor.joblib      # 1 model
```

**After:**
```
checkpoints/{model-name}_multi/
├── limited_score_predictors.joblib     # 15 models (limited only)
├── unlimited_score_predictor.joblib    # 1 model (unlimited quality)
└── unlimited_token_predictor.joblib    # 1 model (unlimited tokens)
```

### Code Files Modified

1. ✅ **`predictor_sklearn.py`**:
   - Updated `__init__()`: Three attributes instead of two
   - Updated `_load_models()`: Load three separate files
   - Updated `fit()`: Three-phase training with clear [1/3], [2/3], [3/3] sections
   - Updated `predict()`: Combine limited + unlimited predictions
   - Updated printing/logging to show three components

2. ✅ **`PREDICTOR_ARCHITECTURE.md`**:
   - Updated to document three-component architecture
   - Updated training flow diagrams
   - Updated prediction flow
   - Updated examples

3. ✅ **`THREE_COMPONENT_REFACTORING.md`**:
   - New document (this file) explaining the change

## Backward Compatibility

The `train_predictor_from_dataset()` convenience function still works unchanged. However, **old checkpoints are incompatible** with the new code because:
- Old: 2 files (`score_predictor.joblib`, `token_predictor.joblib`)
- New: 3 files (`limited_score_predictors.joblib`, `unlimited_score_predictor.joblib`, `unlimited_token_predictor.joblib`)

**Migration**: Retrain all models with the new code.

## Testing

The `predict()` method still returns the same shape:
- Score predictions: `(n_queries, 16)` - Same as before!
- Token predictions: `(n_queries,)` - Same as before!

The output is just constructed differently internally:
```python
# Old way: all from one dict
scores = [score_predictors[t].predict(X) for t in all_16_token_limits]

# New way: limited (15) + unlimited (1)
limited_scores = [limited_score_predictors[t].predict(X) for t in 15_limits]
unlimited_score = unlimited_score_predictor.predict(X)
scores = limited_scores + [unlimited_score]  # Still 16 total!
```

## Why This Design?

### Architectural Clarity
- **Limited budgets** have hard constraints → grouped together
- **Unlimited** has no constraints → treated separately
- Makes the conceptual difference explicit in code

### Flexibility
- Can use different model types for unlimited (e.g., ensemble, neural network)
- Can add unlimited-specific features without affecting limited models
- Easier to experiment with unlimited prediction strategies

### Maintainability
- Clear naming: `limited_score_predictors` vs `unlimited_score_predictor`
- Obvious what each component does
- Easier to debug and extend

## Next Steps

1. ✅ Retrain all models with new architecture
2. ✅ Update documentation
3. ⬜ Update `llm_loader.py` to handle both old and new checkpoint formats (for gradual migration)
4. ⬜ Consider similar refactoring for `predictor.py` (PyTorch version)
