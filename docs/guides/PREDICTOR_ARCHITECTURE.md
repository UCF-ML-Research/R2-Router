# Predictor Architecture Documentation

## Overview

The `TokenPerformancePredictor` trains **3 distinct components** (17 models total) for each LLM:

1. **Limited Budget Score Predictors** (15 models) - Dict of models for constrained token budgets
2. **Unlimited Quality Predictor** (1 model) - Separate model for unlimited quality prediction
3. **Unlimited Token Count Predictor** (1 model) - Predicts actual token usage in unlimited setting

### Component 1: Limited Budget Score Predictors (15 models)

Dictionary of linear regression models for **constrained token budgets**:

| Token Limit | Model Location | Input | Output | Purpose |
|-------------|----------------|-------|--------|---------|
| 10 | `limited_score_predictors['10_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 10 token budget |
| 20 | `limited_score_predictors['20_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 20 token budget |
| 30 | `limited_score_predictors['30_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 30 token budget |
| 40 | `limited_score_predictors['40_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 40 token budget |
| 50 | `limited_score_predictors['50_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 50 token budget |
| 80 | `limited_score_predictors['80_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 80 token budget |
| 100 | `limited_score_predictors['100_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 100 token budget |
| 150 | `limited_score_predictors['150_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 150 token budget |
| 200 | `limited_score_predictors['200_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 200 token budget |
| 300 | `limited_score_predictors['300_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 300 token budget |
| 500 | `limited_score_predictors['500_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 500 token budget |
| 800 | `limited_score_predictors['800_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 800 token budget |
| 1200 | `limited_score_predictors['1200_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 1200 token budget |
| 2000 | `limited_score_predictors['2000_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 2000 token budget |
| 4000 | `limited_score_predictors['4000_score']` | 768-dim embedding | Quality [0,1] | Predict quality with 4000 token budget |

**Storage**: `limited_score_predictors.joblib` - Dictionary with 15 LinearRegression models

### Component 2: Unlimited Quality Predictor (1 model)

Single linear regression model for **unlimited quality prediction**:

| Model | Input | Output | Purpose |
|-------|-------|--------|---------|
| `unlimited_score_predictor` | 768-dim embedding | Quality score [0,1] | Predict quality with no token limit |

**Storage**: `unlimited_score_predictor.joblib` - Single LinearRegression model

**Note**: This is trained **separately** from limited budgets to give it distinct treatment.

### Component 3: Unlimited Token Count Predictor (1 model)

Single linear regression model to predict **actual token usage** in unlimited setting:

| Model | Input | Output | Purpose |
|-------|-------|--------|---------|
| `unlimited_token_predictor` | 768-dim embedding | Token count (scalar) | Predict how many tokens will be used |

**Storage**: `unlimited_token_predictor.joblib` - Single LinearRegression model

**Note**: For limited token settings (10, 20, 30, etc.), the token count is **fixed** to the limit value. Only the unlimited setting requires prediction since the actual usage varies by query.

## Training Flow

```
For each LLM (e.g., GLM-4.5-Air):
├── Load training data
│   ├── X_train: embeddings (n_train × 768)
│   ├── y_train: dict of scores for each token limit
│   │   ├── '10_score': (n_train,) array
│   │   ├── '20_score': (n_train,) array
│   │   ├── ...
│   │   ├── '4000_score': (n_train,) array
│   │   └── 'unlimited_score': (n_train,) array
│   └── token_count_train: unlimited counts (n_train,)
│
├── [1/3] Train 15 limited budget score predictors
│   ├── For token_limit in ['10_score', '20_score', ..., '4000_score']:
│   │   └── model = LinearRegression()
│   │       model.fit(X_train, y_train[token_limit])
│   │       limited_score_predictors[token_limit] = model
│   └── Save: limited_score_predictors.joblib
│
├── [2/3] Train 1 unlimited quality predictor
│   ├── model = LinearRegression()
│   │   model.fit(X_train, y_train['unlimited_score'])
│   │   unlimited_score_predictor = model
│   └── Save: unlimited_score_predictor.joblib
│
└── [3/3] Train 1 unlimited token count predictor
    ├── model = LinearRegression()
    │   model.fit(X_train, token_count_train)
    │   unlimited_token_predictor = model
    └── Save: unlimited_token_predictor.joblib
```

## Prediction Flow

```
Given query embedding x:
├── Predict scores for 15 limited budgets
│   ├── s_10 = limited_score_predictors['10_score'].predict(x)
│   ├── s_20 = limited_score_predictors['20_score'].predict(x)
│   ├── ...
│   └── s_4000 = limited_score_predictors['4000_score'].predict(x)
│
├── Predict score for unlimited (separate model)
│   └── s_unlimited = unlimited_score_predictor.predict(x)
│
├── Combine all scores → [s_10, s_20, ..., s_4000, s_unlimited] (16 total)
│
├── Predict token counts
│   ├── For limited budgets: use fixed values
│   │   ├── c_10 = 10
│   │   ├── c_20 = 20
│   │   └── ...
│   └── For unlimited: use predictor
│       └── c_unlimited = unlimited_token_predictor.predict(x)
│
└── Compute routing decision
    └── For each (model, token_limit) option:
        ├── cost = token_count × model_size
        ├── risk = (1-λ)×predicted_score - λ×cost
        └── Select argmax risk
```

## Key Clarifications

### Q: Why separate unlimited quality predictor from limited budgets?
**A:** Architectural clarity and flexibility:
- Limited budgets are **fundamentally different** - they have hard constraints
- Unlimited has **no constraint** - it's a different prediction problem
- Separating them allows different treatment (e.g., different model types, features, or ensembling)
- Makes the code more maintainable and explicit

### Q: What are the three components?
**A:**
1. **`limited_score_predictors`** (dict): 15 models for quality under token constraints
2. **`unlimited_score_predictor`** (single model): Quality prediction for unconstrained generation
3. **`unlimited_token_predictor`** (single model): Token usage prediction for unlimited setting

### Q: Why do we need both unlimited predictors (quality + token count)?
**A:** They predict different things:
- **`unlimited_score_predictor`**: predicts **quality/correctness** [0,1]
- **`unlimited_token_predictor`**: predicts **token usage** (scalar)

Both are needed for cost-performance tradeoff:
- Quality tells us how well the model will perform
- Token count tells us the cost (cost = tokens × model_size)

## Example

For a query embedding `x`:

```python
# Predict quality for limited budget
score_100 = predictor.limited_score_predictors['100_score'].predict([x])[0]
# Result: 0.70 (predicted correctness with 100 token limit)
cost_100 = 100 × model_size
# Result: 100 × 0.85B = 85 (cost units)

# Predict quality for unlimited setting (separate model)
unlimited_quality = predictor.unlimited_score_predictor.predict([x])[0]
# Result: 0.85 (predicted correctness score)

# Predict token usage for unlimited setting
unlimited_tokens = predictor.unlimited_token_predictor.predict([x])[0]
# Result: 1500 (predicted tokens to be used)

# Compute cost
unlimited_cost = unlimited_tokens × model_size
# Result: 1500 × 0.85B = 1275 (cost units)

# Decision: unlimited gives better quality (0.85 vs 0.70) but higher cost (1275 vs 85)
# Router decides based on λ parameter
```

## File Storage

After training, models are saved to:
```
checkpoints/{model-name}_multi/
├── limited_score_predictors.joblib     # Dict with 15 LinearRegression models
│                                       # Keys: ['10_score', '20_score', ..., '4000_score']
│
├── unlimited_score_predictor.joblib    # Single LinearRegression model for unlimited quality
│
└── unlimited_token_predictor.joblib    # Single LinearRegression model for unlimited token counts
```

**Three separate files** for the three components!

## Implementation Files

- **sklearn version**: `predictor_sklearn.py` - Uses LinearRegression
- **PyTorch version**: `predictor.py` - Uses MLP networks
- Both follow the same architecture: 16 score predictors + 1 token count predictor
