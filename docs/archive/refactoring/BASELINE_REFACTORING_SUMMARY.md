# Baseline Refactoring Summary

## Overview

Refactored both `baselines_carrot.py` and `baselines_irt.py` to follow the sklearn design pattern used in `predictor_sklearn.py`. The key changes eliminate the dependency on `llm_loader.py` and follow a consistent data-passing approach.

## Key Changes

### 1. Updated `__init__` Methods

**Before (old pattern with llms parameter):**
```python
def __init__(self, llms: Dict[str, dict], ...):
    self.llms = llms
    # ... build training data from llms dict
```

**After (new sklearn pattern):**
```python
def __init__(self, load_dir: Optional[str] = None):
    # No llms parameter
    # Load from checkpoint if provided
    if load_dir:
        self.load(load_dir)
```

### 2. Updated `fit()` Methods

**CARROT Baselines:**
```python
def fit(self,
        embedding_train: np.ndarray,
        quality_train: np.ndarray,
        token_count_train: np.ndarray,
        embedding_test: Optional[np.ndarray] = None,
        quality_test: Optional[np.ndarray] = None,
        token_count_test: Optional[np.ndarray] = None,
        save_dir: Optional[str] = None) -> 'CarrotKNNBaseline':
```

**IRT Baselines:**
```python
def fit(self,
        embedding_train: np.ndarray,
        quality_train: np.ndarray,
        llm_embeddings: np.ndarray,
        llm_names: List[str],
        embedding_test: Optional[np.ndarray] = None,
        quality_test: Optional[np.ndarray] = None,
        lr: float = 3e-3,
        batch_size: int = 128,
        epochs: int = 200,
        weight_decay: float = 1e-6,
        save_dir: Optional[str] = None,
        plot_dir: Optional[str] = None) -> 'IRTBaseline':
```

### 3. Updated `predict()` Methods

All baselines now accept optional embeddings:

```python
def predict(self, embedding: Optional[np.ndarray] = None) -> ...:
    """
    Args:
        embedding: Query embeddings to predict on. If None, uses stored test embeddings.
    """
    if embedding is None:
        if self.X_test is None:
            raise ValueError("No test embeddings stored. Pass embedding argument or call fit() with test data.")
        embedding = self.X_test

    # ... perform prediction
```

### 4. Added Main Training Functions

Both files now include standalone training scripts in the `if __name__ == "__main__":` block.

**Pattern (following predictor_sklearn.py):**
```python
if __name__ == "__main__":
    from dataset_manager import DatasetManager
    from router_dataset import RouterDataset

    # 1. Initialize dataset manager
    dataset_manager = DatasetManager(...)
    embeddings = dataset_manager.get_embeddings()
    train_idx, test_idx = dataset_manager.get_split_indices()

    # 2. Define models to load (same as predictor_sklearn.py)
    models_to_load = [
        {"name": "GLM-4.5-Air", "csv": "data/GLM-4.5-Air.csv"},
        {"name": "GLM-4.6", "csv": "data/GLM-4.6.csv"},
        # ... more models
    ]

    # 3. Load data from CSVs
    for model_config in models_to_load:
        dataset = RouterDataset(
            embeddings=embeddings,
            score_df_path=csv_path,
            target_tokens_score=token_limits_score,
            train_idx=train_idx,
            test_idx=test_idx
        )
        # Extract unlimited scores/counts
        # ...

    # 4. Train baselines
    baseline.fit(
        embedding_train=embedding_train,
        quality_train=quality_train,
        ...,
        save_dir="./checkpoints/baseline_name"
    )
```

## Files Modified

### baselines_carrot.py

**Changes:**
- Removed `llms` parameter from `__init__` for both `CarrotKNNBaseline` and `CarrotLinearBaseline`
- Added `load_dir` parameter to `__init__`
- Updated `fit()` to accept training data directly (embeddings, quality, token counts)
- Updated `predict()` to accept optional embeddings parameter
- Removed `build_training_data()` method (no longer needed)
- Added main training function that loads data from CSVs

**Training Command:**
```bash
python baselines_carrot.py
# Trains both CARROT-KNN and CARROT-Linear
# Saves to ./checkpoints/carrot_knn/ and ./checkpoints/carrot_linear/
```

### baselines_irt.py

**Changes:**
- Removed `llms`, `llm_texts`, `llm_embeddings`, `text_encoder_name` parameters from `__init__` for both `IRTBaseline` and `NIRTBaseline`
- Added `load_dir` parameter to `__init__`
- Updated `fit()` to accept training data directly (embeddings, quality, llm_embeddings, llm_names)
- Updated `predict()` to accept optional embeddings parameter
- Removed `build_training_data()` method (no longer needed)
- Added main training function that loads data from CSVs and generates LLM embeddings

**Training Command:**
```bash
python baselines_irt.py
# Trains both MIRT and NIRT
# Saves to ./checkpoints/irt_mirt/ and ./checkpoints/irt_nirt/
```

## Usage Examples

### Training from Scratch

**CARROT Baselines:**
```bash
python baselines_carrot.py
```

**IRT Baselines:**
```bash
python baselines_irt.py
```

### Using in Code (New API)

**CARROT-KNN:**
```python
# Training
carrot_knn = CarrotKNNBaseline(n_neighbors_score=256)
carrot_knn.fit(
    embedding_train=X_train,
    quality_train=y_train,
    token_count_train=counts_train,
    save_dir="./checkpoints/carrot_knn"
)

# Prediction
predictions = carrot_knn.predict(X_test)

# Loading pre-trained
carrot_knn = CarrotKNNBaseline(load_dir="./checkpoints/carrot_knn")
predictions = carrot_knn.predict(X_test)
```

**MIRT:**
```python
# Training
irt = IRTBaseline(latent_dim=32, device="cuda")
irt.fit(
    embedding_train=X_train,
    quality_train=y_train,
    llm_embeddings=llm_emb,
    llm_names=names,
    save_dir="./checkpoints/irt_mirt"
)

# Prediction
predictions = irt.predict(X_test)

# Loading pre-trained
irt = IRTBaseline(latent_dim=32, load_dir="./checkpoints/irt_mirt")
predictions = irt.predict(X_test)
```

## Backward Compatibility

### Aliases
Both files maintain backward compatibility aliases:
```python
# In baselines_carrot.py
CarrotBaseline = CarrotKNNBaseline

# In baselines_irt.py
MIRTBaseline = IRTBaseline
```

### For Old Code Using results.py

The old code in `results.py` that uses `llm_loader.load_llm()` will need to be updated to use the new API. However, this is straightforward:

**Before:**
```python
from baselines_carrot import CarrotKNNBaseline

llms = {...}  # From llm_loader
carrot = CarrotKNNBaseline(llms=llms)
carrot.fit()
```

**After:**
```python
from baselines_carrot import CarrotKNNBaseline

# Build training data
embedding_train = ...
quality_train = ...
token_count_train = ...

carrot = CarrotKNNBaseline()
carrot.fit(
    embedding_train=embedding_train,
    quality_train=quality_train,
    token_count_train=token_count_train
)
```

## Benefits

1. **Consistency**: All predictors/baselines follow the same sklearn-like API
2. **Independence**: No dependency on `llm_loader.py` for baseline training
3. **Flexibility**: Can train baselines standalone without loading full LLM data
4. **Simplicity**: Data is passed explicitly, not hidden in nested dictionaries
5. **Reusability**: Easier to use baselines in different contexts

## Next Steps

### Update results.py

The `results.py` file still uses the old API where baselines are initialized with `llms` dictionaries from `llm_loader`. This needs to be updated to:

1. Build training data arrays from loaded LLM data
2. Call baseline `fit()` methods with arrays instead of passing `llms` dict
3. Update prediction calls to use the new API

**Example refactoring:**
```python
# Old way in results.py:
carrot = CarrotBaseline(llms).fit()
Y_hat_score, Y_hat_count = carrot.predict()

# New way:
# Build training data
any_llm = next(iter(llms.values()))
embedding_test = any_llm["test_embeddings"]
quality_test = np.stack([llms[name]["true_test_unlimited_score"] for name in llm_names], axis=1)
token_count_test = np.stack([llms[name]["true_test_unlimited_count"] for name in llm_names], axis=1)

# Train baseline
carrot = CarrotBaseline(load_dir="./checkpoints/carrot_knn")  # Or train from scratch
Y_hat_score, Y_hat_count = carrot.predict(embedding_test)

# Store ground truth for routing
carrot.Y_score_test_true = quality_test
carrot.Y_count_test_true = token_count_test
```

## Summary

✅ Both baseline files now follow sklearn design pattern
✅ No `llms` parameter in `__init__`
✅ Data passed explicitly to `fit()` and `predict()`
✅ Standalone training scripts added (following `predictor_sklearn.py` pattern)
✅ Checkpoint save/load support maintained
✅ Backward compatibility maintained via aliases
✅ Consistent API across all prediction components

The refactoring makes the baseline training independent of `llm_loader.py` and allows users to train baselines by simply running:
- `python baselines_carrot.py`
- `python baselines_irt.py`
