# Code Refactoring Summary

## Overview

Successfully refactored the codebase to **decouple train/test splits from model training**. All LLMs now share a centralized train/test split, ensuring fair comparisons and eliminating coupling issues.

## Changes Made

### 1. New File: `dataset_manager.py`

**Purpose**: Centralized management of train/test splits

**Key Features**:
- Creates train/test split once for all LLMs
- Uses fixed random seed (42) for reproducibility
- Provides shared embeddings and split indices to all components

**API**:
```python
from dataset_manager import DatasetManager

dm = DatasetManager(embeddings_path="data/prompt_embeddings.pkl", train_ratio=0.8, seed=42)
embeddings = dm.get_embeddings()
train_idx, test_idx = dm.get_split_indices()
```

### 2. Modified: `router_dataset.py`

**Before**: Created its own train/test split internally
```python
# OLD - Creates its own split
dataset = RouterDataset(
    embeddings_path="data/prompt_embeddings.pkl",
    score_df_path="data/GLM-4.6.csv",
    target_tokens_score=token_limits_score,
    train_ratio=0.8
)
```

**After**: Accepts pre-computed split indices
```python
# NEW - Accepts centralized split
dataset = RouterDataset(
    embeddings=embeddings,  # From DatasetManager
    score_df_path="data/GLM-4.6.csv",
    target_tokens_score=token_limits_score,
    train_idx=train_idx,  # From DatasetManager
    test_idx=test_idx     # From DatasetManager
)
```

### 3. Modified: `llm_loader.py`

**Before**: Loaded embeddings internally
```python
# OLD - Each LLM loaded embeddings separately
load_llm(
    name="GLM-4.5-Air",
    size=0.85,
    embeddings_path="data/prompt_embeddings.pkl",  # Loaded separately
    score_df_path="data/GLM-4.5-Air.csv",
    load_dir="./checkpoints/GLM-4.5-Air_1e4",
    token_limits_score=[...],
    token_limits_count=[...]
)
```

**After**: Uses centralized embeddings and split
```python
# NEW - Uses shared embeddings and split
load_llm(
    name="GLM-4.5-Air",
    size=0.85,
    score_df_path="data/GLM-4.5-Air.csv",
    load_dir="./checkpoints/GLM-4.5-Air_1e4",
    embeddings=embeddings,      # From DatasetManager
    train_idx=train_idx,        # From DatasetManager
    test_idx=test_idx,          # From DatasetManager
    token_limits_score=[...],
    token_limits_count=[...]
)
```

### 4. Modified: `results.py`

**Before**: Each LLM created its own split
```python
# OLD - No centralized split
llms = {
    "GLM_4_5_Air": load_llm(
        name="GLM-4.5-Air",
        embeddings_path="data/prompt_embeddings.pkl",
        ...
    )
}
```

**After**: All LLMs use centralized split
```python
# NEW - Centralized split initialization
dataset_manager = DatasetManager(
    embeddings_path="data/prompt_embeddings.pkl",
    train_ratio=0.8,
    seed=42
)

embeddings = dataset_manager.get_embeddings()
train_idx, test_idx = dataset_manager.get_split_indices()

token_limits_score = [...]
token_limits_count = [...]

llms = {
    "GLM_4_5_Air": load_llm(
        name="GLM-4.5-Air",
        embeddings=embeddings,
        train_idx=train_idx,
        test_idx=test_idx,
        token_limits_score=token_limits_score,
        token_limits_count=token_limits_count,
        ...
    )
}
```

### 5. Modified: `predictor.py` (main block)

Updated the `if __name__ == "__main__"` block to use the new architecture for training new models.

## Benefits

1. **Decoupled Architecture**: Train/test splits are no longer tied to individual models
2. **Consistency**: All LLMs guaranteed to use the exact same train/test split
3. **Fair Comparison**: Eliminates potential inconsistencies from separate splits
4. **Single Source of Truth**: `DatasetManager` is the only place that defines the split
5. **Easier Testing**: Can verify all models use the same split

## Verification

Created `test_refactored_code.py` to verify:
- ✓ All LLMs use the same train/test split sizes
- ✓ All LLMs see the same queries (matching embeddings)
- ✓ Ground truth differs appropriately between LLMs
- ✓ Prediction shapes are correct

**Test Results**: All tests passed successfully!

## How to Use

### For Evaluation (results.py)

```python
from dataset_manager import DatasetManager
from llm_loader import load_llm

# Step 1: Initialize centralized split
dataset_manager = DatasetManager(
    embeddings_path="data/prompt_embeddings.pkl",
    train_ratio=0.8,
    seed=42
)

embeddings = dataset_manager.get_embeddings()
train_idx, test_idx = dataset_manager.get_split_indices()

# Step 2: Define token limits (shared)
token_limits_score = ['10_score', ..., 'unlimited_score']
token_limits_count = ['10_count', ..., 'unlimited_count']

# Step 3: Load LLMs with centralized split
llms = {
    "LLM1": load_llm(
        name="LLM1",
        size=0.85,
        score_df_path="data/LLM1.csv",
        load_dir="./checkpoints/LLM1_1e4",
        embeddings=embeddings,
        train_idx=train_idx,
        test_idx=test_idx,
        token_limits_score=token_limits_score,
        token_limits_count=token_limits_count
    ),
    # ... more LLMs
}

# Step 4: Run evaluation as before
routing_cost, routing_perf = route_scores(llms, lamb_range)
```

### For Training New Models (predictor.py)

```python
from dataset_manager import DatasetManager
from router_dataset import RouterDataset
from predictor import TokenPerformancePredictor

# Step 1: Initialize centralized split
dataset_manager = DatasetManager(
    embeddings_path="data/prompt_embeddings.pkl",
    train_ratio=0.8,
    seed=42
)

embeddings = dataset_manager.get_embeddings()
train_idx, test_idx = dataset_manager.get_split_indices()

# Step 2: Create dataset with centralized split
dataset = RouterDataset(
    embeddings=embeddings,
    score_df_path="data/NEW_MODEL.csv",
    target_tokens_score=token_limits_score,
    train_idx=train_idx,
    test_idx=test_idx
)

# Step 3: Train predictor as before
predictor = TokenPerformancePredictor(
    hidden_dims=[256, 128, 64],
    dropout=0.5,
    dataset=dataset
)

predictor.fit(save_dir="./checkpoints/NEW_MODEL_1e4", lr=1e-4, epochs=100)
```

## Backward Compatibility

**Breaking Changes**: Yes, the API has changed for `RouterDataset` and `load_llm`.

**Migration Required**:
- Update `results.py` to use `DatasetManager`
- Update any training scripts to use `DatasetManager`
- No changes needed to trained model checkpoints (they remain compatible)

## Files Modified

1. ✅ `dataset_manager.py` - NEW
2. ✅ `router_dataset.py` - MODIFIED
3. ✅ `llm_loader.py` - MODIFIED
4. ✅ `results.py` - MODIFIED
5. ✅ `predictor.py` - MODIFIED (main block only)
6. ✅ `test_refactored_code.py` - NEW

## Status

**Completed and Tested**: All refactoring complete and verified to work correctly.

The code is now **runnable** and maintains all original functionality while providing better architecture and consistency.
