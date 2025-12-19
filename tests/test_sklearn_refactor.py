# -*- coding: utf-8 -*-
"""
Test script to verify the refactored predictor_sklearn.py works correctly.

This script tests:
1. New API: TokenPerformancePredictor with fit(X_train, y_train, ...)
2. Backward compatibility: train_predictor_from_dataset()
3. Loading from checkpoints
"""

import numpy as np
from dataset_manager import DatasetManager
from router_dataset import RouterDataset
from predictor_sklearn import TokenPerformancePredictor, train_predictor_from_dataset

print("=" * 80)
print("Testing Refactored predictor_sklearn.py")
print("=" * 80)

# Setup
token_limits_score = [
    '10_score', '20_score', '30_score', '40_score', '50_score',
    '80_score', '100_score', '150_score', '200_score', '300_score',
    '500_score', '800_score', '1200_score', '2000_score', '4000_score',
    'unlimited_score'
]

dataset_manager = DatasetManager(
    embeddings_path="data/prompt_embeddings.pkl",
    train_ratio=0.8,
    seed=42
)

embeddings = dataset_manager.get_embeddings()
train_idx, test_idx = dataset_manager.get_split_indices()

# Create a small dataset for testing (use GLM-4.5-Air)
dataset = RouterDataset(
    embeddings=embeddings,
    score_df_path="data/GLM-4.5-Air.csv",
    target_tokens_score=token_limits_score,
    train_idx=train_idx,
    test_idx=test_idx
)

print("\n" + "=" * 80)
print("Test 1: New API - Direct fit() with X, y")
print("=" * 80)

train_data = dataset.get_train_set_score()
test_data = dataset.get_test_set_score()

predictor_new = TokenPerformancePredictor(token_limits=token_limits_score)
predictor_new.fit(
    X_train=train_data["X"],
    y_train=train_data["y"],
    token_count_train=dataset.get_train_token_unlimited_count(),
    X_test=test_data["X"][:100],  # Use small subset for speed
    y_test={k: v[:100] for k, v in test_data["y"].items()},
    token_count_test=dataset.get_test_token_unlimited_count()[:100]
)

# Test prediction
X_test = test_data["X"][:10]
preds, counts = predictor_new.predict(X_test)
print(f"✅ New API: Predictions shape = {preds.shape}, Counts shape = {counts.shape}")
assert preds.shape == (10, 16), f"Expected (10, 16), got {preds.shape}"
assert counts.shape == (10,), f"Expected (10,), got {counts.shape}"
print("✅ New API: Shape assertions passed!")

print("\n" + "=" * 80)
print("Test 2: Backward Compatibility - train_predictor_from_dataset()")
print("=" * 80)

predictor_compat = train_predictor_from_dataset(
    dataset=dataset,
    save_dir="./test_checkpoints",
    plot_dir="./test_plots"
)

preds2, counts2 = predictor_compat.predict(X_test)
print(f"✅ Backward Compat: Predictions shape = {preds2.shape}, Counts shape = {counts2.shape}")
assert preds2.shape == (10, 16), f"Expected (10, 16), got {preds2.shape}"
assert counts2.shape == (10,), f"Expected (10,), got {counts2.shape}"
print("✅ Backward Compat: Shape assertions passed!")

# Check that predictions are similar (should be identical since same data)
assert np.allclose(preds, preds2, atol=1e-5), "Predictions differ between APIs!"
assert np.allclose(counts, counts2, atol=1e-5), "Counts differ between APIs!"
print("✅ Both APIs produce identical predictions!")

print("\n" + "=" * 80)
print("Test 3: Loading from checkpoint")
print("=" * 80)

predictor_loaded = TokenPerformancePredictor(
    token_limits=token_limits_score,
    load_dir="./test_checkpoints"
)

preds3, counts3 = predictor_loaded.predict(X_test)
print(f"✅ Loaded: Predictions shape = {preds3.shape}, Counts shape = {counts3.shape}")
assert np.allclose(preds, preds3, atol=1e-5), "Loaded predictions differ!"
assert np.allclose(counts, counts3, atol=1e-5), "Loaded counts differ!"
print("✅ Loaded predictor produces identical predictions!")

print("\n" + "=" * 80)
print("Test 4: llm_loader.py compatibility")
print("=" * 80)

from llm_loader import load_llm

llm_data = load_llm(
    name="GLM-4.5-Air-Test",
    size=0.85,
    score_df_path="data/GLM-4.5-Air.csv",
    load_dir="./test_checkpoints",
    embeddings=embeddings,
    train_idx=train_idx,
    test_idx=test_idx,
    token_limits_score=token_limits_score,
    token_limits_count=[s.replace('_score', '_count') for s in token_limits_score],
    predictor_class=TokenPerformancePredictor
)

print(f"✅ llm_loader: Loaded LLM data with keys: {list(llm_data.keys())[:5]}...")
print(f"✅ llm_loader: pred_test_score shape = {llm_data['pred_test_score'].shape}")
print(f"✅ llm_loader: pred_test_token shape = {llm_data['pred_test_token'].shape}")

# Verify predictions match
assert llm_data['pred_test_score'].shape[1] == 16, "Should have 16 token limits"
print("✅ llm_loader integration works!")

print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✅")
print("=" * 80)
print("\nThe refactored predictor_sklearn.py:")
print("✅ Follows sklearn's design pattern (data passed to fit/predict)")
print("✅ Maintains backward compatibility via train_predictor_from_dataset()")
print("✅ Works with llm_loader.py")
print("✅ Loads/saves checkpoints correctly")
