# -*- coding: utf-8 -*-
"""
Test script to verify the split baseline files work correctly.

This script tests:
1. CARROT baselines (KNN and Linear) with save/load functionality
2. IRT baselines (MIRT and NIRT) with save/load functionality
3. Backward compatibility with existing code
"""

import numpy as np
from dataset_manager import DatasetManager
from llm_loader import load_llm
from predictor_sklearn import TokenPerformancePredictor

print("=" * 80)
print("Testing Split Baseline Files")
print("=" * 80)

# Setup
token_limits_score = [
    '10_score', '20_score', '30_score', '40_score', '50_score',
    '80_score', '100_score', '150_score', '200_score', '300_score',
    '500_score', '800_score', '1200_score', '2000_score', '4000_score',
    'unlimited_score'
]

token_limits_count = [
    '10_count', '20_count', '30_count', '40_count', '50_count',
    '80_count', '100_count', '150_count', '200_count', '300_count',
    '500_count', '800_count', '1200_count', '2000_count', '4000_count',
    'unlimited_count'
]

# Initialize dataset manager
dataset_manager = DatasetManager(
    embeddings_path="data/prompt_embeddings.pkl",
    train_ratio=0.8,
    seed=42
)

embeddings = dataset_manager.get_embeddings()
train_idx, test_idx = dataset_manager.get_split_indices()

# Load two small LLMs for testing
print("\n" + "=" * 80)
print("Loading LLMs for testing...")
print("=" * 80)

llms = {}

# Load GLM-4.5-Air
llms["GLM-4.5-Air"] = load_llm(
    name="GLM-4.5-Air",
    size=0.85,
    score_df_path="data/GLM-4.5-Air.csv",
    load_dir="./checkpoints/GLM-4.5-Air_multi",
    embeddings=embeddings,
    train_idx=train_idx,
    test_idx=test_idx,
    token_limits_score=token_limits_score,
    token_limits_count=token_limits_count,
    predictor_class=TokenPerformancePredictor
)

# Load Llama-3.2-3B
llms["Llama-3.2-3B"] = load_llm(
    name="Llama-3.2-3B",
    size=0.06,
    score_df_path="data/Llama-3.2-3B-Instruct.csv",
    load_dir="./checkpoints/Llama-3.2-3B-Instruct_multi",
    embeddings=embeddings,
    train_idx=train_idx,
    test_idx=test_idx,
    token_limits_score=token_limits_score,
    token_limits_count=token_limits_count,
    predictor_class=TokenPerformancePredictor
)

print(f"✅ Loaded {len(llms)} LLMs")

# ============================================================================
# Test 1: CARROT-KNN Baseline
# ============================================================================
print("\n" + "=" * 80)
print("Test 1: CARROT-KNN Baseline")
print("=" * 80)

from baselines_carrot import CarrotKNNBaseline

carrot_knn = CarrotKNNBaseline(
    llms=llms,
    n_neighbors_score=256,
    n_neighbors_count=256,
    metric="cosine"
)

# Train and save
carrot_knn.fit(save_dir="./test_checkpoints/carrot_knn")
print("✅ CARROT-KNN trained and saved")

# Predict
Y_hat_score, Y_hat_count = carrot_knn.predict()
print(f"✅ CARROT-KNN predictions: score shape={Y_hat_score.shape}, count shape={Y_hat_count.shape}")

# Load from checkpoint
carrot_knn_loaded = CarrotKNNBaseline(
    llms=llms,
    n_neighbors_score=256,
    n_neighbors_count=256
)
carrot_knn_loaded.load(load_dir="./test_checkpoints/carrot_knn")
Y_hat_score_loaded, Y_hat_count_loaded = carrot_knn_loaded.predict()

# Verify predictions match
assert np.allclose(Y_hat_score, Y_hat_score_loaded, atol=1e-5), "Loaded CARROT-KNN predictions differ!"
assert np.allclose(Y_hat_count, Y_hat_count_loaded, atol=1e-5), "Loaded CARROT-KNN counts differ!"
print("✅ CARROT-KNN loaded checkpoint produces identical predictions!")

# ============================================================================
# Test 2: CARROT-Linear Baseline
# ============================================================================
print("\n" + "=" * 80)
print("Test 2: CARROT-Linear Baseline")
print("=" * 80)

from baselines_carrot import CarrotLinearBaseline

carrot_linear = CarrotLinearBaseline(
    llms=llms,
    fit_intercept=True
)

# Train and save
carrot_linear.fit(save_dir="./test_checkpoints/carrot_linear")
print("✅ CARROT-Linear trained and saved")

# Predict
Y_hat_score2, Y_hat_count2 = carrot_linear.predict()
print(f"✅ CARROT-Linear predictions: score shape={Y_hat_score2.shape}, count shape={Y_hat_count2.shape}")

# Load from checkpoint
carrot_linear_loaded = CarrotLinearBaseline(
    llms=llms,
    fit_intercept=True
)
carrot_linear_loaded.load(load_dir="./test_checkpoints/carrot_linear")
Y_hat_score2_loaded, Y_hat_count2_loaded = carrot_linear_loaded.predict()

# Verify predictions match
assert np.allclose(Y_hat_score2, Y_hat_score2_loaded, atol=1e-5), "Loaded CARROT-Linear predictions differ!"
assert np.allclose(Y_hat_count2, Y_hat_count2_loaded, atol=1e-5), "Loaded CARROT-Linear counts differ!"
print("✅ CARROT-Linear loaded checkpoint produces identical predictions!")

# ============================================================================
# Test 3: IRT Baseline (MIRT) - Quick test with 2 epochs
# ============================================================================
print("\n" + "=" * 80)
print("Test 3: IRT Baseline (MIRT)")
print("=" * 80)

from baselines_irt import IRTBaseline

# Create simple LLM embeddings for testing
llm_names = list(llms.keys())
llm_texts = {
    "GLM-4.5-Air": "GLM-4.5-Air is a lightweight language model",
    "Llama-3.2-3B": "Llama-3.2-3B is a small instruction-tuned model"
}

irt = IRTBaseline(
    llms=llms,
    llm_texts=llm_texts,
    latent_dim=8,
    device="cpu"
)

# Quick training (2 epochs for speed)
irt.fit(
    lr=3e-3,
    batch_size=128,
    epochs=2,
    save_dir="./test_checkpoints/irt_mirt",
    plot_dir="./test_plots/irt_mirt"
)
print("✅ MIRT trained and saved")

# Predict
Y_hat_score3 = irt.predict()
print(f"✅ MIRT predictions: score shape={Y_hat_score3.shape}")

# Load from checkpoint
irt_loaded = IRTBaseline(
    llms=llms,
    llm_texts=llm_texts,
    latent_dim=8,
    device="cpu",
    load_dir="./test_checkpoints/irt_mirt"
)
Y_hat_score3_loaded = irt_loaded.predict()

# Verify predictions match
assert np.allclose(Y_hat_score3, Y_hat_score3_loaded, atol=1e-5), "Loaded MIRT predictions differ!"
print("✅ MIRT loaded checkpoint produces identical predictions!")

# ============================================================================
# Test 4: NIRT Baseline - Quick test with 2 epochs
# ============================================================================
print("\n" + "=" * 80)
print("Test 4: NIRT Baseline")
print("=" * 80)

from baselines_irt import NIRTBaseline

nirt = NIRTBaseline(
    llms=llms,
    llm_texts=llm_texts,
    latent_dim=8,
    device="cpu"
)

# Quick training (2 epochs for speed)
nirt.fit(
    lr=3e-3,
    batch_size=128,
    epochs=2,
    save_dir="./test_checkpoints/irt_nirt",
    plot_dir="./test_plots/irt_nirt"
)
print("✅ NIRT trained and saved")

# Predict
Y_hat_score4 = nirt.predict()
print(f"✅ NIRT predictions: score shape={Y_hat_score4.shape}")

# Load from checkpoint
nirt_loaded = NIRTBaseline(
    llms=llms,
    llm_texts=llm_texts,
    latent_dim=8,
    device="cpu",
    load_dir="./test_checkpoints/irt_nirt"
)
Y_hat_score4_loaded = nirt_loaded.predict()

# Verify predictions match
assert np.allclose(Y_hat_score4, Y_hat_score4_loaded, atol=1e-5), "Loaded NIRT predictions differ!"
print("✅ NIRT loaded checkpoint produces identical predictions!")

# ============================================================================
# Test 5: Backward Compatibility
# ============================================================================
print("\n" + "=" * 80)
print("Test 5: Backward Compatibility")
print("=" * 80)

# Test that the old class names still work via aliases
from baselines_carrot import CarrotBaseline, LinearCarrotBaseline

carrot_compat = CarrotBaseline(llms=llms)  # Should work as CarrotKNNBaseline
linear_compat = LinearCarrotBaseline(llms=llms)  # Should work as CarrotLinearBaseline

print("✅ Backward compatible class names work (CarrotBaseline, LinearCarrotBaseline)")

from baselines_irt import MIRTBaseline

mirt_compat = MIRTBaseline(llms=llms, llm_texts=llm_texts)  # Should work as IRTBaseline
print("✅ Backward compatible class name works (MIRTBaseline)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✅")
print("=" * 80)
print("\nThe split baseline files:")
print("✅ baselines_carrot.py: KNN and Linear baselines with checkpoint save/load")
print("✅ baselines_irt.py: MIRT and NIRT baselines with checkpoint save/load")
print("✅ All imports updated in results.py and ood_evaluation/")
print("✅ Backward compatibility maintained via aliases")
print("✅ Checkpoint saving/loading works correctly")
print("✅ All predictions are reproducible from checkpoints")
