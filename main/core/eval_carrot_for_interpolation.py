#!/usr/bin/env python3
"""
Evaluate CARROT baselines for interpolation experiment comparison.

Note: CARROT only routes among LLMs with unlimited tokens,
while CoRE routes among (LLM, token_limit) pairs.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from main.shared.dataset_manager import DatasetManager
from main.shared.router_dataset import RouterDataset
from main.baselines.carrot.baselines_carrot import CarrotKNNBaseline, CarrotLinearBaseline, route_baseline


def compute_audc(costs: np.ndarray, perfs: np.ndarray) -> float:
    """Compute Area Under Deferral Curve (AUDC) with normalized cost."""
    sorted_idx = np.argsort(costs)
    sorted_cost = costs[sorted_idx]
    sorted_perf = perfs[sorted_idx]

    min_cost, max_cost = sorted_cost.min(), sorted_cost.max()
    if max_cost > min_cost:
        norm_cost = (sorted_cost - min_cost) / (max_cost - min_cost)
    else:
        norm_cost = np.zeros_like(sorted_cost)

    audc = np.trapz(sorted_perf, norm_cost)
    return audc


def main():
    """Evaluate CARROT baselines."""
    print("=" * 80)
    print("CARROT Baseline Evaluation for Interpolation Experiment")
    print("=" * 80)
    print("\nNote: CARROT routes among LLMs (unlimited tokens only)")
    print("      CoRE routes among (LLM, token_limit) pairs")

    # Initialize dataset manager
    dataset_manager = DatasetManager(
        embeddings_path='data/prompt_embeddings.pkl',
        train_ratio=0.8,
        seed=42
    )

    embeddings = dataset_manager.get_embeddings()
    train_idx, test_idx = dataset_manager.get_split_indices()

    # Token limits
    token_limits_score = [f'{t}_score' for t in [10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000]]
    token_limits_score.append('unlimited_score')

    # Same 7 models as interpolation experiment
    models = [
        ('Mistral-7B-Instruct-v0.2', 0.20, 'data/Mistral-7B-Instruct-v0.2.csv'),
        ('GLM-4.5-Air', 0.85, 'data/GLM-4.5-Air.csv'),
        ('gemma-3-4b-it', 0.06815, 'data/gemma-3-4b-it.csv'),
        ('Llama-3.1-70B-Instruct', 0.40, 'data/Llama-3.1-70B-Instruct.csv'),
        ('Llama-3.2-3B-Instruct', 0.02, 'data/Llama-3.2-3B-Instruct.csv'),
        ('Qwen2.5-Math-7B-Instruct', 0.35, 'data/Qwen2.5-Math-7B-Instruct.csv'),
        ('Qwen3-0.6B', 0.0173, 'data/Qwen3-0.6B.csv'),
    ]

    # Load data for CARROT (unlimited only)
    print(f"\nLoading {len(models)} LLMs...")
    llm_names = []
    embedding_train_list = []
    embedding_test_list = []
    quality_train_list = []
    quality_test_list = []
    token_count_train_list = []
    token_count_test_list = []
    sizes_list = []

    for model_name, model_size, csv_path in models:
        print(f"  Loading {model_name}...")
        llm_names.append(model_name)
        sizes_list.append(model_size)

        dataset = RouterDataset(
            embeddings=embeddings,
            score_df_path=csv_path,
            target_tokens_score=token_limits_score,
            train_idx=train_idx,
            test_idx=test_idx
        )

        # CARROT uses unlimited quality and token counts
        train_data = dataset.get_train_set_score()
        test_data = dataset.get_test_set_score()

        quality_train_list.append(train_data['y']['unlimited_score'])
        quality_test_list.append(test_data['y']['unlimited_score'])
        token_count_train_list.append(dataset.get_train_token_unlimited_count())
        token_count_test_list.append(dataset.get_test_token_unlimited_count())

        # Embeddings (same for all models)
        if len(embedding_train_list) == 0:
            embedding_train_list = train_data['X']
            embedding_test_list = test_data['X']

    # Stack to (n_queries, n_models)
    quality_train = np.stack(quality_train_list, axis=1)
    quality_test = np.stack(quality_test_list, axis=1)
    token_count_train = np.stack(token_count_train_list, axis=1)
    token_count_test = np.stack(token_count_test_list, axis=1)
    sizes_vec = np.array(sizes_list)[None, :]

    print(f"\nData shapes:")
    print(f"  embedding_train: {embedding_train_list.shape}")
    print(f"  quality_train: {quality_train.shape}")
    print(f"  token_count_train: {token_count_train.shape}")

    # Lambda range
    lambda_range = np.linspace(0, 1e-4, 100)

    # Evaluate CARROT-KNN
    print("\n" + "=" * 80)
    print("Training and Evaluating CARROT-KNN")
    print("=" * 80)
    carrot_knn = CarrotKNNBaseline(n_neighbors_score=256, n_neighbors_count=256)
    carrot_knn.fit(embedding_train_list, quality_train, token_count_train)
    Y_hat_score_knn, Y_hat_count_knn = carrot_knn.predict(embedding_test_list)
    costs_knn, perfs_knn = route_baseline(
        Y_hat_score_knn, Y_hat_count_knn, quality_test, token_count_test, lambda_range, sizes_vec
    )
    audc_knn = compute_audc(costs_knn, perfs_knn)
    peak_acc_knn = perfs_knn.max()
    print(f"CARROT-KNN: AUDC = {audc_knn:.4f}, Peak Accuracy = {peak_acc_knn:.4f}")

    # Evaluate CARROT-Linear
    print("\n" + "=" * 80)
    print("Training and Evaluating CARROT-Linear")
    print("=" * 80)
    carrot_linear = CarrotLinearBaseline(fit_intercept=True)
    carrot_linear.fit(embedding_train_list, quality_train, token_count_train)
    Y_hat_score_linear, Y_hat_count_linear = carrot_linear.predict(embedding_test_list)
    costs_linear, perfs_linear = route_baseline(
        Y_hat_score_linear, Y_hat_count_linear, quality_test, token_count_test, lambda_range, sizes_vec
    )
    audc_linear = compute_audc(costs_linear, perfs_linear)
    peak_acc_linear = perfs_linear.max()
    print(f"CARROT-Linear: AUDC = {audc_linear:.4f}, Peak Accuracy = {peak_acc_linear:.4f}")

    # Save results
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"CARROT-KNN:    AUDC = {audc_knn:.4f}, Peak Accuracy = {peak_acc_knn:.4f}")
    print(f"CARROT-Linear: AUDC = {audc_linear:.4f}, Peak Accuracy = {peak_acc_linear:.4f}")

    # Save to file for plotting
    import os
    os.makedirs('comparison_results/interpolation_experiment', exist_ok=True)
    with open('comparison_results/interpolation_experiment/carrot_baselines.txt', 'w') as f:
        f.write(f"CARROT-KNN,{audc_knn:.6f},{peak_acc_knn:.6f}\n")
        f.write(f"CARROT-Linear,{audc_linear:.6f},{peak_acc_linear:.6f}\n")

    print("\n✓ Saved baseline results to comparison_results/interpolation_experiment/carrot_baselines.txt")


if __name__ == "__main__":
    main()
