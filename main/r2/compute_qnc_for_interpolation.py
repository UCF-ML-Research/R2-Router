#!/usr/bin/env python3
"""
Compute QNC for interpolation experiment and generate comparison plots.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from main.shared.dataset_manager import DatasetManager
from main.shared.router_dataset import RouterDataset
from main.r2.predictor_sklearn import TokenPerformancePredictor, route_scores
from main.baselines.carrot.baselines_carrot import CarrotKNNBaseline, CarrotLinearBaseline, route_baseline
from main.r2.experiment_interpolation import (
    TOKEN_LIMITS_NAMES,
    select_training_heads,
    train_with_subset_heads,
    compute_audc
)


def compute_qnc(costs: np.ndarray, perfs: np.ndarray,
                target_accuracy: float,
                global_min_cost: float,
                global_max_cost: float) -> float:
    """
    Compute Query-Normalized Cost with global normalization.

    Args:
        costs: Array of cost values
        perfs: Array of performance values
        target_accuracy: Target accuracy to reach
        global_min_cost: Global minimum cost across all methods
        global_max_cost: Global maximum cost across all methods

    Returns:
        QNC value [0, 1] where lower is better
    """
    # Find points where performance >= target
    valid_idx = perfs >= target_accuracy

    if not valid_idx.any():
        # Cannot reach target accuracy
        return 1.0

    # Find minimum cost among valid points
    min_cost_to_reach_target = costs[valid_idx].min()

    # Normalize using global min/max
    if global_max_cost > global_min_cost:
        qnc = (min_cost_to_reach_target - global_min_cost) / (global_max_cost - global_min_cost)
    else:
        qnc = 0.0

    return np.clip(qnc, 0, 1)


def main():
    """Compute QNC for all methods and generate plots."""
    print("=" * 80)
    print("Computing QNC for Interpolation Experiment")
    print("=" * 80)

    # Initialize dataset manager
    dataset_manager = DatasetManager(
        embeddings_path='data/prompt_embeddings.pkl',
        train_ratio=0.8,
        seed=42
    )

    embeddings = dataset_manager.get_embeddings()
    train_idx, test_idx = dataset_manager.get_split_indices()

    # Models (same as interpolation experiment)
    models = [
        ('Mistral-7B-Instruct-v0.2', 0.20, 'data/Mistral-7B-Instruct-v0.2.csv'),
        ('GLM-4.5-Air', 0.85, 'data/GLM-4.5-Air.csv'),
        ('gemma-3-4b-it', 0.06815, 'data/gemma-3-4b-it.csv'),
        ('Llama-3.1-70B-Instruct', 0.40, 'data/Llama-3.1-70B-Instruct.csv'),
        ('Llama-3.2-3B-Instruct', 0.02, 'data/Llama-3.2-3B-Instruct.csv'),
        ('Qwen2.5-Math-7B-Instruct', 0.35, 'data/Qwen2.5-Math-7B-Instruct.csv'),
        ('Qwen3-0.6B', 0.0173, 'data/Qwen3-0.6B.csv'),
    ]

    # Lambda range
    lambda_range = np.linspace(0, 1e-4, 100)
    n_heads_list = [2, 4, 6, 8, 10, 12, 14, 16]

    # Determine target accuracy (best single LLM performance)
    print("\nFinding best single LLM performance...")
    token_limits_score = TOKEN_LIMITS_NAMES + ['unlimited_score']
    best_llm_acc = 0.0
    for model_name, model_size, csv_path in models:
        dataset = RouterDataset(
            embeddings=embeddings,
            score_df_path=csv_path,
            target_tokens_score=token_limits_score,
            train_idx=train_idx,
            test_idx=test_idx
        )
        test_data = dataset.get_test_set_score()
        unlimited_acc = test_data['y']['unlimited_score'].mean()
        if unlimited_acc > best_llm_acc:
            best_llm_acc = unlimited_acc
            best_llm = model_name

    target_accuracy = best_llm_acc * 0.95  # 95% of best LLM
    print(f"Best LLM: {best_llm} with accuracy {best_llm_acc:.4f}")
    print(f"Target accuracy (95%): {target_accuracy:.4f}")

    # Collect all costs and performances for global normalization
    print("\n" + "=" * 80)
    print("Computing routing curves for all methods...")
    print("=" * 80)

    all_costs_list = []
    all_perfs_list = []
    method_results = {}

    # R2-Router with different heads
    for n_heads in tqdm(n_heads_list, desc="R2-Router configurations"):
        if n_heads == 16:
            trained_heads = TOKEN_LIMITS_NAMES.copy()
        else:
            trained_heads = select_training_heads(n_heads, strategy='uniform')

        llms = {}
        for model_name, model_size, csv_path in models:
            llms[model_name] = train_with_subset_heads(
                dataset_manager=dataset_manager,
                model_name=model_name,
                model_size=model_size,
                csv_path=csv_path,
                trained_heads=trained_heads
            )

        costs, perfs = route_scores(llms, lambda_range)
        all_costs_list.extend(costs)
        all_perfs_list.extend(perfs)
        method_results[f'R2-Router_{n_heads}heads'] = (costs, perfs)

    # CARROT baselines
    print("\nEvaluating CARROT baselines...")
    llm_names = []
    embedding_train_list = []
    embedding_test_list = []
    quality_train_list = []
    quality_test_list = []
    token_count_train_list = []
    token_count_test_list = []
    sizes_list = []

    for model_name, model_size, csv_path in models:
        llm_names.append(model_name)
        sizes_list.append(model_size)

        dataset = RouterDataset(
            embeddings=embeddings,
            score_df_path=csv_path,
            target_tokens_score=token_limits_score,
            train_idx=train_idx,
            test_idx=test_idx
        )

        train_data = dataset.get_train_set_score()
        test_data = dataset.get_test_set_score()

        quality_train_list.append(train_data['y']['unlimited_score'])
        quality_test_list.append(test_data['y']['unlimited_score'])
        token_count_train_list.append(dataset.get_train_token_unlimited_count())
        token_count_test_list.append(dataset.get_test_token_unlimited_count())

        if len(embedding_train_list) == 0:
            embedding_train_list = train_data['X']
            embedding_test_list = test_data['X']

    quality_train = np.stack(quality_train_list, axis=1)
    quality_test = np.stack(quality_test_list, axis=1)
    token_count_train = np.stack(token_count_train_list, axis=1)
    token_count_test = np.stack(token_count_test_list, axis=1)
    sizes_vec = np.array(sizes_list)[None, :]

    # CARROT-KNN
    carrot_knn = CarrotKNNBaseline(n_neighbors_score=256, n_neighbors_count=256)
    carrot_knn.fit(embedding_train_list, quality_train, token_count_train)
    Y_hat_score_knn, Y_hat_count_knn = carrot_knn.predict(embedding_test_list)
    costs_knn, perfs_knn = route_baseline(
        Y_hat_score_knn, Y_hat_count_knn, quality_test, token_count_test, lambda_range, sizes_vec
    )
    all_costs_list.extend(costs_knn)
    all_perfs_list.extend(perfs_knn)
    method_results['CARROT-KNN'] = (costs_knn, perfs_knn)

    # CARROT-Linear
    carrot_linear = CarrotLinearBaseline(fit_intercept=True)
    carrot_linear.fit(embedding_train_list, quality_train, token_count_train)
    Y_hat_score_linear, Y_hat_count_linear = carrot_linear.predict(embedding_test_list)
    costs_linear, perfs_linear = route_baseline(
        Y_hat_score_linear, Y_hat_count_linear, quality_test, token_count_test, lambda_range, sizes_vec
    )
    all_costs_list.extend(costs_linear)
    all_perfs_list.extend(perfs_linear)
    method_results['CARROT-Linear'] = (costs_linear, perfs_linear)

    # Global normalization
    global_min_cost = np.min(all_costs_list)
    global_max_cost = np.max(all_costs_list)
    print(f"\nGlobal cost range: [{global_min_cost:.2f}, {global_max_cost:.2f}]")

    # Compute metrics for all methods
    print("\n" + "=" * 80)
    print("Computing AUDC and QNC...")
    print("=" * 80)

    results = []
    for method_name, (costs, perfs) in method_results.items():
        audc = compute_audc(costs, perfs)
        qnc = compute_qnc(costs, perfs, target_accuracy, global_min_cost, global_max_cost)
        peak_acc = perfs.max()

        # Extract n_heads for R2-Router methods
        if 'R2-Router' in method_name:
            n_heads = int(method_name.split('_')[1].replace('heads', ''))
        else:
            n_heads = None

        results.append({
            'method': method_name,
            'n_heads': n_heads,
            'audc': audc,
            'qnc': qnc,
            'peak_accuracy': peak_acc
        })
        print(f"{method_name:20s}: AUDC={audc:.4f}, QNC={qnc:.4f}, Peak={peak_acc:.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    output_dir = 'comparison_results/interpolation_experiment'
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'all_methods_metrics.csv'), index=False)
    print(f"\n✓ Saved metrics to {output_dir}/all_methods_metrics.csv")

    # Generate plots
    generate_plots(results_df, output_dir)


def generate_plots(results_df: pd.DataFrame, output_dir: str):
    """Generate comparison plots with AUDC and QNC."""
    # Separate R2-Router and CARROT results
    r2_results = results_df[results_df['method'].str.contains('R2-Router')].sort_values('n_heads')
    carrot_results = results_df[results_df['method'].str.contains('CARROT')]

    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: AUDC
    ax1.plot(r2_results['n_heads'], r2_results['audc'],
             marker='o', linewidth=2.5, markersize=8,
             color='#2E86AB', label='R2-Router (Interpolation)', zorder=3)

    # R2-Router Full (16 heads)
    core_16 = r2_results[r2_results['n_heads'] == 16]
    if not core_16.empty:
        ax1.axhline(y=core_16['audc'].values[0], color='red', linestyle='--', linewidth=2,
                    label=f'R2-Router Full (16 heads): {core_16["audc"].values[0]:.4f}', zorder=2)

    # CARROT baselines
    for _, row in carrot_results.iterrows():
        if 'KNN' in row['method']:
            ax1.axhline(y=row['audc'], color='#E63946', linestyle=':', linewidth=2.5,
                        label=f"CARROT-KNN: {row['audc']:.4f}", zorder=1)
        elif 'Linear' in row['method']:
            ax1.axhline(y=row['audc'], color='#A8DADC', linestyle=':', linewidth=2.5,
                        label=f"CARROT-Linear: {row['audc']:.4f}", zorder=1)

    ax1.set_xlabel('Number of Trained Heads', fontsize=13, fontweight='bold')
    ax1.set_ylabel('AUDC (Normalized) ↑', fontsize=13, fontweight='bold')
    ax1.set_title('Routing Performance (AUDC)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=10, framealpha=0.95, loc='best')
    ax1.set_xticks(r2_results['n_heads'])

    # Plot 2: QNC (lower is better)
    ax2.plot(r2_results['n_heads'], r2_results['qnc'],
             marker='o', linewidth=2.5, markersize=8,
             color='#2E86AB', label='R2-Router (Interpolation)', zorder=3)

    # R2-Router Full (16 heads)
    if not core_16.empty:
        ax2.axhline(y=core_16['qnc'].values[0], color='red', linestyle='--', linewidth=2,
                    label=f'R2-Router Full (16 heads): {core_16["qnc"].values[0]:.4f}', zorder=2)

    # CARROT baselines
    for _, row in carrot_results.iterrows():
        if 'KNN' in row['method']:
            ax2.axhline(y=row['qnc'], color='#E63946', linestyle=':', linewidth=2.5,
                        label=f"CARROT-KNN: {row['qnc']:.4f}", zorder=1)
        elif 'Linear' in row['method']:
            ax2.axhline(y=row['qnc'], color='#A8DADC', linestyle=':', linewidth=2.5,
                        label=f"CARROT-Linear: {row['qnc']:.4f}", zorder=1)

    ax2.set_xlabel('Number of Trained Heads', fontsize=13, fontweight='bold')
    ax2.set_ylabel('QNC (Normalized) ↓', fontsize=13, fontweight='bold')
    ax2.set_title('Cost Efficiency (QNC)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=10, framealpha=0.95, loc='best')
    ax2.set_xticks(r2_results['n_heads'])

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'audc_and_qnc_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")

    output_path_pdf = os.path.join(output_dir, 'audc_and_qnc_comparison.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved PDF to {output_path_pdf}")

    plt.close()


if __name__ == "__main__":
    main()
