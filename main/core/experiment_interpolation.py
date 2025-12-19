#!/usr/bin/env python3
"""
Experiment: Training CoRE with subset of token limit heads and interpolating others.

This experiment investigates the trade-off between training efficiency and routing performance:
- Train predictors for a subset of token limits (e.g., 2, 4, 6, ..., 16 heads)
- Interpolate predictions for untrained token limits
- Evaluate routing performance (AUDC) vs. number of trained heads

Token limits (16 total):
  Limited: 10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000
  Unlimited: unlimited (always trained)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from main.shared.dataset_manager import DatasetManager
from main.shared.router_dataset import RouterDataset
from main.core.predictor_sklearn import TokenPerformancePredictor, route_scores


# Token limits (numeric values for interpolation)
TOKEN_LIMITS_NUMERIC = [10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000]
TOKEN_LIMITS_NAMES = [f'{t}_score' for t in TOKEN_LIMITS_NUMERIC]


def select_training_heads(n_heads: int, strategy: str = 'uniform') -> List[str]:
    """
    Select which token limit heads to train.

    Args:
        n_heads: Number of heads to train (2 to 15)
        strategy: Selection strategy
            - 'uniform': Evenly spaced across log-scale token limits
            - 'extremes': Include extremes (min and max)

    Returns:
        List of token limit names to train (e.g., ['10_score', '100_score', ...])
    """
    if n_heads < 2 or n_heads > 15:
        raise ValueError(f"n_heads must be between 2 and 15, got {n_heads}")

    if n_heads == 15:
        # Train all limited token limits
        return TOKEN_LIMITS_NAMES.copy()

    if strategy == 'uniform':
        # Select evenly spaced indices
        indices = np.linspace(0, len(TOKEN_LIMITS_NUMERIC) - 1, n_heads).astype(int)
        selected = [TOKEN_LIMITS_NAMES[i] for i in indices]
    elif strategy == 'extremes':
        # Always include min and max
        if n_heads == 2:
            selected = [TOKEN_LIMITS_NAMES[0], TOKEN_LIMITS_NAMES[-1]]
        else:
            # Add middle points uniformly
            middle_n = n_heads - 2
            middle_indices = np.linspace(1, len(TOKEN_LIMITS_NUMERIC) - 2, middle_n).astype(int)
            selected = [TOKEN_LIMITS_NAMES[0]] + [TOKEN_LIMITS_NAMES[i] for i in middle_indices] + [TOKEN_LIMITS_NAMES[-1]]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return selected


def interpolate_predictions(trained_heads: List[str],
                            trained_predictions: Dict[str, np.ndarray],
                            all_heads: List[str]) -> Dict[str, np.ndarray]:
    """
    Interpolate predictions for untrained token limit heads.

    Args:
        trained_heads: List of trained token limit names (e.g., ['10_score', '100_score'])
        trained_predictions: Dict mapping trained heads to predictions (n_queries, )
        all_heads: List of all token limit names to predict for

    Returns:
        Dict mapping all heads to predictions (interpolated where needed)
    """
    # Extract numeric token limits for trained heads
    trained_limits = [int(h.split('_')[0]) for h in trained_heads]

    # Sort by token limit
    sorted_indices = np.argsort(trained_limits)
    trained_limits_sorted = [trained_limits[i] for i in sorted_indices]
    trained_heads_sorted = [trained_heads[i] for i in sorted_indices]

    # Get predictions for each query
    n_queries = trained_predictions[trained_heads[0]].shape[0]
    all_predictions = {}

    for head in all_heads:
        if head in trained_heads:
            # Use trained predictions
            all_predictions[head] = trained_predictions[head]
        else:
            # Interpolate
            target_limit = int(head.split('_')[0])

            # Create interpolated predictions for all queries
            interpolated = np.zeros(n_queries)

            for q in range(n_queries):
                # Get trained predictions for this query
                trained_values = [trained_predictions[h][q] for h in trained_heads_sorted]

                # Linear interpolation (extrapolate if needed)
                interp_func = interp1d(trained_limits_sorted, trained_values,
                                      kind='linear', fill_value='extrapolate')
                interpolated[q] = np.clip(interp_func(target_limit), 0, 1)

            all_predictions[head] = interpolated

    return all_predictions


def train_with_subset_heads(dataset_manager: DatasetManager,
                            model_name: str,
                            model_size: float,
                            csv_path: str,
                            trained_heads: List[str]) -> Dict:
    """
    Train CoRE predictor with only a subset of token limit heads.

    Args:
        dataset_manager: DatasetManager instance
        model_name: LLM model name
        model_size: Model size in billions
        csv_path: Path to CSV data file
        trained_heads: List of token limit names to train

    Returns:
        Dictionary with predictions for all heads (trained + interpolated)
    """
    # Load data
    embeddings = dataset_manager.get_embeddings()
    train_idx, test_idx = dataset_manager.get_split_indices()

    # Load RouterDataset
    token_limits_score = TOKEN_LIMITS_NAMES + ['unlimited_score']
    token_limits_count = [t.replace('_score', '_count') for t in token_limits_score]

    dataset = RouterDataset(
        embeddings=embeddings,
        score_df_path=csv_path,
        target_tokens_score=token_limits_score,
        train_idx=train_idx,
        test_idx=test_idx
    )

    # Get embeddings
    embedding_train = dataset.get_train_set_score()["X"]
    embedding_test = dataset.get_test_set_score()["X"]

    # Train limited budget predictors (only selected heads)
    quality_train_limited = {}
    for head in trained_heads:
        quality_train_limited[head] = dataset.get_train_set_score()["y"][head]

    # Always train unlimited predictor
    quality_train_unlimited = dataset.get_train_set_score()["y"]['unlimited_score']
    token_count_train = dataset.get_train_token_unlimited_count()

    # Initialize and train predictor
    predictor = TokenPerformancePredictor(
        token_limits=trained_heads + ['unlimited_score'],
        model_type='linear'
    )

    predictor.fit(
        embedding_train=embedding_train,
        quality_train_limited=quality_train_limited,
        quality_train_unlimited=quality_train_unlimited,
        token_count_train=token_count_train
    )

    # Get predictions for trained heads
    trained_predictions_test = {}
    for head in trained_heads:
        trained_predictions_test[head] = predictor.limited_score_predictors[head].predict(embedding_test)

    # Interpolate for all heads
    all_predictions_test = interpolate_predictions(
        trained_heads=trained_heads,
        trained_predictions=trained_predictions_test,
        all_heads=TOKEN_LIMITS_NAMES
    )

    # Add unlimited predictions
    all_predictions_test['unlimited_score'] = predictor.unlimited_score_predictor.predict(embedding_test)

    # Prepare pred_test_score
    pred_test_score = []
    for head in token_limits_score:
        pred_test_score.append(all_predictions_test[head])
    pred_test_score = np.column_stack(pred_test_score)

    # Predict token counts: use limits for limited budgets, predicted count for unlimited
    pred_test_unlimited_count = predictor.unlimited_token_predictor.predict(embedding_test)
    token_limits_values = np.array([int(s.replace('_score', '')) for s in TOKEN_LIMITS_NAMES])
    pred_test_token_limited = np.tile(token_limits_values[None, :], (len(test_idx), 1))
    pred_test_token = np.column_stack((pred_test_token_limited, pred_test_unlimited_count))

    # Get true scores and counts from dataset
    df = pd.read_csv(csv_path)
    gt_test_df = df.loc[test_idx, token_limits_score + token_limits_count].reset_index(drop=True)
    true_test_score = np.array(gt_test_df.loc[:, token_limits_score])
    true_test_count = np.array(gt_test_df.loc[:, token_limits_count])

    return {
        'size': model_size,
        'pred_test_score': pred_test_score,
        'pred_test_token': pred_test_token,
        'true_test_score': true_test_score,
        'true_test_count': true_test_count
    }


def compute_audc(costs: np.ndarray, perfs: np.ndarray) -> float:
    """
    Compute Area Under Deferral Curve (AUDC) with normalized cost.

    Args:
        costs: Array of cost values
        perfs: Array of performance values

    Returns:
        AUDC (normalized)
    """
    # Sort by cost
    sorted_idx = np.argsort(costs)
    sorted_cost = costs[sorted_idx]
    sorted_perf = perfs[sorted_idx]

    # Normalize cost to [0, 1]
    min_cost, max_cost = sorted_cost.min(), sorted_cost.max()
    if max_cost > min_cost:
        norm_cost = (sorted_cost - min_cost) / (max_cost - min_cost)
    else:
        norm_cost = np.zeros_like(sorted_cost)

    # Compute AUDC
    audc = np.trapz(sorted_perf, norm_cost)
    return audc


def run_experiment(models: List[Tuple[str, float, str]],
                  n_heads_list: List[int],
                  lambda_range: np.ndarray,
                  strategy: str = 'uniform') -> pd.DataFrame:
    """
    Run interpolation experiment for different numbers of trained heads.

    Args:
        models: List of (model_name, model_size, csv_path) tuples
        n_heads_list: List of head counts to test (e.g., [2, 4, 6, 8, 10, 12, 14, 16])
        lambda_range: Array of lambda values for routing
        strategy: Head selection strategy

    Returns:
        DataFrame with results
    """
    # Initialize dataset manager
    dataset_manager = DatasetManager(
        embeddings_path='data/prompt_embeddings.pkl',
        train_ratio=0.8,
        seed=42
    )

    results = []

    for n_heads in tqdm(n_heads_list, desc="Testing different head counts"):
        # Select which heads to train
        if n_heads == 16:
            # Train all heads (15 limited + 1 unlimited)
            trained_heads = TOKEN_LIMITS_NAMES.copy()
        else:
            trained_heads = select_training_heads(n_heads, strategy=strategy)

        print(f"\nTraining {n_heads} heads: {trained_heads}")

        # Train models
        llms = {}
        for model_name, model_size, csv_path in models:
            llms[model_name] = train_with_subset_heads(
                dataset_manager=dataset_manager,
                model_name=model_name,
                model_size=model_size,
                csv_path=csv_path,
                trained_heads=trained_heads
            )

        # Evaluate routing performance
        costs, perfs = route_scores(llms, lambda_range)
        audc = compute_audc(costs, perfs)
        peak_accuracy = perfs.max()

        results.append({
            'n_heads': n_heads,
            'trained_heads': ','.join(trained_heads),
            'audc': audc,
            'peak_accuracy': peak_accuracy
        })

        print(f"  AUDC: {audc:.4f}, Peak Accuracy: {peak_accuracy:.4f}")

    return pd.DataFrame(results)


def plot_results(results_df: pd.DataFrame, output_dir: str):
    """
    Plot AUDC vs. number of trained heads.

    Args:
        results_df: Results DataFrame
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.4)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot AUDC vs n_heads
    ax.plot(results_df['n_heads'], results_df['audc'],
            marker='o', linewidth=2.5, markersize=8,
            color='#2E86AB', label='CoRE (Interpolation)', zorder=3)

    # Highlight full model (16 heads)
    full_model = results_df[results_df['n_heads'] == 16]
    if not full_model.empty:
        ax.axhline(y=full_model['audc'].values[0],
                   color='red', linestyle='--', linewidth=2,
                   label=f'CoRE Full (16 heads): {full_model["audc"].values[0]:.4f}',
                   zorder=2)

    # Load and plot CARROT baselines
    carrot_file = os.path.join(output_dir, 'carrot_baselines.txt')
    if os.path.exists(carrot_file):
        with open(carrot_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    method, audc, peak = parts
                    audc_val = float(audc)
                    if 'KNN' in method:
                        ax.axhline(y=audc_val, color='#E63946', linestyle=':', linewidth=2.5,
                                   label=f'CARROT-KNN: {audc_val:.4f}', zorder=1)
                    elif 'Linear' in method:
                        ax.axhline(y=audc_val, color='#A8DADC', linestyle=':', linewidth=2.5,
                                   label=f'CARROT-Linear: {audc_val:.4f}', zorder=1)

    # Labels and title
    ax.set_xlabel('Number of Trained Heads', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUDC (Normalized) ↑', fontsize=14, fontweight='bold')
    ax.set_title('Routing Performance vs. Training Efficiency\n(CoRE with Interpolation vs. CARROT)',
                fontsize=16, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    ax.legend(fontsize=11, framealpha=0.95, loc='best')

    # Set x-axis ticks
    ax.set_xticks(results_df['n_heads'])

    # Tight layout
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'audc_vs_nheads.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_path}")

    # Save PDF
    output_path_pdf = os.path.join(output_dir, 'audc_vs_nheads.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved PDF to {output_path_pdf}")

    plt.close()


def main():
    """Run interpolation experiment."""
    print("=" * 80)
    print("CoRE Interpolation Experiment")
    print("=" * 80)

    # Models to test - using same pool as main experiment
    models = [
        ('Mistral-7B-Instruct-v0.2', 0.20, 'data/Mistral-7B-Instruct-v0.2.csv'),
        ('GLM-4.5-Air', 0.85, 'data/GLM-4.5-Air.csv'),
        ('gemma-3-4b-it', 0.06815, 'data/gemma-3-4b-it.csv'),
        ('Llama-3.1-70B-Instruct', 0.40, 'data/Llama-3.1-70B-Instruct.csv'),
        ('Llama-3.2-3B-Instruct', 0.02, 'data/Llama-3.2-3B-Instruct.csv'),  # FIXED: was 3.0
        ('Qwen2.5-Math-7B-Instruct', 0.35, 'data/Qwen2.5-Math-7B-Instruct.csv'),
        ('Qwen3-0.6B', 0.0173, 'data/Qwen3-0.6B.csv'),
    ]

    # Head counts to test
    n_heads_list = [2, 4, 6, 8, 10, 12, 14, 16]

    # Lambda range for routing
    lambda_range = np.linspace(0, 1e-4, 100)

    # Run experiment
    print(f"\nTesting {len(models)} models with {len(n_heads_list)} different head counts")
    results_df = run_experiment(
        models=models,
        n_heads_list=n_heads_list,
        lambda_range=lambda_range,
        strategy='uniform'
    )

    # Save results
    output_dir = 'comparison_results/interpolation_experiment'
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, 'interpolation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved results to {results_path}")

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Plot results
    plot_results(results_df, output_dir)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
