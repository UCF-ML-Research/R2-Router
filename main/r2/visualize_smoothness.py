#!/usr/bin/env python3
"""
Visualize smoothness of quality-token curves for different numbers of trained heads.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from main.shared.dataset_manager import DatasetManager
from main.shared.router_dataset import RouterDataset
from main.r2.predictor_sklearn import TokenPerformancePredictor
from main.r2.experiment_interpolation import (
    TOKEN_LIMITS_NAMES,
    TOKEN_LIMITS_NUMERIC,
    interpolate_predictions,
    select_training_heads
)


def visualize_quality_curves(csv_path: str, query_idx: int = 0):
    """
    Visualize quality-token curves for a specific query.
    """
    # Initialize dataset manager
    dataset_manager = DatasetManager(
        embeddings_path='data/prompt_embeddings.pkl',
        train_ratio=0.8,
        seed=42
    )

    embeddings = dataset_manager.get_embeddings()
    train_idx, test_idx = dataset_manager.get_split_indices()

    # Load data
    token_limits_score = TOKEN_LIMITS_NAMES + ['unlimited_score']
    dataset = RouterDataset(
        embeddings=embeddings,
        score_df_path=csv_path,
        target_tokens_score=token_limits_score,
        train_idx=train_idx,
        test_idx=test_idx
    )

    embedding_test = dataset.get_test_set_score()["X"]

    # Test different numbers of heads
    n_heads_list = [2, 4, 8, 15]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, n_heads in zip(axes, n_heads_list):
        # Select heads to train
        if n_heads == 15:
            trained_heads = TOKEN_LIMITS_NAMES.copy()
        else:
            trained_heads = select_training_heads(n_heads, strategy='uniform')

        # Train predictor
        quality_train_limited = {}
        for head in trained_heads:
            quality_train_limited[head] = dataset.get_train_set_score()["y"][head]

        quality_train_unlimited = dataset.get_train_set_score()["y"]['unlimited_score']
        token_count_train = dataset.get_train_token_unlimited_count()

        predictor = TokenPerformancePredictor(
            token_limits=trained_heads + ['unlimited_score'],
            model_type='linear'
        )

        predictor.fit(
            embedding_train=dataset.get_train_set_score()["X"],
            quality_train_limited=quality_train_limited,
            quality_train_unlimited=quality_train_unlimited,
            token_count_train=token_count_train
        )

        # Get predictions
        trained_predictions_test = {}
        for head in trained_heads:
            trained_predictions_test[head] = predictor.limited_score_predictors[head].predict(embedding_test)

        all_predictions_test = interpolate_predictions(
            trained_heads=trained_heads,
            trained_predictions=trained_predictions_test,
            all_heads=TOKEN_LIMITS_NAMES
        )

        # Extract for this query
        pred_curve = [all_predictions_test[head][query_idx] for head in TOKEN_LIMITS_NAMES]

        # Ground truth
        import pandas as pd
        df = pd.read_csv(csv_path)
        true_curve = df.loc[test_idx[query_idx], TOKEN_LIMITS_NAMES].values

        # Plot
        ax.plot(TOKEN_LIMITS_NUMERIC, true_curve, 'o-', label='Ground Truth',
                linewidth=2.5, markersize=8, color='black', alpha=0.7)
        ax.plot(TOKEN_LIMITS_NUMERIC, pred_curve, 's--', label='Predicted',
                linewidth=2.5, markersize=8, color='red', alpha=0.7)

        # Mark trained heads
        trained_indices = [TOKEN_LIMITS_NAMES.index(h) for h in trained_heads]
        trained_tokens = [TOKEN_LIMITS_NUMERIC[i] for i in trained_indices]
        trained_preds = [pred_curve[i] for i in trained_indices]
        ax.scatter(trained_tokens, trained_preds, s=200, c='green',
                  marker='*', zorder=5, label='Trained Heads', edgecolors='black', linewidths=2)

        # Calculate smoothness (second derivative)
        second_deriv = np.abs(np.diff(np.diff(pred_curve)))
        smoothness = np.mean(second_deriv)

        ax.set_title(f'{n_heads} Trained Heads\nSmoothness: {smoothness:.4f}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Token Limit', fontsize=11)
        ax.set_ylabel('Quality Score', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.suptitle(f'Quality-Token Curves for Query #{query_idx}\n' +
                 '(Lower smoothness = more jagged curve)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    output_dir = Path('comparison_results/interpolation_experiment')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'smoothness_visualization_query_{query_idx}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    visualize_quality_curves('data/GLM-4.5-Air.csv', query_idx=100)
