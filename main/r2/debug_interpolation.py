#!/usr/bin/env python3
"""
Debug script to verify interpolation experiment results.
Check if interpolated predictions are actually worse than trained ones.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

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


def evaluate_prediction_quality(csv_path: str, n_heads: int = 2):
    """
    Evaluate prediction quality for different numbers of trained heads.

    Compare:
    1. Trained head predictions vs ground truth
    2. Interpolated head predictions vs ground truth
    """
    print(f"\n{'='*80}")
    print(f"Evaluating with {n_heads} trained heads")
    print(f"{'='*80}")

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

    # Get embeddings
    embedding_test = dataset.get_test_set_score()["X"]

    # Select heads to train
    trained_heads = select_training_heads(n_heads, strategy='uniform')
    print(f"Trained heads: {trained_heads}")

    # Train predictor with subset
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

    # Get ground truth
    df = pd.read_csv(csv_path)
    gt_test_df = df.loc[test_idx, token_limits_score].reset_index(drop=True)

    # Evaluate each head
    print(f"\n{'Head':<15} {'Type':<15} {'MSE':<10} {'MAE':<10} {'R2':<10}")
    print('-' * 65)

    mse_trained = []
    mse_interpolated = []

    for head in TOKEN_LIMITS_NAMES:
        pred = all_predictions_test[head]
        true = gt_test_df[head].values

        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)

        # Calculate R2
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        head_type = "TRAINED" if head in trained_heads else "INTERP"

        if head_type == "TRAINED":
            mse_trained.append(mse)
        else:
            mse_interpolated.append(mse)

        print(f"{head:<15} {head_type:<15} {mse:<10.6f} {mae:<10.6f} {r2:<10.6f}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Avg MSE for TRAINED heads:      {np.mean(mse_trained):.6f}")
    print(f"Avg MSE for INTERPOLATED heads: {np.mean(mse_interpolated):.6f}")
    print(f"Difference: {np.mean(mse_interpolated) - np.mean(mse_trained):.6f}")

    if np.mean(mse_interpolated) < np.mean(mse_trained):
        print("\n⚠️  WARNING: Interpolated predictions have LOWER error than trained!")
        print("   This suggests a potential bug in the code.")
    else:
        print("\n✓ Expected: Interpolated predictions have higher error than trained.")

    return {
        'n_heads': n_heads,
        'mse_trained': np.mean(mse_trained),
        'mse_interpolated': np.mean(mse_interpolated)
    }


def main():
    """Run debug analysis."""
    csv_path = 'data/GLM-4.5-Air.csv'

    results = []
    for n_heads in [2, 4, 8, 16]:
        result = evaluate_prediction_quality(csv_path, n_heads)
        results.append(result)

    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
