#!/usr/bin/env python3
"""
Train R2-Router predictors and evaluate performance for different embedding models.

This script:
1. Loads embeddings for each embedding model
2. Trains R2-Router predictors (sklearn-based, fast)
3. Evaluates AUDC, QNC, Peak Accuracy
4. Saves results for comparison
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from main.shared.dataset_manager import DatasetManager
from main.r2.predictor_sklearn import TokenPerformancePredictor, route_scores


def compute_audc(costs: np.ndarray, perfs: np.ndarray) -> float:
    """Compute Area Under Deferral Curve with normalized cost."""
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


def compute_qnc(costs: np.ndarray, perfs: np.ndarray,
                target_accuracy: float,
                global_min_cost: float,
                global_max_cost: float) -> float:
    """Compute Query-Normalized Cost with global normalization."""
    valid_idx = perfs >= target_accuracy

    if not valid_idx.any():
        return 1.0

    min_cost_to_reach_target = costs[valid_idx].min()

    if global_max_cost > global_min_cost:
        qnc = (min_cost_to_reach_target - global_min_cost) / (global_max_cost - global_min_cost)
    else:
        qnc = 0.0

    return np.clip(qnc, 0, 1)


def train_and_evaluate_embedding(embedding_key: str,
                                 embedding_path: str,
                                 models: list,
                                 lambda_range: np.ndarray,
                                 dataset_manager: DatasetManager,
                                 token_limits_score: list,
                                 token_limits_count: list) -> dict:
    """
    Train R2-Router predictors and evaluate for one embedding model.

    Args:
        embedding_key: Key identifier for embedding model
        embedding_path: Path to embedding pickle file
        models: List of (model_name, model_size, csv_path) tuples
        lambda_range: Array of lambda values
        dataset_manager: DatasetManager with train/test split
        token_limits_score: List of score column names
        token_limits_count: List of count column names

    Returns:
        Dictionary with results
    """
    print("\n" + "=" * 80)
    print(f"Training and Evaluating: {embedding_key}")
    print("=" * 80)

    # Load embeddings
    print(f"Loading embeddings from {embedding_path}...")
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Embeddings shape: {embeddings.shape}")

    # Create custom DatasetManager with these embeddings
    train_idx, test_idx = dataset_manager.get_split_indices()

    # Train and load LLMs
    from main.shared.llm_loader import load_llm
    from main.shared.router_dataset import RouterDataset
    llms = {}
    checkpoint_dir = f'ablation/checkpoints/{embedding_key}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for model_name, model_size, csv_path in tqdm(models, desc=f"Training models for {embedding_key}"):
        model_checkpoint = os.path.join(checkpoint_dir, model_name)
        os.makedirs(model_checkpoint, exist_ok=True)

        # Check if checkpoints exist
        limited_ckpt = os.path.join(model_checkpoint, "limited_score_predictors.joblib")
        unlimited_score_ckpt = os.path.join(model_checkpoint, "unlimited_score_predictor.joblib")
        unlimited_token_ckpt = os.path.join(model_checkpoint, "unlimited_token_predictor.joblib")

        checkpoints_exist = (
            os.path.exists(limited_ckpt) and
            os.path.exists(unlimited_score_ckpt) and
            os.path.exists(unlimited_token_ckpt)
        )

        if not checkpoints_exist:
            print(f"\n  Training {model_name} with {embedding_key} embeddings...")

            # Create dataset
            dataset = RouterDataset(
                embeddings=embeddings,
                score_df_path=csv_path,
                target_tokens_score=token_limits_score,
                train_idx=train_idx,
                test_idx=test_idx
            )

            # Create and train predictor (use Ridge for numerical stability)
            predictor = TokenPerformancePredictor(
                token_limits=token_limits_score,
                model_type="ridge",  # Ridge regression for numerical stability
                alpha=0.01  # Small regularization to avoid SVD convergence issues
            )

            # Prepare training data
            train_data = dataset.get_train_set_score()
            test_data = dataset.get_test_set_score()

            X_train = train_data['X']
            y_train = train_data['y']
            X_test = test_data['X']
            y_test = test_data['y']

            # Extract limited and unlimited scores
            # y_train/y_test are dicts mapping token_name -> numpy array
            limited_token_names = [t for t in token_limits_score if t != 'unlimited_score']
            quality_train_limited = {t: y_train[t] for t in limited_token_names}
            quality_test_limited = {t: y_test[t] for t in limited_token_names}
            quality_train_unlimited = y_train['unlimited_score']
            quality_test_unlimited = y_test['unlimited_score']

            # Token counts for unlimited setting
            token_count_train = dataset.get_train_token_unlimited_count()
            token_count_test = dataset.get_test_token_unlimited_count()

            # Train
            predictor.fit(
                embedding_train=X_train,
                quality_train_limited=quality_train_limited,
                quality_train_unlimited=quality_train_unlimited,
                token_count_train=token_count_train,
                embedding_test=X_test,
                quality_test_limited=quality_test_limited,
                quality_test_unlimited=quality_test_unlimited,
                token_count_test=token_count_test,
                save_dir=model_checkpoint,
                plot_dir=None  # Skip plots for ablation
            )
            print(f"  ✓ Trained and saved {model_name}")
        else:
            print(f"\n  Checkpoints exist for {model_name}, loading...")

        # Now load the LLM
        llms[model_name] = load_llm(
            name=model_name,
            size=model_size,
            score_df_path=csv_path,
            load_dir=model_checkpoint,
            embeddings=embeddings,
            train_idx=train_idx,
            test_idx=test_idx,
            token_limits_score=token_limits_score,
            token_limits_count=token_limits_count,
            predictor_class=TokenPerformancePredictor
        )

    # Evaluate routing
    print(f"Evaluating routing for {embedding_key}...")
    costs, perfs = route_scores(llms, lambda_range)

    # Compute metrics
    audc = compute_audc(costs, perfs)
    peak_acc = perfs.max()

    print(f"{embedding_key}: AUDC={audc:.4f}, Peak Accuracy={peak_acc:.4f}")

    return {
        'embedding_key': embedding_key,
        'costs': costs,
        'perfs': perfs,
        'audc': audc,
        'peak_accuracy': peak_acc
    }


def main():
    """Main evaluation pipeline."""
    print("=" * 80)
    print("Embedding Model Ablation Study - Training and Evaluation")
    print("=" * 80)

    # Initialize dataset manager (for train/test split)
    dataset_manager = DatasetManager(
        embeddings_path='data/prompt_embeddings.pkl',  # Just for split, not actual embeddings
        train_ratio=0.8,
        seed=42
    )

    # Models to train (subset for faster ablation, excluding Qwen3-0.6B due to numerical issues)
    models = [
        ('GLM-4.5-Air', 0.85, 'data/GLM-4.5-Air.csv'),
        ('Llama-3.2-3B-Instruct', 0.02, 'data/Llama-3.2-3B-Instruct.csv'),
        # ('Qwen3-0.6B', 0.0173, 'data/Qwen3-0.6B.csv'),  # Skipped - causes NaN errors
    ]

    # Token limits
    token_limits_score = [f'{t}_score' for t in [10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000]]
    token_limits_score.append('unlimited_score')
    token_limits_count = [t.replace('_score', '_count') for t in token_limits_score]

    # Lambda range
    lambda_range = np.linspace(0, 1e-4, 100)

    # Find embedding files (skip roberta and current_qwen3 due to numerical issues)
    embedding_dir = 'ablation/embeddings'
    all_embedding_files = sorted([f for f in os.listdir(embedding_dir) if f.endswith('_embeddings.pkl')])
    # Only use all_minilm_l6 and all_mpnet_base (both work reliably)
    embedding_files = [f for f in all_embedding_files if any(name in f for name in ['minilm', 'mpnet'])]
    print(f"\nFound {len(embedding_files)} embedding files:")
    for f in embedding_files:
        print(f"  - {f}")

    # Train and evaluate each embedding model
    all_results = []
    all_costs_list = []
    all_perfs_list = []

    for emb_file in embedding_files:
        embedding_key = emb_file.replace('_embeddings.pkl', '')
        embedding_path = os.path.join(embedding_dir, emb_file)

        result = train_and_evaluate_embedding(
            embedding_key=embedding_key,
            embedding_path=embedding_path,
            models=models,
            lambda_range=lambda_range,
            dataset_manager=dataset_manager,
            token_limits_score=token_limits_score,
            token_limits_count=token_limits_count
        )

        all_results.append(result)
        all_costs_list.extend(result['costs'])
        all_perfs_list.extend(result['perfs'])

    # Determine target accuracy for QNC (95% of best single LLM)
    print("\n" + "=" * 80)
    print("Computing QNC with global normalization...")
    print("=" * 80)

    # Use first embedding to find target accuracy
    from main.shared.router_dataset import RouterDataset
    first_embedding_path = os.path.join(embedding_dir, embedding_files[0])
    with open(first_embedding_path, 'rb') as f:
        embeddings = pickle.load(f)

    best_llm_acc = 0.0
    train_idx, test_idx = dataset_manager.get_split_indices()
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

    target_accuracy = best_llm_acc * 0.95
    print(f"Best LLM: {best_llm} with accuracy {best_llm_acc:.4f}")
    print(f"Target accuracy (95%): {target_accuracy:.4f}")

    # Global normalization
    global_min_cost = np.min(all_costs_list)
    global_max_cost = np.max(all_costs_list)
    print(f"Global cost range: [{global_min_cost:.2f}, {global_max_cost:.2f}]")

    # Compute QNC for each embedding
    for result in all_results:
        qnc = compute_qnc(result['costs'], result['perfs'], target_accuracy,
                         global_min_cost, global_max_cost)
        result['qnc'] = qnc

    # Save results
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summary_data = []
    for result in all_results:
        print(f"{result['embedding_key']:25s}: AUDC={result['audc']:.4f}, QNC={result['qnc']:.4f}, Peak={result['peak_accuracy']:.4f}")
        summary_data.append({
            'embedding_model': result['embedding_key'],
            'audc': result['audc'],
            'qnc': result['qnc'],
            'peak_accuracy': result['peak_accuracy']
        })

    # Save summary
    summary_df = pd.DataFrame(summary_data)
    output_dir = 'ablation/results'
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(output_dir, 'embedding_comparison.csv'), index=False)
    print(f"\n✓ Saved summary to ablation/results/embedding_comparison.csv")

    # Save detailed curves
    curves_data = []
    for result in all_results:
        for i, (cost, perf) in enumerate(zip(result['costs'], result['perfs'])):
            curves_data.append({
                'embedding_model': result['embedding_key'],
                'lambda_idx': i,
                'cost': cost,
                'performance': perf
            })
    curves_df = pd.DataFrame(curves_data)
    curves_df.to_csv(os.path.join(output_dir, 'embedding_curves.csv'), index=False)
    print(f"✓ Saved curves to ablation/results/embedding_curves.csv")


if __name__ == "__main__":
    main()
