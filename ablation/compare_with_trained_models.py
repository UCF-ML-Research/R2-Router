#!/usr/bin/env python3
"""
Compare CoRE vs CARROT-Linear vs MIRT using all-MiniLM-L6-v2 embeddings.
Uses only successfully trained models to avoid encoding issues.
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
from main.core.predictor_sklearn import TokenPerformancePredictor, route_scores
from main.baselines.carrot.baselines_carrot import CarrotLinearBaseline
from main.baselines.irt.baselines_irt import IRTBaseline
from sentence_transformers import SentenceTransformer


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


def route_baseline(Y_hat_score, Y_hat_count, quality_test, token_count_test, lambda_range, sizes_vec):
    """Route using baseline predictions."""
    costs_list = []
    perfs_list = []

    for lam in lambda_range:
        # Compute risk for each (model, token_limit)
        predicted_cost = Y_hat_count * sizes_vec  # (n_test, n_options)
        risk = (1 - lam) * Y_hat_score - lam * predicted_cost

        # Select best option for each query
        best_idx = np.argmax(risk, axis=1)

        # Get actual performance and cost
        n_test = quality_test.shape[0]
        selected_quality = quality_test[np.arange(n_test), best_idx]
        selected_count = token_count_test[np.arange(n_test), best_idx]
        selected_size = sizes_vec[best_idx]
        selected_cost = selected_count * selected_size

        # Average across test set
        costs_list.append(selected_cost.mean())
        perfs_list.append(selected_quality.mean())

    return np.array(costs_list), np.array(perfs_list)


def main():
    """Main comparison pipeline."""
    print("=" * 80)
    print("Embedding Ablation: CoRE vs CARROT-Linear vs MIRT")
    print("Using all-MiniLM-L6-v2 embeddings with successfully trained models")
    print("=" * 80)

    # Use all-MiniLM-L6-v2 embedding
    embedding_key = 'all_minilm_l6'
    embedding_path = 'ablation/embeddings/all_minilm_l6_embeddings.pkl'

    # Initialize dataset manager
    dataset_manager = DatasetManager(
        embeddings_path='data/prompt_embeddings.pkl',
        train_ratio=0.8,
        seed=42
    )

    # Load embeddings
    print(f"\nLoading embeddings from {embedding_path}...")
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Embeddings shape: {embeddings.shape}")

    train_idx, test_idx = dataset_manager.get_split_indices()
    embedding_train = embeddings[train_idx]
    embedding_test = embeddings[test_idx]

    # Use only successfully trained models without encoding issues (2 models to avoid segfault)
    models = [
        ('Mistral-7B-Instruct-v0.2', 0.20, 'data/Mistral-7B-Instruct-v0.2.csv'),
        ('GLM-4.5-Air', 0.85, 'data/GLM-4.5-Air.csv'),
        # ('gemma-3-4b-it', 0.06815, 'data/gemma-3-4b-it.csv'),
        # ('gemma-3-1b-it', 0.0170375, 'data/gemma-3-1b-it.csv'),
        # ('gemma-3-270m-it', 0.004259375, 'data/gemma-3-270m-it.csv'),
    ]

    # Token limits
    token_limits_score = [f'{t}_score' for t in [10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000]]
    token_limits_score.append('unlimited_score')
    token_limits_count = [t.replace('_score', '_count') for t in token_limits_score]

    # Lambda range for routing
    lambda_range = np.linspace(0, 1e-4, 100)

    # Load data and build matrices for baselines
    print("\n" + "=" * 80)
    print("Loading Model Data")
    print("=" * 80)

    quality_train_list = []
    token_count_train_list = []
    quality_test_list = []
    token_count_test_list = []
    sizes_list = []
    option_names = []

    for model_name, model_size, csv_path in tqdm(models, desc="Loading models"):
        # Use latin-1 encoding directly to avoid hanging on encoding errors
        df = pd.read_csv(csv_path, encoding='latin-1', encoding_errors='replace')

        # For each token limit, add as a separate option
        for limit_score, limit_count in zip(token_limits_score, token_limits_count):
            quality_train_list.append(df.loc[train_idx, limit_score].values)
            token_count_train_list.append(df.loc[train_idx, limit_count].values)
            quality_test_list.append(df.loc[test_idx, limit_score].values)
            token_count_test_list.append(df.loc[test_idx, limit_count].values)
            sizes_list.append(model_size)
            option_names.append(f"{model_name}_{limit_score}")

    # Stack into matrices (n_samples, n_options)
    quality_train = np.column_stack(quality_train_list)
    token_count_train = np.column_stack(token_count_train_list)
    quality_test = np.column_stack(quality_test_list)
    token_count_test = np.column_stack(token_count_test_list)
    sizes_vec = np.array(sizes_list)

    print(f"Training data shape: {quality_train.shape}")
    print(f"Test data shape: {quality_test.shape}")
    print(f"Number of options (models × token_limits): {len(option_names)}")

    # Prepare LLM embeddings for MIRT
    print("\nPreparing LLM embeddings for MIRT...")
    llm_texts = {
        'Mistral-7B-Instruct-v0.2': 'Mistral-7B-Instruct-v0.2 is a 7B parameter instruction-tuned model',
        'GLM-4.5-Air': 'GLM-4.5-Air is a lightweight general-purpose language model',
    }

    # Create LLM embeddings (one per option)
    text_encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    llm_embeddings_list = []
    for option_name in option_names:
        # Extract model name from option_name (remove token limit suffix)
        model_name = '_'.join(option_name.split('_')[:-2]) + '-' + option_name.split('_')[-2]
        if model_name not in llm_texts:
            # Try without the last part
            model_name = option_name.rsplit('_', 2)[0]

        text = llm_texts.get(model_name, f"{model_name} is a language model")
        # Add token limit info to distinguish options
        limit_name = option_name.split('_')[-2] + '_' + option_name.split('_')[-1]
        text_with_limit = f"{text} with token limit {limit_name}"
        llm_embeddings_list.append(text_encoder.encode(text_with_limit))

    llm_embeddings = np.array(llm_embeddings_list)
    print(f"LLM embeddings shape: {llm_embeddings.shape}")

    # Train CARROT-Linear
    print("\n" + "=" * 80)
    print("Training CARROT-Linear")
    print("=" * 80)

    carrot_checkpoint = f'ablation/checkpoints_comparison/{embedding_key}/carrot_linear'
    os.makedirs(carrot_checkpoint, exist_ok=True)

    if os.path.exists(os.path.join(carrot_checkpoint, 'linear_score.joblib')):
        print("Loading existing CARROT-Linear checkpoint...")
        carrot_linear = CarrotLinearBaseline(load_dir=carrot_checkpoint)
    else:
        print("Training CARROT-Linear...")
        carrot_linear = CarrotLinearBaseline(fit_intercept=True)
        carrot_linear.fit(
            embedding_train=embedding_train,
            quality_train=quality_train,
            token_count_train=token_count_train,
            save_dir=carrot_checkpoint
        )
        print("✓ CARROT-Linear training complete")

    # Train MIRT
    print("\n" + "=" * 80)
    print("Training MIRT")
    print("=" * 80)

    mirt_checkpoint = f'ablation/checkpoints_comparison/{embedding_key}/irt_mirt'
    os.makedirs(mirt_checkpoint, exist_ok=True)

    if os.path.exists(os.path.join(mirt_checkpoint, 'mirt_model.pt')):
        print("Loading existing MIRT checkpoint...")
        mirt = IRTBaseline(latent_dim=32, device='cuda', load_dir=mirt_checkpoint)
    else:
        print("Training MIRT...")
        mirt = IRTBaseline(latent_dim=32, device='cuda')
        mirt.fit(
            embedding_train=embedding_train,
            quality_train=quality_train,
            llm_embeddings=llm_embeddings,
            llm_names=option_names,
            lr=3e-3,
            batch_size=128,
            epochs=10,
            save_dir=mirt_checkpoint
        )
        print("✓ MIRT training complete")

    # Load CoRE predictors
    print("\n" + "=" * 80)
    print("Loading CoRE Predictors")
    print("=" * 80)

    # Build CoRE predictions
    core_pred_score_test = []
    core_pred_count_test = []
    core_true_score_test = []
    core_true_count_test = []

    checkpoint_dir = f'ablation/checkpoints_comparison/{embedding_key}'

    for model_name, model_size, csv_path in tqdm(models, desc="Loading CoRE"):
        model_checkpoint = os.path.join(checkpoint_dir, model_name)

        # Load predictor
        predictor = TokenPerformancePredictor(token_limits=token_limits_score, load_dir=model_checkpoint)

        # Get predictions for test set (16 token limits)
        quality_combined, _ = predictor.predict_combined(embedding_test)  # (n_test, 16)

        # Load ground truth
        df = pd.read_csv(csv_path, encoding='latin-1', encoding_errors='replace')

        # For each token limit (16 limits)
        for i, (limit_score, limit_count) in enumerate(zip(token_limits_score, token_limits_count)):
            core_pred_score_test.append(quality_combined[:, i])  # Extract column i
            core_pred_count_test.append(df.loc[test_idx, limit_count].values)
            core_true_score_test.append(df.loc[test_idx, limit_score].values)
            core_true_count_test.append(df.loc[test_idx, limit_count].values)

    # Stack into matrices
    core_Y_hat_score = np.column_stack(core_pred_score_test)
    core_Y_hat_count = np.column_stack(core_pred_count_test)

    # Evaluate all methods
    print("\n" + "=" * 80)
    print("Evaluating Routing Performance")
    print("=" * 80)

    # Evaluate CoRE
    print("\nEvaluating CoRE...")
    core_costs, core_perfs = route_baseline(
        core_Y_hat_score, core_Y_hat_count, quality_test, token_count_test, lambda_range, sizes_vec
    )
    core_audc = compute_audc(core_costs, core_perfs)
    core_peak = core_perfs.max()

    # Evaluate CARROT-Linear
    print("Evaluating CARROT-Linear...")
    carrot_Y_hat_score, carrot_Y_hat_count = carrot_linear.predict(embedding_test)
    carrot_costs, carrot_perfs = route_baseline(
        carrot_Y_hat_score, carrot_Y_hat_count, quality_test, token_count_test, lambda_range, sizes_vec
    )
    carrot_audc = compute_audc(carrot_costs, carrot_perfs)
    carrot_peak = carrot_perfs.max()

    # Evaluate MIRT
    print("Evaluating MIRT...")
    mirt_Y_hat_score = mirt.predict(embedding_test)
    # MIRT doesn't predict token counts, use ground truth
    mirt_costs, mirt_perfs = route_baseline(
        mirt_Y_hat_score, token_count_test, quality_test, token_count_test, lambda_range, sizes_vec
    )
    mirt_audc = compute_audc(mirt_costs, mirt_perfs)
    mirt_peak = mirt_perfs.max()

    # Compute QNC with global normalization
    print("\n" + "=" * 80)
    print("Computing QNC with global normalization")
    print("=" * 80)

    # Find target accuracy (best model with unlimited tokens)
    best_llm_acc = 0.0
    for i, (model_name, model_size, csv_path) in enumerate(models):
        df = pd.read_csv(csv_path, encoding='latin-1', encoding_errors='replace')
        unlimited_acc = df.loc[test_idx, 'unlimited_score'].mean()
        if unlimited_acc > best_llm_acc:
            best_llm_acc = unlimited_acc
            best_llm = model_name

    target_accuracy = best_llm_acc * 0.95
    print(f"Best LLM: {best_llm} with accuracy {best_llm_acc:.4f}")
    print(f"Target accuracy (95%): {target_accuracy:.4f}")

    # Global cost normalization
    all_costs = np.concatenate([core_costs, carrot_costs, mirt_costs])
    global_min_cost = np.min(all_costs)
    global_max_cost = np.max(all_costs)
    print(f"Global cost range: [{global_min_cost:.2f}, {global_max_cost:.2f}]")

    # Compute QNC for each method
    core_qnc = compute_qnc(core_costs, core_perfs, target_accuracy, global_min_cost, global_max_cost)
    carrot_qnc = compute_qnc(carrot_costs, carrot_perfs, target_accuracy, global_min_cost, global_max_cost)
    mirt_qnc = compute_qnc(mirt_costs, mirt_perfs, target_accuracy, global_min_cost, global_max_cost)

    # Save results
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    results = {
        'Method': ['CoRE', 'CARROT-Linear', 'MIRT'],
        'AUDC': [core_audc, carrot_audc, mirt_audc],
        'QNC': [core_qnc, carrot_qnc, mirt_qnc],
        'Peak_Accuracy': [core_peak, carrot_peak, mirt_peak]
    }

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Save to CSV
    output_dir = 'ablation/results'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'method_comparison.csv'), index=False)
    print(f"\n✓ Saved comparison to ablation/results/method_comparison.csv")

    # Save detailed curves
    curves_data = []
    for i, lam in enumerate(lambda_range):
        curves_data.append({
            'method': 'CoRE',
            'lambda': lam,
            'cost': core_costs[i],
            'performance': core_perfs[i]
        })
        curves_data.append({
            'method': 'CARROT-Linear',
            'lambda': lam,
            'cost': carrot_costs[i],
            'performance': carrot_perfs[i]
        })
        curves_data.append({
            'method': 'MIRT',
            'lambda': lam,
            'cost': mirt_costs[i],
            'performance': mirt_perfs[i]
        })

    curves_df = pd.DataFrame(curves_data)
    curves_df.to_csv(os.path.join(output_dir, 'method_comparison_curves.csv'), index=False)
    print(f"✓ Saved curves to ablation/results/method_comparison_curves.csv")

    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print(f"Used {len(models)} models: {', '.join([m[0] for m in models])}")
    print(f"Embedding: all-MiniLM-L6-v2 (384-dim)")
    print("=" * 80)


if __name__ == "__main__":
    main()
