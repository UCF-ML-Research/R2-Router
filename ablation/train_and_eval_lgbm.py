#!/usr/bin/env python3
"""
Train LGBM-based CoRE predictors and compare with main experiment baselines.
This is a predictor architecture ablation study.
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
from main.shared.router_dataset import RouterDataset
from ablation.predictor_lgbm import TokenPerformancePredictor as RFPredictor
from main.baselines.carrot.baselines_carrot import CarrotLinearBaseline
from main.baselines.irt.baselines_irt import IRTBaseline


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


def route_rf_core(predictors_dict, models, embedding_test, lambda_range,
                  quality_test_dict, token_count_test_dict, token_limits_score, token_limits_count):
    """Route using RandomForest CoRE predictors."""
    costs_list = []
    perfs_list = []

    for lam in lambda_range:
        # For each query, compute risk for all (model, token_limit) options
        all_risks = []
        all_true_scores = []
        all_true_costs = []

        for model_name, model_size, _ in models:
            predictor = predictors_dict[model_name]
            quality_combined, token_count_pred = predictor.predict_combined(embedding_test)

            # For each token limit
            for i, (limit_score, limit_count) in enumerate(zip(token_limits_score, token_limits_count)):
                pred_score = quality_combined[:, i]
                true_count = token_count_test_dict[model_name][limit_count]
                pred_cost = true_count * model_size

                risk = (1 - lam) * pred_score - lam * pred_cost

                all_risks.append(risk)
                all_true_scores.append(quality_test_dict[model_name][limit_score])
                all_true_costs.append(true_count * model_size)

        # Stack and find best option for each query
        risks_matrix = np.column_stack(all_risks)  # (n_test, n_options)
        best_idx = np.argmax(risks_matrix, axis=1)

        # Get actual performance and cost for selected options
        n_test = len(embedding_test)
        selected_scores = []
        selected_costs = []
        for i in range(n_test):
            option_idx = best_idx[i]
            selected_scores.append(all_true_scores[option_idx][i])
            selected_costs.append(all_true_costs[option_idx][i])

        # Average across test set
        costs_list.append(np.mean(selected_costs))
        perfs_list.append(np.mean(selected_scores))

    return np.array(costs_list), np.array(perfs_list)


def main():
    """Main training and evaluation pipeline."""
    print("=" * 80)
    print("Predictor Ablation: RandomForest-based CoRE vs Main Baselines")
    print("=" * 80)

    # Initialize dataset manager
    dataset_manager = DatasetManager(
        embeddings_path='data/prompt_embeddings.pkl',
        train_ratio=0.8,
        seed=42
    )

    embeddings = dataset_manager.get_embeddings()
    train_idx, test_idx = dataset_manager.get_split_indices()
    embedding_train = embeddings[train_idx]
    embedding_test = embeddings[test_idx]

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    # LLM Pool from main experiment (10 models)
    models = [
        ('Mistral-7B-Instruct-v0.2', 0.20, 'data/Mistral-7B-Instruct-v0.2.csv'),
        ('GLM-4.5-Air', 0.85, 'data/GLM-4.5-Air.csv'),
        ('gemma-3-4b-it', 0.06815, 'data/gemma-3-4b-it.csv'),
        ('gemma-3-1b-it', 0.0170375, 'data/gemma-3-1b-it.csv'),
        ('gemma-3-270m-it', 0.004259375, 'data/gemma-3-270m-it.csv'),
        ('Llama-3.1-70B-Instruct', 0.40, 'data/Llama-3.1-70B-Instruct.csv'),
        ('Llama-3.2-3B-Instruct', 0.02, 'data/Llama-3.2-3B-Instruct.csv'),
        ('Qwen2.5-Math-1.5B-Instruct', 0.09, 'data/Qwen2.5-Math-1.5B-Instruct.csv'),
        ('Qwen2.5-Math-7B-Instruct', 0.35, 'data/Qwen2.5-Math-7B-Instruct.csv'),
        ('Qwen3-0.6B', 0.0173, 'data/Qwen3-0.6B.csv'),
    ]

    # Token limits
    token_limits_score = [f'{t}_score' for t in [10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000]]
    token_limits_score.append('unlimited_score')
    token_limits_count = [t.replace('_score', '_count') for t in token_limits_score]

    # Lambda range for routing
    lambda_range = np.linspace(0, 1e-4, 100)

    # Train RandomForest CoRE predictors
    print("\n" + "=" * 80)
    print("Training RandomForest CoRE Predictors")
    print("=" * 80)

    rf_predictors = {}
    quality_test_dict = {}
    token_count_test_dict = {}

    checkpoint_dir = 'ablation/checkpoints_rf'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for model_name, model_size, csv_path in tqdm(models, desc="Training RF CoRE"):
        model_checkpoint = os.path.join(checkpoint_dir, model_name)

        # Check if already trained
        if os.path.exists(os.path.join(model_checkpoint, 'limited_score_predictors.joblib')):
            print(f"  Loading existing checkpoint for {model_name}...")
            predictor = RFPredictor(token_limits=token_limits_score, load_dir=model_checkpoint)
        else:
            # Create RouterDataset
            dataset = RouterDataset(
                embeddings=embeddings,
                score_df_path=csv_path,
                target_tokens_score=token_limits_score,
                train_idx=train_idx,
                test_idx=test_idx
            )

            # Get training data
            train_set = dataset.get_train_set_score()
            quality_train = train_set['y']  # Dict mapping token_name -> numpy array
            token_count_train = dataset.get_train_token_unlimited_count()

            # Train RandomForest predictor
            predictor = RFPredictor(
                token_limits=token_limits_score,
                n_estimators=100,
                max_depth=None,  # No max depth for RF
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                n_jobs=-1,  # Use all cores for parallel training
                random_state=42
            )

            predictor.fit(embedding_train, quality_train, token_count_train, save_dir=model_checkpoint)

        rf_predictors[model_name] = predictor

        # Store test data for evaluation
        df = pd.read_csv(csv_path, encoding='latin-1', encoding_errors='replace')
        quality_test_dict[model_name] = {t: df.loc[test_idx, t].values for t in token_limits_score}
        token_count_test_dict[model_name] = {t: df.loc[test_idx, t].values for t in token_limits_count}

    # Evaluate RandomForest CoRE
    print("\n" + "=" * 80)
    print("Evaluating RandomForest CoRE")
    print("=" * 80)

    rf_costs, rf_perfs = route_rf_core(
        rf_predictors, models, embedding_test, lambda_range,
        quality_test_dict, token_count_test_dict, token_limits_score, token_limits_count
    )
    rf_audc = compute_audc(rf_costs, rf_perfs)
    rf_peak = rf_perfs.max()

    print(f"RandomForest CoRE - AUDC: {rf_audc:.4f}, Peak: {rf_peak:.4f}")

    # Load and evaluate main experiment baselines
    print("\n" + "=" * 80)
    print("Loading Main Experiment Baselines")
    print("=" * 80)

    # Prepare data matrices for baselines
    quality_train_list = []
    token_count_train_list = []
    quality_test_list = []
    token_count_test_list = []
    sizes_list = []

    for model_name, model_size, csv_path in tqdm(models, desc="Loading data"):
        df = pd.read_csv(csv_path, encoding='latin-1', encoding_errors='replace')

        # For each token limit, add as a separate option
        for limit_score, limit_count in zip(token_limits_score, token_limits_count):
            quality_train_list.append(df.loc[train_idx, limit_score].values)
            token_count_train_list.append(df.loc[train_idx, limit_count].values)
            quality_test_list.append(df.loc[test_idx, limit_score].values)
            token_count_test_list.append(df.loc[test_idx, limit_count].values)
            sizes_list.append(model_size)

    # Stack into matrices
    quality_train = np.column_stack(quality_train_list)
    token_count_train = np.column_stack(token_count_train_list)
    quality_test = np.column_stack(quality_test_list)
    token_count_test = np.column_stack(token_count_test_list)
    sizes_vec = np.array(sizes_list)

    print(f"Data shape: {quality_test.shape} (n_test, n_options)")

    # Evaluate CARROT-Linear (from main experiment)
    print("\nEvaluating CARROT-Linear (main)...")
    carrot_checkpoint = './checkpoints/main/carrot_linear'
    if os.path.exists(os.path.join(carrot_checkpoint, 'linear_score.joblib')):
        carrot_linear = CarrotLinearBaseline(load_dir=carrot_checkpoint)
    else:
        print("  Training CARROT-Linear...")
        carrot_linear = CarrotLinearBaseline(fit_intercept=True)
        carrot_linear.fit(embedding_train, quality_train, token_count_train, save_dir=carrot_checkpoint)

    carrot_Y_hat_score, carrot_Y_hat_count = carrot_linear.predict(embedding_test)
    carrot_costs, carrot_perfs = route_baseline(
        carrot_Y_hat_score, carrot_Y_hat_count, quality_test, token_count_test, lambda_range, sizes_vec
    )
    carrot_audc = compute_audc(carrot_costs, carrot_perfs)
    carrot_peak = carrot_perfs.max()

    print(f"CARROT-Linear - AUDC: {carrot_audc:.4f}, Peak: {carrot_peak:.4f}")

    # Evaluate MIRT (from main experiment)
    print("\nEvaluating MIRT (main)...")
    mirt_checkpoint = './checkpoints/main/irt_mirt'
    if os.path.exists(os.path.join(mirt_checkpoint, 'mirt_model.pt')):
        mirt = IRTBaseline(latent_dim=32, device='cuda', load_dir=mirt_checkpoint)
        mirt_Y_hat_score = mirt.predict(embedding_test)
        mirt_costs, mirt_perfs = route_baseline(
            mirt_Y_hat_score, token_count_test, quality_test, token_count_test, lambda_range, sizes_vec
        )
        mirt_audc = compute_audc(mirt_costs, mirt_perfs)
        mirt_peak = mirt_perfs.max()
        print(f"MIRT - AUDC: {mirt_audc:.4f}, Peak: {mirt_peak:.4f}")
    else:
        print("  MIRT checkpoint not found, skipping...")
        mirt_costs = None
        mirt_perfs = None
        mirt_audc = None
        mirt_peak = None

    # Compute QNC with global normalization
    print("\n" + "=" * 80)
    print("Computing QNC")
    print("=" * 80)

    # Find target accuracy (best model with unlimited tokens)
    best_llm_acc = 0.0
    for model_name, model_size, csv_path in models:
        df = pd.read_csv(csv_path, encoding='latin-1', encoding_errors='replace')
        unlimited_acc = df.loc[test_idx, 'unlimited_score'].mean()
        if unlimited_acc > best_llm_acc:
            best_llm_acc = unlimited_acc
            best_llm = model_name

    target_accuracy = best_llm_acc * 0.95
    print(f"Best LLM: {best_llm} with accuracy {best_llm_acc:.4f}")
    print(f"Target accuracy (95%): {target_accuracy:.4f}")

    # Global cost normalization
    if mirt_costs is not None:
        all_costs = np.concatenate([rf_costs, carrot_costs, mirt_costs])
    else:
        all_costs = np.concatenate([rf_costs, carrot_costs])

    global_min_cost = np.min(all_costs)
    global_max_cost = np.max(all_costs)
    print(f"Global cost range: [{global_min_cost:.2f}, {global_max_cost:.2f}]")

    # Compute QNC for each method
    rf_qnc = compute_qnc(rf_costs, rf_perfs, target_accuracy, global_min_cost, global_max_cost)
    carrot_qnc = compute_qnc(carrot_costs, carrot_perfs, target_accuracy, global_min_cost, global_max_cost)

    if mirt_costs is not None:
        mirt_qnc = compute_qnc(mirt_costs, mirt_perfs, target_accuracy, global_min_cost, global_max_cost)
    else:
        mirt_qnc = None

    # Save results
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if mirt_qnc is not None:
        results = {
            'Method': ['CoRE-RF', 'CARROT-Linear', 'MIRT'],
            'AUDC': [rf_audc, carrot_audc, mirt_audc],
            'QNC': [rf_qnc, carrot_qnc, mirt_qnc],
            'Peak_Accuracy': [rf_peak, carrot_peak, mirt_peak]
        }
    else:
        results = {
            'Method': ['CoRE-RF', 'CARROT-Linear'],
            'AUDC': [rf_audc, carrot_audc],
            'QNC': [rf_qnc, carrot_qnc],
            'Peak_Accuracy': [rf_peak, carrot_peak]
        }

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Save to CSV
    output_dir = 'ablation/results'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'rf_ablation_comparison.csv'), index=False)
    print(f"\n✓ Saved comparison to ablation/results/rf_ablation_comparison.csv")

    # Save detailed curves
    curves_data = []
    for i, lam in enumerate(lambda_range):
        curves_data.append({
            'method': 'CoRE-RF',
            'lambda': lam,
            'cost': rf_costs[i],
            'performance': rf_perfs[i]
        })
        curves_data.append({
            'method': 'CARROT-Linear',
            'lambda': lam,
            'cost': carrot_costs[i],
            'performance': carrot_perfs[i]
        })
        if mirt_costs is not None:
            curves_data.append({
                'method': 'MIRT',
                'lambda': lam,
                'cost': mirt_costs[i],
                'performance': mirt_perfs[i]
            })

    curves_df = pd.DataFrame(curves_data)
    curves_df.to_csv(os.path.join(output_dir, 'rf_ablation_curves.csv'), index=False)
    print(f"✓ Saved curves to ablation/results/rf_ablation_curves.csv")

    print("\n" + "=" * 80)
    print("RandomForest Ablation Study Complete!")
    print(f"Used {len(models)} models with main experiment embeddings")
    print("=" * 80)


if __name__ == "__main__":
    main()
