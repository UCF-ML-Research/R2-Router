#!/usr/bin/env python3
"""
Compare CoRE vs CARROT-Linear vs MIRT using all-MiniLM-L6-v2 embeddings.

This script trains all three methods with the same embedding model and LLM pool
to provide a fair comparison of routing performance.
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
from main.shared.llm_loader import load_llm
from main.shared.router_dataset import RouterDataset
from main.core.predictor_sklearn import TokenPerformancePredictor, route_scores
from main.baselines.carrot.baselines_carrot import CarrotLinearBaseline, route_baseline
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


def train_core_predictors(embedding_key: str,
                          embedding_path: str,
                          models: list,
                          dataset_manager: DatasetManager,
                          token_limits_score: list,
                          token_limits_count: list) -> dict:
    """Train CoRE predictors for all LLMs with given embedding."""
    print("\n" + "=" * 80)
    print(f"Training CoRE Predictors with {embedding_key}")
    print("=" * 80)

    # Load embeddings
    print(f"Loading embeddings from {embedding_path}...")
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Embeddings shape: {embeddings.shape}")

    train_idx, test_idx = dataset_manager.get_split_indices()

    # Train predictors for each LLM
    llms = {}
    checkpoint_dir = f'ablation/checkpoints_comparison/{embedding_key}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for model_name, model_size, csv_path in tqdm(models, desc=f"Training CoRE for {embedding_key}"):
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
            print(f"\n  Training {model_name}...")

            # Create dataset
            dataset = RouterDataset(
                embeddings=embeddings,
                score_df_path=csv_path,
                target_tokens_score=token_limits_score,
                train_idx=train_idx,
                test_idx=test_idx
            )

            # Create and train predictor (use Ridge for stability)
            predictor = TokenPerformancePredictor(
                token_limits=token_limits_score,
                model_type="ridge",
                alpha=0.01
            )

            # Prepare training data
            train_data = dataset.get_train_set_score()
            test_data = dataset.get_test_set_score()

            X_train = train_data['X']
            y_train = train_data['y']
            X_test = test_data['X']
            y_test = test_data['y']

            # Extract limited and unlimited scores
            limited_token_names = [t for t in token_limits_score if t != 'unlimited_score']
            quality_train_limited = {t: y_train[t] for t in limited_token_names}
            quality_test_limited = {t: y_test[t] for t in limited_token_names}
            quality_train_unlimited = y_train['unlimited_score']
            quality_test_unlimited = y_test['unlimited_score']

            # Token counts
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
                plot_dir=None
            )
            print(f"  ✓ Trained {model_name}")
        else:
            print(f"\n  Checkpoints exist for {model_name}, loading...")

        # Load the LLM
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

    return llms


def train_carrot_linear(llms: dict, embedding_key: str) -> CarrotLinearBaseline:
    """Train CARROT-Linear baseline."""
    print("\n" + "=" * 80)
    print(f"Training CARROT-Linear with {embedding_key}")
    print("=" * 80)

    checkpoint_dir = f'ablation/checkpoints_comparison/{embedding_key}/carrot_linear'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check if checkpoints exist
    if os.path.exists(os.path.join(checkpoint_dir, 'linear_score.joblib')) and \
       os.path.exists(os.path.join(checkpoint_dir, 'linear_count.joblib')):
        print("Loading existing CARROT-Linear checkpoint...")
        carrot = CarrotLinearBaseline(llms, load_dir=checkpoint_dir)
    else:
        print("Training CARROT-Linear...")
        carrot = CarrotLinearBaseline(llms, fit_intercept=True)
        carrot.fit(save_dir=checkpoint_dir, plot_dir=None)
        print("✓ CARROT-Linear training complete")

    return carrot


def train_mirt(llms: dict, embedding_key: str, llm_texts: dict) -> IRTBaseline:
    """Train MIRT baseline."""
    print("\n" + "=" * 80)
    print(f"Training MIRT with {embedding_key}")
    print("=" * 80)

    checkpoint_dir = f'ablation/checkpoints_comparison/{embedding_key}/irt_mirt'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check if checkpoints exist
    if os.path.exists(os.path.join(checkpoint_dir, 'mirt_model.pt')):
        print("Loading existing MIRT checkpoint...")
        mirt = IRTBaseline(llms, llm_texts=llm_texts, latent_dim=32,
                          device='cuda', load_dir=checkpoint_dir)
    else:
        print("Training MIRT...")
        mirt = IRTBaseline(llms, llm_texts=llm_texts, latent_dim=32, device='cuda')
        mirt.fit(lr=3e-3, batch_size=128, epochs=10,
                save_dir=checkpoint_dir, plot_dir=None)
        print("✓ MIRT training complete")

    return mirt


def main():
    """Main comparison pipeline."""
    print("=" * 80)
    print("Embedding Ablation: CoRE vs CARROT-Linear vs MIRT")
    print("=" * 80)

    # Use all-MiniLM-L6-v2 embedding
    embedding_key = 'all_minilm_l6'
    embedding_path = 'ablation/embeddings/all_minilm_l6_embeddings.pkl'

    # Initialize dataset manager
    dataset_manager = DatasetManager(
        embeddings_path='data/prompt_embeddings.pkl',  # Just for split
        train_ratio=0.8,
        seed=42
    )

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

    # Step 1: Train CoRE predictors
    llms = train_core_predictors(
        embedding_key=embedding_key,
        embedding_path=embedding_path,
        models=models,
        dataset_manager=dataset_manager,
        token_limits_score=token_limits_score,
        token_limits_count=token_limits_count
    )

    # Step 2: Train CARROT-Linear
    carrot_linear = train_carrot_linear(llms, embedding_key)

    # Step 3: Train MIRT
    llm_texts = {
        'Mistral-7B-Instruct-v0.2': 'Mistral-7B-Instruct-v0.2 is a 7B parameter instruction-tuned model',
        'GLM-4.5-Air': 'GLM-4.5-Air is a lightweight general-purpose language model',
        'gemma-3-4b-it': 'gemma-3-4b-it is a 4B parameter instruction-tuned model from Google',
        'gemma-3-1b-it': 'gemma-3-1b-it is a 1B parameter compact instruction-tuned model',
        'gemma-3-270m-it': 'gemma-3-270m-it is a 270M parameter tiny instruction-tuned model',
        'Llama-3.1-70B-Instruct': 'Llama-3.1-70B-Instruct is a large 70B parameter instruction-tuned model',
        'Llama-3.2-3B-Instruct': 'Llama-3.2-3B-Instruct is a 3B parameter instruction-tuned model',
        'Qwen2.5-Math-1.5B-Instruct': 'Qwen2.5-Math-1.5B-Instruct is specialized for mathematical reasoning',
        'Qwen2.5-Math-7B-Instruct': 'Qwen2.5-Math-7B-Instruct is a large math-specialized model',
        'Qwen3-0.6B': 'Qwen3-0.6B is a compact general-purpose language model',
    }
    mirt = train_mirt(llms, embedding_key, llm_texts)

    # Step 4: Evaluate all methods
    print("\n" + "=" * 80)
    print("Evaluating Routing Performance")
    print("=" * 80)

    # Evaluate CoRE
    print("\nEvaluating CoRE...")
    core_costs, core_perfs = route_scores(llms, lambda_range)
    core_audc = compute_audc(core_costs, core_perfs)
    core_peak = core_perfs.max()

    # Evaluate CARROT-Linear
    print("Evaluating CARROT-Linear...")
    carrot_costs, carrot_perfs = route_baseline(carrot_linear, llms, lambda_range)
    carrot_audc = compute_audc(carrot_costs, carrot_perfs)
    carrot_peak = carrot_perfs.max()

    # Evaluate MIRT
    print("Evaluating MIRT...")
    mirt_costs, mirt_perfs = route_baseline(mirt, llms, lambda_range)
    mirt_audc = compute_audc(mirt_costs, mirt_perfs)
    mirt_peak = mirt_perfs.max()

    # Compute QNC with global normalization
    print("\n" + "=" * 80)
    print("Computing QNC with global normalization")
    print("=" * 80)

    # Find target accuracy (95% of best LLM)
    train_idx, test_idx = dataset_manager.get_split_indices()
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)

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


if __name__ == "__main__":
    main()
