"""Train R2-Router (sklearn-based) predictors on specified LLM pool."""

import argparse
import os
import numpy as np
from ..shared.dataset_manager import DatasetManager
from ..shared.router_dataset import RouterDataset
from .predictor_sklearn import TokenPerformancePredictor as SklearnPredictor


def parse_args():
    parser = argparse.ArgumentParser(description='Train R2-Router predictors')
    parser.add_argument('--model', action='append', nargs=4,
                        metavar=('NAME', 'SIZE', 'CSV', 'CHECKPOINT'),
                        help='Model configuration: name size csv_path checkpoint_path (checkpoint auto-generated)')
    parser.add_argument('--model_type', type=str, default='linear',
                        help='Model type: linear, random_forest, gradient_boosting')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of estimators (for tree-based models)')
    parser.add_argument('--max_depth', type=int, default=3,
                        help='Max depth (for tree-based models, 0=None)')
    return parser.parse_args()


def get_training_scheme_suffix(args, max_depth):
    """Generate suffix for checkpoint directory based on training scheme."""
    if args.model_type == "linear":
        return "linear"
    elif args.model_type == "random_forest":
        depth_str = "unlimited" if max_depth is None else f"d{max_depth}"
        return f"rf_n{args.n_estimators}_{depth_str}"
    elif args.model_type == "gradient_boosting":
        depth_str = "unlimited" if max_depth is None else f"d{max_depth}"
        return f"gbm_n{args.n_estimators}_{depth_str}"
    else:
        return args.model_type


def main():
    args = parse_args()

    max_depth = None if args.max_depth == 0 else args.max_depth

    # Generate training scheme suffix
    scheme_suffix = get_training_scheme_suffix(args, max_depth)

    # Define token limits
    token_limits_score = [
        '10_score', '20_score', '30_score', '40_score', '50_score',
        '80_score', '100_score', '150_score', '200_score', '300_score',
        '500_score', '800_score', '1200_score', '2000_score', '4000_score',
        'unlimited_score'
    ]

    # Initialize centralized dataset manager
    dataset_manager = DatasetManager(
        embeddings_path="data/prompt_embeddings.pkl",
        train_ratio=0.8,
        seed=42
    )

    embeddings = dataset_manager.get_embeddings()
    train_idx, test_idx = dataset_manager.get_split_indices()

    print("=" * 80)
    print(f"TRAINING R2-Router WITH MODEL TYPE: {args.model_type}")
    print(f"Training Scheme: {scheme_suffix}")
    if args.model_type in ["random_forest", "gradient_boosting"]:
        print(f"  N Estimators: {args.n_estimators}")
        print(f"  Max Depth: {max_depth}")
    print("=" * 80)

    # Prepare model-specific kwargs
    if args.model_type == "random_forest":
        model_kwargs = {"n_estimators": args.n_estimators, "max_depth": max_depth}
    elif args.model_type == "gradient_boosting":
        model_kwargs = {"n_estimators": args.n_estimators, "max_depth": max_depth}
    else:
        model_kwargs = {}

    # Train each model
    for name, size, csv_path, checkpoint in args.model:
        print("\n" + "=" * 80)
        print(f"Training model: {name}")
        print("=" * 80)

        # Check if CSV file exists
        if not os.path.exists(csv_path):
            print(f"[SKIP] {name}: CSV file not found at {csv_path}")
            continue

        # Create dataset with centralized split
        dataset = RouterDataset(
            embeddings=embeddings,
            score_df_path=csv_path,
            target_tokens_score=token_limits_score,
            train_idx=train_idx,
            test_idx=test_idx
        )

        checkpoint_with_scheme = checkpoint
        plot_dir_with_scheme = f"./plots/{name}_{scheme_suffix}"

        print(f"Training with sklearn {args.model_type}...")
        predictor = SklearnPredictor(
            token_limits=token_limits_score,
            model_type=args.model_type,
            **model_kwargs
        )

        # Split data for training
        train_data = dataset.get_train_set_score()
        test_data = dataset.get_test_set_score()

        quality_train_limited = {k: v for k, v in train_data["y"].items() if k != 'unlimited_score'}
        quality_train_unlimited = train_data["y"]['unlimited_score']
        quality_test_limited = {k: v for k, v in test_data["y"].items() if k != 'unlimited_score'}
        quality_test_unlimited = test_data["y"]['unlimited_score']

        # Train predictor
        predictor.fit(
            embedding_train=train_data["X"],
            quality_train_limited=quality_train_limited,
            quality_train_unlimited=quality_train_unlimited,
            token_count_train=dataset.get_train_token_unlimited_count(),
            embedding_test=test_data["X"],
            quality_test_limited=quality_test_limited,
            quality_test_unlimited=quality_test_unlimited,
            token_count_test=dataset.get_test_token_unlimited_count(),
            save_dir=checkpoint_with_scheme,
            plot_dir=plot_dir_with_scheme
        )

        # Test prediction
        quality_limited, quality_unlimited, token_count = predictor.predict(test_data["X"])
        print(f"[OK] {name}: Prediction done.")
        print(f"   - Limited quality shape: {quality_limited.shape}")
        print(f"   - Unlimited quality shape: {quality_unlimited.shape}")
        print(f"   - Token count shape: {token_count.shape}")

    print("\n" + "=" * 80)
    print("All R2-Router models trained successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
