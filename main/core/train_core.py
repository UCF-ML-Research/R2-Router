"""Train CoRE (sklearn-based) predictors on specified LLM pool."""

import argparse
import os
import numpy as np
from ..shared.dataset_manager import DatasetManager
from ..shared.router_dataset import RouterDataset
from .predictor_sklearn import TokenPerformancePredictor as SklearnPredictor
from .predictor import TokenPerformancePredictor as TorchPredictor


def parse_args():
    parser = argparse.ArgumentParser(description='Train CoRE predictors')
    parser.add_argument('--model', action='append', nargs=4,
                        metavar=('NAME', 'SIZE', 'CSV', 'CHECKPOINT'),
                        help='Model configuration: name size csv_path checkpoint_path (checkpoint auto-generated)')
    parser.add_argument('--model_type', type=str, default='ridge',
                        help='Model type: linear, ridge, lasso, elasticnet, random_forest, gradient_boosting, mlp, torch_mlp')
    parser.add_argument('--alpha', type=float, default=10.0,
                        help='Regularization strength (for ridge/lasso/elasticnet)')
    parser.add_argument('--l1_ratio', type=float, default=0.5,
                        help='ElasticNet mixing parameter (0=Ridge, 1=Lasso)')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of estimators (for tree-based models)')
    parser.add_argument('--max_depth', type=int, default=3,
                        help='Max depth (for tree-based models, 0=None)')
    parser.add_argument('--hidden_layers', type=str, default='100,50',
                        help='MLP hidden layer sizes (comma-separated)')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Max iterations for MLP (default: 1000)')
    # PyTorch MLP parameters
    parser.add_argument('--torch_epochs', type=int, default=100,
                        help='Training epochs for torch_mlp (default: 100)')
    parser.add_argument('--torch_lr', type=float, default=1e-4,
                        help='Learning rate for torch_mlp (default: 1e-4)')
    parser.add_argument('--torch_dropout', type=float, default=0.5,
                        help='Dropout rate for torch_mlp (default: 0.5)')
    parser.add_argument('--torch_batch_size', type=int, default=128,
                        help='Batch size for torch_mlp (default: 128)')
    return parser.parse_args()


def get_training_scheme_suffix(args, hidden_layer_sizes, max_depth):
    """Generate suffix for checkpoint directory based on training scheme."""
    if args.model_type == "linear":
        return "linear"
    elif args.model_type == "ridge":
        return f"ridge_alpha{args.alpha}"
    elif args.model_type == "lasso":
        return f"lasso_alpha{args.alpha}"
    elif args.model_type == "elasticnet":
        return f"elasticnet_alpha{args.alpha}_l1{args.l1_ratio}"
    elif args.model_type == "random_forest":
        depth_str = "unlimited" if max_depth is None else f"d{max_depth}"
        return f"rf_n{args.n_estimators}_{depth_str}"
    elif args.model_type == "gradient_boosting":
        depth_str = "unlimited" if max_depth is None else f"d{max_depth}"
        return f"gbm_n{args.n_estimators}_{depth_str}"
    elif args.model_type == "mlp":
        layers_str = "_".join(str(x) for x in hidden_layer_sizes)
        return f"mlp_{layers_str}"
    elif args.model_type == "torch_mlp":
        layers_str = "_".join(str(x) for x in hidden_layer_sizes)
        return f"torch_mlp_{layers_str}_ep{args.torch_epochs}"
    else:
        return args.model_type


def main():
    args = parse_args()

    # Parse hidden layers
    hidden_layer_sizes = tuple([int(x) for x in args.hidden_layers.split(',')])
    max_depth = None if args.max_depth == 0 else args.max_depth

    # Generate training scheme suffix
    scheme_suffix = get_training_scheme_suffix(args, hidden_layer_sizes, max_depth)

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
    print(f"TRAINING CoRE WITH MODEL TYPE: {args.model_type}")
    print(f"Training Scheme: {scheme_suffix}")
    if args.model_type in ["ridge", "lasso", "elasticnet"]:
        print(f"  Alpha: {args.alpha}")
        if args.model_type == "elasticnet":
            print(f"  L1 Ratio: {args.l1_ratio}")
    elif args.model_type in ["random_forest", "gradient_boosting"]:
        print(f"  N Estimators: {args.n_estimators}")
        print(f"  Max Depth: {max_depth}")
    elif args.model_type == "mlp":
        print(f"  Hidden Layers: {hidden_layer_sizes}")
        print(f"  Max Iterations: {args.max_iter}")
    elif args.model_type == "torch_mlp":
        print(f"  Hidden Layers: {hidden_layer_sizes}")
        print(f"  Epochs: {args.torch_epochs}")
        print(f"  Learning Rate: {args.torch_lr}")
        print(f"  Dropout: {args.torch_dropout}")
        print(f"  Batch Size: {args.torch_batch_size}")
    print("=" * 80)

    # Prepare model-specific kwargs
    if args.model_type == "random_forest":
        model_kwargs = {"n_estimators": args.n_estimators, "max_depth": max_depth}
    elif args.model_type == "gradient_boosting":
        model_kwargs = {"n_estimators": args.n_estimators, "max_depth": max_depth}
    elif args.model_type == "mlp":
        model_kwargs = {"hidden_layer_sizes": hidden_layer_sizes, "max_iter": args.max_iter}
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

        # Handle PyTorch MLP separately
        if args.model_type == "torch_mlp":
            print(f"Training with PyTorch MLP...")
            predictor = TorchPredictor(
                hidden_dims=list(hidden_layer_sizes),
                dropout=args.torch_dropout,
                dataset=dataset
            )
            predictor.fit(
                save_dir=checkpoint_with_scheme,
                lr=args.torch_lr,
                batch_size=args.torch_batch_size,
                epochs=args.torch_epochs,
                plot_dir=plot_dir_with_scheme
            )

            # Test prediction
            test_data = dataset.get_test_set_score()
            score_pred, count_pred = predictor.predict(test_data["X"])
            print(f"[OK] {name}: PyTorch MLP training done.")
            print(f"   - Score predictions shape: {score_pred.shape}")
            print(f"   - Token count predictions shape: {count_pred.shape}")
        else:
            # Use sklearn-based predictor
            print(f"Training with sklearn {args.model_type}...")
            predictor = SklearnPredictor(
                token_limits=token_limits_score,
                model_type=args.model_type,
                alpha=args.alpha,
                l1_ratio=args.l1_ratio,
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
    print("All CoRE models trained successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
