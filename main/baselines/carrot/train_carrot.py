"""Train CARROT baselines on specified LLM pool."""

import argparse
import os
import numpy as np
import pandas as pd
from ...shared.dataset_manager import DatasetManager
from .baselines_carrot import CarrotKNNBaseline, CarrotLinearBaseline


def parse_args():
    parser = argparse.ArgumentParser(description='Train CARROT baselines')
    parser.add_argument('--model', action='append', nargs=4,
                        metavar=('NAME', 'SIZE', 'CSV', 'CHECKPOINT'),
                        help='Model configuration: name size csv_path checkpoint_path')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    dataset_manager = DatasetManager(
        embeddings_path="data/prompt_embeddings.pkl",
        train_ratio=0.8,
        seed=42
    )
    embeddings = dataset_manager.get_embeddings()
    train_idx, test_idx = dataset_manager.get_split_indices()

    # Load model data
    print("\nLoading model data...")
    quality_train_list = []
    token_count_train_list = []
    quality_test_list = []
    token_count_test_list = []
    model_names_loaded = []

    for name, size, csv_path, checkpoint in args.model:
        if not os.path.exists(csv_path):
            print(f"[SKIP] {name}: CSV not found")
            continue

        print(f"  Loading {name}...")
        df = pd.read_csv(csv_path)

        quality_train_list.append(df.loc[train_idx, 'unlimited_score'].values)
        token_count_train_list.append(df.loc[train_idx, 'unlimited_count'].values)
        quality_test_list.append(df.loc[test_idx, 'unlimited_score'].values)
        token_count_test_list.append(df.loc[test_idx, 'unlimited_count'].values)
        model_names_loaded.append(name)

    # Prepare data
    embedding_train = embeddings[train_idx]
    embedding_test = embeddings[test_idx]
    quality_train = np.column_stack(quality_train_list)
    token_count_train = np.column_stack(token_count_train_list)
    quality_test = np.column_stack(quality_test_list)
    token_count_test = np.column_stack(token_count_test_list)

    print(f"\n[OK] Loaded {len(model_names_loaded)} models")
    print(f"  Training shape: {quality_train.shape}")

    # Train CARROT-KNN
    print("\n>>> Training CARROT-KNN...")
    carrot_knn = CarrotKNNBaseline(n_neighbors_score=256, n_neighbors_count=256, metric="cosine")
    carrot_knn.fit(
        embedding_train=embedding_train,
        quality_train=quality_train,
        token_count_train=token_count_train,
        save_dir="./checkpoints/main/carrot_knn"
    )
    print("[OK] CARROT-KNN trained")

    # Train CARROT-Linear
    print("\n>>> Training CARROT-Linear...")
    carrot_linear = CarrotLinearBaseline(fit_intercept=True)
    carrot_linear.fit(
        embedding_train=embedding_train,
        quality_train=quality_train,
        token_count_train=token_count_train,
        save_dir="./checkpoints/main/carrot_linear"
    )
    print("[OK] CARROT-Linear trained")
    print("\n[DONE] CARROT training complete!")


if __name__ == "__main__":
    main()
