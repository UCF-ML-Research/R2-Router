"""Train IRT baselines (MIRT and NIRT) on specified LLM pool."""

import argparse
import os
import json
import numpy as np
import pandas as pd
from ...shared.dataset_manager import DatasetManager
from .baselines_irt import IRTBaseline, NIRTBaseline
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(description='Train IRT baselines (MIRT and NIRT)')
    parser.add_argument('--model', action='append', nargs=4,
                        metavar=('NAME', 'SIZE', 'CSV', 'CHECKPOINT'),
                        help='Model configuration: name size csv_path checkpoint_path')
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Dimension of latent ability space (default: 32)')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Learning rate (default: 3e-3)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs (default: 200)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device for training (cuda or cpu, default: auto-detect)')
    parser.add_argument('--llm-descriptions', type=str, default=None,
                        help='Path to JSON file with custom LLM descriptions (optional)')
    return parser.parse_args()


def load_llm_descriptions(json_path):
    """
    Load custom LLM descriptions from JSON file.

    Expected format:
    {
        "GLM-4.5-Air": "GLM-4.5-Air is a lightweight Chinese-English bilingual model...",
        "Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct is a small instruction-tuned model...",
        ...
    }

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary mapping model names to descriptions
    """
    with open(json_path, 'r') as f:
        descriptions = json.load(f)
    return descriptions


def main():
    args = parse_args()

    print("=" * 80)
    print("Training IRT Baselines (MIRT and NIRT)")
    print("=" * 80)
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device or 'auto-detect'}")
    print("=" * 80)

    # Load dataset
    dataset_manager = DatasetManager(
        embeddings_path="data/prompt_embeddings.pkl",
        train_ratio=0.8,
        seed=42
    )
    embeddings = dataset_manager.get_embeddings()
    train_idx, test_idx = dataset_manager.get_split_indices()

    # Load custom LLM descriptions if provided
    custom_descriptions = None
    if args.llm_descriptions:
        if os.path.exists(args.llm_descriptions):
            print(f"\nLoading custom LLM descriptions from {args.llm_descriptions}")
            custom_descriptions = load_llm_descriptions(args.llm_descriptions)
            print(f"[OK] Loaded descriptions for {len(custom_descriptions)} models")
        else:
            print(f"[WARNING] Custom descriptions file not found: {args.llm_descriptions}")
            print("Falling back to auto-generated descriptions")

    # Load model data
    print("\nLoading model data...")
    quality_train_list = []
    quality_test_list = []
    model_names_loaded = []
    llm_texts = {}

    for name, size, csv_path, checkpoint in args.model:
        if not os.path.exists(csv_path):
            print(f"[SKIP] {name}: CSV not found")
            continue

        print(f"  Loading {name}...")
        df = pd.read_csv(csv_path)

        quality_train_list.append(df.loc[train_idx, 'unlimited_score'].values)
        quality_test_list.append(df.loc[test_idx, 'unlimited_score'].values)
        model_names_loaded.append(name)

        # Use custom description if available, otherwise auto-generate
        if custom_descriptions and name in custom_descriptions:
            llm_texts[name] = custom_descriptions[name]
            print(f"    Using custom description: {llm_texts[name][:80]}...")
        else:
            llm_texts[name] = f"{name} is a language model with {size}B parameters"
            print(f"    Using auto-generated description")

    # Prepare data
    embedding_train = embeddings[train_idx]
    embedding_test = embeddings[test_idx]
    quality_train = np.column_stack(quality_train_list)
    quality_test = np.column_stack(quality_test_list)

    print(f"\n[OK] Loaded {len(model_names_loaded)} models")
    print(f"  Training shape: {quality_train.shape}")
    print(f"  Test shape: {quality_test.shape}")

    # Generate LLM embeddings
    print("\n>>> Generating LLM embeddings...")
    encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    llm_embedding_texts = [llm_texts[name] for name in model_names_loaded]
    llm_embeddings = encoder.encode(llm_embedding_texts, show_progress_bar=True)
    llm_embeddings = np.array(llm_embeddings)
    print(f"[OK] LLM embeddings shape: {llm_embeddings.shape}")

    # Train MIRT
    print("\n" + "=" * 80)
    print("Training MIRT (Multidimensional IRT)")
    print("=" * 80)
    mirt = IRTBaseline(
        latent_dim=args.latent_dim,
        device=args.device
    )
    mirt.fit(
        embedding_train=embedding_train,
        quality_train=quality_train,
        llm_embeddings=llm_embeddings,
        llm_names=model_names_loaded,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_dir="./checkpoints/main/irt_mirt"
    )
    print("[OK] MIRT trained and saved to ./checkpoints/main/irt_mirt")

    # Train NIRT
    print("\n" + "=" * 80)
    print("Training NIRT (Neural IRT)")
    print("=" * 80)
    nirt = NIRTBaseline(
        latent_dim=args.latent_dim,
        device=args.device
    )
    nirt.fit(
        embedding_train=embedding_train,
        quality_train=quality_train,
        llm_embeddings=llm_embeddings,
        llm_names=model_names_loaded,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_dir="./checkpoints/main/irt_nirt"
    )
    print("[OK] NIRT trained and saved to ./checkpoints/main/irt_nirt")

    print("\n" + "=" * 80)
    print("IRT TRAINING COMPLETE!")
    print("=" * 80)
    print("Checkpoints saved to:")
    print("  - ./checkpoints/main/irt_mirt/")
    print("  - ./checkpoints/main/irt_nirt/")


if __name__ == "__main__":
    main()
