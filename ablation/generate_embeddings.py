#!/usr/bin/env python3
"""
Generate query embeddings using different embedding models.

This script generates embeddings for all queries in the dataset using:
1. Qwen3-Embedding-0.6B (current baseline)
2. RoBERTa-large
3. BERT-base-uncased
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    os.system("pip install sentence-transformers -q")
    from sentence_transformers import SentenceTransformer


def load_prompts(csv_path: str) -> list:
    """Load prompts from CSV file."""
    df = pd.read_csv(csv_path)
    return df['original_prompt'].tolist()


def generate_embeddings(prompts: list, model_name: str, batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings using specified model.

    Args:
        prompts: List of text prompts
        model_name: Name of the embedding model
        batch_size: Batch size for encoding

    Returns:
        Embeddings array of shape (n_prompts, embedding_dim)
    """
    print(f"\nGenerating embeddings with {model_name}...")
    print(f"Number of prompts: {len(prompts)}")

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = SentenceTransformer(model_name, device=device)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Generate embeddings in batches
    embeddings = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Encoding"):
        batch = prompts[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings


def main():
    """Generate embeddings for all models."""
    print("=" * 80)
    print("Embedding Model Ablation Study - Embedding Generation")
    print("=" * 80)

    # Load prompts from one of the CSV files (all have same prompts)
    csv_path = 'data/GLM-4.5-Air.csv'
    print(f"\nLoading prompts from {csv_path}...")
    prompts = load_prompts(csv_path)
    print(f"Loaded {len(prompts)} prompts")

    # Embedding models to test
    embedding_models = {
        'current_qwen3': None,  # Use existing embeddings from data/prompt_embeddings.pkl (1024-dim)
        'all_mpnet_base': 'sentence-transformers/all-mpnet-base-v2',  # 768-dim, MPNet (BERT-like, SOTA)
        'all_roberta_large': 'sentence-transformers/all-roberta-large-v1',  # 1024-dim, RoBERTa-large
        'all_minilm_l6': 'sentence-transformers/all-MiniLM-L6-v2',  # 384-dim, tiny but effective
    }

    # Generate embeddings for each model
    output_dir = 'ablation/embeddings'
    os.makedirs(output_dir, exist_ok=True)

    for model_key, model_name in embedding_models.items():
        print("\n" + "=" * 80)
        print(f"Processing: {model_key} ({model_name})")
        print("=" * 80)

        output_path = os.path.join(output_dir, f'{model_key}_embeddings.pkl')

        if os.path.exists(output_path):
            print(f"Embeddings already exist at {output_path}. Skipping...")
            continue

        # Special case: use existing embeddings for current_qwen3
        if model_name is None and model_key == 'current_qwen3':
            print("Copying existing embeddings from data/prompt_embeddings.pkl...")
            with open('data/prompt_embeddings.pkl', 'rb') as f:
                existing_data = pickle.load(f)
            embeddings = existing_data['embeddings']
            original_metadata = existing_data['metadata']
            print(f"Loaded existing embeddings: shape {embeddings.shape}")
            print(f"Original metadata: {original_metadata}")
        else:
            # Generate embeddings (smaller batch for large models)
            batch_size = 16 if 'large' in model_name else 64
            embeddings = generate_embeddings(prompts, model_name, batch_size=batch_size)

        # Save embeddings
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"✓ Saved embeddings to {output_path}")

        # Save metadata
        metadata = {
            'model_name': model_name if model_name else 'Qwen/Qwen3-0.6B (vLLM)',
            'model_key': model_key,
            'n_prompts': len(prompts),
            'embedding_dim': embeddings.shape[1],
            'dtype': str(embeddings.dtype)
        }
        metadata_path = os.path.join(output_dir, f'{model_key}_metadata.txt')
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"✓ Saved metadata to {metadata_path}")

    print("\n" + "=" * 80)
    print("Embedding generation completed!")
    print("=" * 80)

    # List all generated embeddings
    print("\nGenerated embeddings:")
    for model_key in embedding_models.keys():
        emb_path = os.path.join(output_dir, f'{model_key}_embeddings.pkl')
        if os.path.exists(emb_path):
            with open(emb_path, 'rb') as f:
                emb = pickle.load(f)
            print(f"  {model_key:20s}: shape {emb.shape}, size {os.path.getsize(emb_path)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
