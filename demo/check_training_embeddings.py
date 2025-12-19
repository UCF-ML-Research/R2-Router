"""
Check Training Embeddings
Identifies the embedding model used in your training data
"""

import sys
from pathlib import Path
import pickle
import numpy as np

print("="*60)
print("Training Embeddings Analysis")
print("="*60)
print()

# Path to training embeddings
embeddings_path = Path(__file__).parent.parent / "data" / "prompt_embeddings.pkl"

print(f"Loading embeddings from: {embeddings_path}")

if not embeddings_path.exists():
    print(f"❌ File not found!")
    print(f"   Expected: {embeddings_path}")
    print(f"   Please check the path in your data directory.")
    sys.exit(1)

# Load embeddings
try:
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
except Exception as e:
    print(f"❌ Failed to load embeddings: {e}")
    sys.exit(1)

print(f"✓ Embeddings loaded successfully")
print()

# Analyze embeddings
print("="*60)
print("Embedding Statistics")
print("="*60)

if isinstance(embeddings, np.ndarray):
    print(f"Type: NumPy array")
    print(f"Shape: {embeddings.shape}")
    print(f"Number of queries: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Data type: {embeddings.dtype}")
    print()

    # Statistics
    print(f"Min value: {embeddings.min():.4f}")
    print(f"Max value: {embeddings.max():.4f}")
    print(f"Mean value: {embeddings.mean():.4f}")
    print(f"Std deviation: {embeddings.std():.4f}")
    print()

    # Check if normalized
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Embedding norms:")
    print(f"  Min: {norms.min():.4f}")
    print(f"  Max: {norms.max():.4f}")
    print(f"  Mean: {norms.mean():.4f}")

    if np.allclose(norms, 1.0, atol=0.01):
        print(f"  → Embeddings are NORMALIZED (unit vectors)")
    else:
        print(f"  → Embeddings are NOT normalized")
    print()

    dimension = embeddings.shape[1]

else:
    print(f"Type: {type(embeddings)}")
    print(f"Warning: Unexpected format!")
    sys.exit(1)

# Suggest models based on dimension
print("="*60)
print("Likely Embedding Models (by dimension)")
print("="*60)

model_suggestions = {
    384: [
        ("all-MiniLM-L6-v2", "Most common, fast, good quality"),
        ("paraphrase-MiniLM-L6-v2", "Paraphrase detection focused"),
        ("all-MiniLM-L12-v2", "Larger variant, better quality"),
    ],
    768: [
        ("all-mpnet-base-v2", "High quality, slower"),
        ("all-distilroberta-v1", "RoBERTa-based, good quality"),
        ("bert-base-uncased", "Original BERT"),
        ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "Multilingual"),
    ],
    1024: [
        ("all-roberta-large-v1", "Large RoBERTa variant"),
    ],
    1536: [
        ("text-embedding-ada-002", "OpenAI Embeddings (most common)"),
    ],
    3072: [
        ("text-embedding-3-large", "OpenAI Embeddings (latest)"),
    ],
}

if dimension in model_suggestions:
    print(f"Dimension {dimension} matches these models:")
    print()
    for model, description in model_suggestions[dimension]:
        print(f"  • {model}")
        print(f"    {description}")
        print()

    print("Most likely:", model_suggestions[dimension][0][0])
else:
    print(f"Dimension {dimension} is uncommon.")
    print(f"You may be using a custom model.")

print()

# Configuration suggestion
print("="*60)
print("Demo Configuration")
print("="*60)

if dimension == 384:
    recommended_model = "all-MiniLM-L6-v2"
    print(f"✅ Default demo config should work!")
    print(f"   Current setting: all-MiniLM-L6-v2 (384-dim)")
elif dimension == 768:
    recommended_model = "all-mpnet-base-v2"
    print(f"⚠️  Update demo config:")
    print(f"   Edit demo/config.py:")
    print(f"   SENTENCE_TRANSFORMER_MODEL = 'all-mpnet-base-v2'")
elif dimension == 1536:
    print(f"⚠️  Using OpenAI embeddings")
    print(f"   You need to implement OpenAI embedder in demo/embedder.py")
    print(f"   See EMBEDDING_GUIDE.md for instructions")
else:
    print(f"⚠️  Unknown dimension {dimension}")
    print(f"   Check your training scripts to identify the model")

print()

# Test with demo embedder
print("="*60)
print("Testing Demo Embedder")
print("="*60)

try:
    # Add demo to path
    demo_dir = Path(__file__).parent
    if str(demo_dir) not in sys.path:
        sys.path.insert(0, str(demo_dir))

    import config
    from embedder import get_embedder

    print(f"Current demo config: {config.SENTENCE_TRANSFORMER_MODEL}")

    embedder = get_embedder(mock=False)
    embedder.load_model()

    print(f"Demo embedder dimension: {embedder.embedding_dim}")

    if embedder.embedding_dim == dimension:
        print(f"✅ DIMENSIONS MATCH! Demo embeddings will work correctly.")
    else:
        print(f"❌ DIMENSION MISMATCH!")
        print(f"   Training: {dimension}")
        print(f"   Demo: {embedder.embedding_dim}")
        print(f"   → Update config.SENTENCE_TRANSFORMER_MODEL")

except Exception as e:
    print(f"⚠️  Could not test demo embedder: {e}")
    print(f"   This is OK - just ensure dimensions match manually")

print()
print("="*60)
print("Summary")
print("="*60)
print(f"Training embeddings: {embeddings.shape[0]} queries × {dimension} dims")

if dimension in model_suggestions:
    print(f"Likely model: {model_suggestions[dimension][0][0]}")

print()
print("Next steps:")
print("  1. Update demo/config.py with correct model name")
print("  2. Run: python test_demo_basic.py")
print("  3. Check that dimensions match")
print()
print("See EMBEDDING_GUIDE.md for detailed instructions!")
print()
