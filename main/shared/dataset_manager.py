"""
Centralized dataset manager for handling train/test splits across all LLMs.
This ensures all models use the same split for fair comparison.
"""
import pickle
import numpy as np
from typing import Tuple, Optional


class DatasetManager:
    """Manages global train/test splits for the router dataset."""

    def __init__(self, embeddings_path: str, train_ratio: float = 0.8, seed: int = 42):
        """
        Initialize the dataset manager with embeddings and create train/test split.

        Args:
            embeddings_path: Path to the pickle file containing embeddings
            train_ratio: Ratio of training data (default 0.8)
            seed: Random seed for reproducibility (default 42)
        """
        with open(embeddings_path, "rb") as f:
            emb_data = pickle.load(f)
            if isinstance(emb_data, dict):
                self.embeddings = emb_data["embeddings"]
            else:
                self.embeddings = emb_data

        self.train_ratio = float(train_ratio)
        self.seed = seed

        # Create train/test split once for all LLMs
        rng = np.random.default_rng(seed)
        self.permutation = rng.permutation(len(self.embeddings))

        split_idx = int(len(self.embeddings) * train_ratio)
        self.train_idx = self.permutation[:split_idx]
        self.test_idx = self.permutation[split_idx:]

        print(f"DatasetManager initialized:")
        print(f"  Total samples: {len(self.embeddings)}")
        print(f"  Train samples: {len(self.train_idx)}")
        print(f"  Test samples: {len(self.test_idx)}")
        print(f"  Train ratio: {train_ratio}")
        print(f"  Random seed: {seed}")

    def get_embeddings(self) -> np.ndarray:
        """Return the full embeddings array."""
        return self.embeddings

    def get_train_embeddings(self) -> np.ndarray:
        """Return training embeddings."""
        return self.embeddings[self.train_idx]

    def get_test_embeddings(self) -> np.ndarray:
        """Return test embeddings."""
        return self.embeddings[self.test_idx]

    def get_split_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return train and test indices."""
        return self.train_idx, self.test_idx

    def get_train_idx(self) -> np.ndarray:
        """Return training indices."""
        return self.train_idx

    def get_test_idx(self) -> np.ndarray:
        """Return test indices."""
        return self.test_idx
