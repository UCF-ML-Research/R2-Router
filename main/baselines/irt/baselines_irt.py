"""
IRT-Router Baselines: MIRT and NIRT

Two Item Response Theory-based baselines for LLM routing:
1. MIRT (Multidimensional IRT): Linear IRT model
2. NIRT (Neural IRT): Neural IRT model based on NCDM
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ...core.predictor import create_confusion_matrix_plot


class _MIRTModule(nn.Module):
    """
    Multidimensional IRT (MIRT) module.

    Models performance as:
        logit(p_ij) = a_i^T θ_j - b_i
    where:
        - θ_j is the ability vector for LLM j (latent_dim,)
        - a_i is the discrimination vector for query i (latent_dim,)
        - b_i is the difficulty scalar for query i
    """
    def __init__(self, query_dim: int, llm_dim: int, latent_dim: int):
        """
        Initialize MIRT module.

        Args:
            query_dim: Dimension of query embeddings
            llm_dim: Dimension of LLM embeddings
            latent_dim: Dimension of latent ability space
        """
        super().__init__()
        self.W_theta = nn.Linear(llm_dim, latent_dim, bias=False)  # LLM ability projection
        self.W_a = nn.Linear(query_dim, latent_dim, bias=False)    # Query discrimination
        self.w_b = nn.Linear(query_dim, 1, bias=False)             # Query difficulty

    def forward(self, X_queries: torch.Tensor, Z_llms: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X_queries: Query embeddings [N, query_dim]
            Z_llms: LLM embeddings [M, llm_dim]

        Returns:
            logits: Performance logits [N, M]
        """
        theta = self.W_theta(Z_llms)              # [M, K]
        a = self.W_a(X_queries)                   # [N, K]
        b = self.w_b(X_queries).squeeze(-1)       # [N]
        logits = a @ theta.T                      # [N, M]
        logits = logits - b[:, None]
        return logits


class _NIRTModule(nn.Module):
    """
    Neural IRT (NIRT) module based on NCDM.

    Incorporates relevance vector for interpretable multidimensional abilities.
    Models performance using a neural architecture with:
        - Relevance-weighted ability differences
        - Query-specific discrimination and difficulty
    """
    def __init__(self, query_dim: int, llm_dim: int, latent_dim: int):
        """
        Initialize NIRT module.

        Args:
            query_dim: Dimension of query embeddings
            llm_dim: Dimension of LLM embeddings
            latent_dim: Dimension of latent ability space
        """
        super().__init__()
        self.W_theta = nn.Linear(llm_dim, latent_dim, bias=False)  # LLM ability projection
        self.W_a = nn.Linear(query_dim, 1, bias=False)              # Discrimination scalar
        self.W_b = nn.Linear(query_dim, latent_dim, bias=False)    # Query difficulty vector
        self.W1 = nn.Linear(latent_dim, 1, bias=True)              # Final projection

    def forward(self, X_queries: torch.Tensor, Z_llms: torch.Tensor, r_queries: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X_queries: Query embeddings [N, query_dim]
            Z_llms: LLM embeddings [M, llm_dim]
            r_queries: Relevance vectors [N, latent_dim]

        Returns:
            logits: Performance logits [N, M]
        """
        theta = torch.sigmoid(self.W_theta(Z_llms))  # [M, K]
        a = self.W_a(X_queries)                       # [N, 1]
        b = torch.sigmoid(self.W_b(X_queries))        # [N, K]
        r = torch.softmax(r_queries, dim=-1)          # [N, K]

        # Broadcast theta for all queries: [N, M, K]
        theta_expand = theta.unsqueeze(0).expand(X_queries.size(0), -1, -1)  # [N, M, K]
        b_expand = b.unsqueeze(1).expand(-1, Z_llms.size(0), -1)             # [N, M, K]
        r_expand = r.unsqueeze(1).expand(-1, Z_llms.size(0), -1)             # [N, M, K]
        a_expand = a.unsqueeze(1)                                              # [N, M, 1]

        # x_ij = r_qi ⊙ (theta_Mj - b_i) × a_i
        x_ij = r_expand * (theta_expand - b_expand) * a_expand   # [N, M, K]

        # Pass through final layer
        logits = self.W1(x_ij).squeeze(-1)  # [N, M]

        return logits


class IRTBaseline:
    """
    MIRT-Router baseline (Multidimensional IRT).

    Uses a linear IRT model to predict LLM performance based on query embeddings
    and LLM embeddings.

    Attributes:
        llms: Dictionary of LLM data
        llm_names: List of LLM names
        latent_dim: Dimension of latent ability space
        device: Device for PyTorch computations
        Z_llms: LLM embeddings tensor
        model: Trained MIRT module
    """

    def __init__(self,
                 latent_dim: int = 32,
                 device: Optional[str] = None,
                 load_dir: Optional[str] = None):
        """
        Initialize MIRT baseline.

        Args:
            latent_dim: Dimension of latent ability space
            device: Device for PyTorch ('cuda' or 'cpu')
            load_dir: Optional directory to load pre-trained model
        """
        self.llm_names: List[str] = []
        self.latent_dim = latent_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.Z_llms: Optional[torch.Tensor] = None
        self.model: Optional[_MIRTModule] = None
        self.text_encoder_name = 'sentence-transformers/all-mpnet-base-v2'  # Default encoder

        # Load model if load_dir is provided
        if load_dir:
            self.load(load_dir)

    def fit(self,
            embedding_train: np.ndarray,
            quality_train: np.ndarray,
            llm_embeddings: np.ndarray,
            llm_names: List[str],
            lr: float = 3e-3,
            batch_size: int = 128,
            epochs: int = 200,
            weight_decay: float = 1e-6,
            save_dir: Optional[str] = None) -> 'IRTBaseline':
        """
        Train MIRT model (standard sklearn pattern).

        Args:
            embedding_train: Training query embeddings, shape (n_train, embedding_dim)
            quality_train: Training quality scores, shape (n_train, n_models)
            llm_embeddings: LLM embeddings, shape (n_models, llm_embedding_dim)
            llm_names: List of LLM names
            lr: Learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            weight_decay: Weight decay for AdamW optimizer
            save_dir: Optional directory to save trained model

        Returns:
            self (for method chaining)
        """

        # Store LLM information
        self.llm_names = llm_names
        self.Z_llms = torch.from_numpy(llm_embeddings.astype(np.float32)).to(self.device)

        X = torch.from_numpy(embedding_train.astype(np.float32)).to(self.device)
        Y = torch.from_numpy(quality_train.astype(np.float32)).to(self.device)

        query_dim = X.shape[1]
        llm_dim = self.Z_llms.shape[1]
        self.model = _MIRTModule(query_dim, llm_dim, self.latent_dim).to(self.device)

        ds = TensorDataset(X, Y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

        print(f"🚀 Training MIRT (latent_dim={self.latent_dim}, epochs={epochs})...")
        pbar = tqdm(range(epochs), desc="MIRT training")
        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.model(xb, self.Z_llms)            # [B, M]
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)

            avg_loss = epoch_loss / len(ds)
            pbar.set_postfix_str(f"LR={opt.param_groups[0]['lr']:.6f}, Loss={avg_loss:.4f}")

        print("✅ MIRT training complete")

        # Save model if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._save_model(save_dir)
            print(f"💾 Saved MIRT model to {save_dir}")

        return self

    def _save_model(self, save_dir: str) -> None:
        """Save model and configuration."""
        # Save model state dict
        torch.save(self.model.state_dict(), os.path.join(save_dir, "mirt_model.pt"))

        # Save configuration
        config = {
            'latent_dim': self.latent_dim,
            'llm_names': self.llm_names,
            'text_encoder_name': self.text_encoder_name,
            'query_dim': self.model.W_a.in_features,
            'llm_dim': self.Z_llms.shape[1]
        }
        torch.save(config, os.path.join(save_dir, "mirt_config.pt"))

        # Save LLM embeddings
        torch.save(self.Z_llms.cpu(), os.path.join(save_dir, "mirt_llm_embeddings.pt"))

    def load(self, load_dir: str) -> 'IRTBaseline':
        """
        Load pre-trained MIRT model.

        Args:
            load_dir: Directory containing saved model

        Returns:
            self
        """
        model_path = os.path.join(load_dir, "mirt_model.pt")
        config_path = os.path.join(load_dir, "mirt_config.pt")
        embeddings_path = os.path.join(load_dir, "mirt_llm_embeddings.pt")

        if not os.path.isfile(model_path):
            print(f"⚠️  Missing: {model_path}")
            return self

        if not os.path.isfile(config_path):
            print(f"⚠️  Missing: {config_path}")
            return self

        # Load configuration
        config = torch.load(config_path, map_location=self.device)
        self.latent_dim = config['latent_dim']
        self.llm_names = config['llm_names']
        self.text_encoder_name = config.get('text_encoder_name', 'sentence-transformers/all-mpnet-base-v2')

        # Load LLM embeddings
        if os.path.isfile(embeddings_path):
            self.Z_llms = torch.load(embeddings_path, map_location=self.device)

        # Initialize model
        self.model = _MIRTModule(
            query_dim=config['query_dim'],
            llm_dim=config['llm_dim'],
            latent_dim=self.latent_dim
        ).to(self.device)

        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"✅ Loaded MIRT model from {load_dir}")

        return self

    @torch.no_grad()
    def predict(self, embedding: np.ndarray) -> np.ndarray:
        """
        Predict LLM performance scores (standard sklearn pattern).

        Args:
            embedding: Query embeddings to predict on, shape (n_queries, embedding_dim)

        Returns:
            predictions: Predicted quality scores, shape (n_queries, n_llms)
        """
        assert self.model is not None, "Call fit() or load() first"

        self.model.eval()
        X_t = torch.from_numpy(embedding.astype(np.float32)).to(self.device)
        logits = self.model(X_t, self.Z_llms)
        preds = torch.sigmoid(logits).cpu().numpy()
        return preds


class NIRTBaseline:
    """
    NIRT-Router baseline (Neural IRT).

    Uses a neural IRT model with relevance vectors to predict LLM performance
    based on query embeddings and LLM embeddings.

    Attributes:
        llms: Dictionary of LLM data
        llm_names: List of LLM names
        latent_dim: Dimension of latent ability space
        device: Device for PyTorch computations
        Z_llms: LLM embeddings tensor
        model: Trained NIRT module
        r_test: Relevance vectors for test queries
    """

    def __init__(self,
                 latent_dim: int = 32,
                 device: Optional[str] = None,
                 load_dir: Optional[str] = None):
        """
        Initialize NIRT baseline.

        Args:
            latent_dim: Dimension of latent ability space
            device: Device for PyTorch ('cuda' or 'cpu')
            load_dir: Optional directory to load pre-trained model
        """
        self.llm_names: List[str] = []
        self.latent_dim = latent_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.Z_llms: Optional[torch.Tensor] = None
        self.model: Optional[_NIRTModule] = None
        self.r_train: Optional[np.ndarray] = None
        self._pca: Optional[PCA] = None
        self.text_encoder_name = 'sentence-transformers/all-mpnet-base-v2'  # Default encoder

        # Load model if load_dir is provided
        if load_dir:
            self.load(load_dir)

    def _generate_relevance_vectors(self, X: np.ndarray) -> np.ndarray:
        """
        Generate relevance vectors for queries.

        Uses PCA to project query embeddings to latent space, then normalizes
        to create soft relevance distributions over latent dimensions.

        Args:
            X: Query embeddings, shape (n_queries, embedding_dim)

        Returns:
            r: Relevance vectors, shape (n_queries, latent_dim)
        """
        if self._pca is None:
            # Determine the actual number of components we can use
            n_samples, n_features = X.shape
            actual_components = min(n_samples, n_features, self.latent_dim)

            if actual_components < self.latent_dim:
                print(f"⚠️  Reducing latent_dim from {self.latent_dim} to {actual_components} "
                      f"due to data constraints (n_samples={n_samples}, n_features={n_features})")
                # Update the actual latent_dim to use
                self.latent_dim = actual_components

            self._pca = PCA(n_components=actual_components)
            self._pca.fit(X)

        # Transform to latent space and normalize to get relevance vectors
        r = self._pca.transform(X)

        # Apply softmax-like normalization to get positive weights summing to 1
        r = np.exp(r - r.max(axis=1, keepdims=True))
        r = r / r.sum(axis=1, keepdims=True)
        return r.astype(np.float32)

    def fit(self,
            embedding_train: np.ndarray,
            quality_train: np.ndarray,
            llm_embeddings: np.ndarray,
            llm_names: List[str],
            lr: float = 3e-3,
            batch_size: int = 128,
            epochs: int = 200,
            weight_decay: float = 1e-6,
            save_dir: Optional[str] = None) -> 'NIRTBaseline':
        """
        Train NIRT model (standard sklearn pattern).

        Args:
            embedding_train: Training query embeddings, shape (n_train, embedding_dim)
            quality_train: Training quality scores, shape (n_train, n_models)
            llm_embeddings: LLM embeddings, shape (n_models, llm_embedding_dim)
            llm_names: List of LLM names
            lr: Learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            weight_decay: Weight decay for AdamW optimizer
            save_dir: Optional directory to save trained model

        Returns:
            self (for method chaining)
        """
        # Store LLM information
        self.llm_names = llm_names
        self.Z_llms = torch.from_numpy(llm_embeddings.astype(np.float32)).to(self.device)

        print(f"NIRT Training data shape: {embedding_train.shape}")
        print(f"NIRT Requested latent_dim: {self.latent_dim}")

        # Generate relevance vectors for queries
        r_train = self._generate_relevance_vectors(embedding_train)

        print(f"NIRT Relevance vector shape (train): {r_train.shape}")

        X = torch.from_numpy(embedding_train.astype(np.float32)).to(self.device)
        Y = torch.from_numpy(quality_train.astype(np.float32)).to(self.device)
        R = torch.from_numpy(r_train).to(self.device)

        query_dim = X.shape[1]
        llm_dim = self.Z_llms.shape[1]
        print(f"NIRT Initializing model with latent_dim: {self.latent_dim}")
        self.model = _NIRTModule(query_dim, llm_dim, self.latent_dim).to(self.device)

        ds = TensorDataset(X, Y, R)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        print(f"🚀 Training NIRT (latent_dim={self.latent_dim}, epochs={epochs})...")
        pbar = tqdm(range(epochs), desc="NIRT training")
        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0
            for xb, yb, rb in loader:
                opt.zero_grad()
                logits = self.model(xb, self.Z_llms, rb)  # [B, M]
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)

            avg_loss = epoch_loss / len(ds)
            pbar.set_postfix_str(f"LR={opt.param_groups[0]['lr']:.6f}, Loss={avg_loss:.4f}")

        print("✅ NIRT training complete")

        # Store training relevance vectors (needed for PCA state)
        self.r_train = r_train

        # Save model if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._save_model(save_dir)
            print(f"💾 Saved NIRT model to {save_dir}")

        return self

    def _save_model(self, save_dir: str) -> None:
        """Save model and configuration."""
        # Save model state dict
        torch.save(self.model.state_dict(), os.path.join(save_dir, "nirt_model.pt"))

        # Save configuration
        config = {
            'latent_dim': self.latent_dim,
            'llm_names': self.llm_names,
            'text_encoder_name': self.text_encoder_name,
            'query_dim': self.model.W_a.in_features,
            'llm_dim': self.Z_llms.shape[1]
        }
        torch.save(config, os.path.join(save_dir, "nirt_config.pt"))

        # Save LLM embeddings
        torch.save(self.Z_llms.cpu(), os.path.join(save_dir, "nirt_llm_embeddings.pt"))

        # Save PCA for relevance vector generation
        if self._pca is not None:
            import pickle
            with open(os.path.join(save_dir, "nirt_pca.pkl"), 'wb') as f:
                pickle.dump(self._pca, f)

    def load(self, load_dir: str) -> 'NIRTBaseline':
        """
        Load pre-trained NIRT model.

        Args:
            load_dir: Directory containing saved model

        Returns:
            self
        """
        model_path = os.path.join(load_dir, "nirt_model.pt")
        config_path = os.path.join(load_dir, "nirt_config.pt")
        embeddings_path = os.path.join(load_dir, "nirt_llm_embeddings.pt")
        pca_path = os.path.join(load_dir, "nirt_pca.pkl")

        if not os.path.isfile(model_path):
            print(f"⚠️  Missing: {model_path}")
            return self

        if not os.path.isfile(config_path):
            print(f"⚠️  Missing: {config_path}")
            return self

        # Load configuration
        config = torch.load(config_path, map_location=self.device)
        self.latent_dim = config['latent_dim']
        self.llm_names = config['llm_names']
        self.text_encoder_name = config.get('text_encoder_name', 'sentence-transformers/all-mpnet-base-v2')

        # Load LLM embeddings
        if os.path.isfile(embeddings_path):
            self.Z_llms = torch.load(embeddings_path, map_location=self.device)

        # Load PCA
        if os.path.isfile(pca_path):
            import pickle
            with open(pca_path, 'rb') as f:
                self._pca = pickle.load(f)

        # Initialize model
        self.model = _NIRTModule(
            query_dim=config['query_dim'],
            llm_dim=config['llm_dim'],
            latent_dim=self.latent_dim
        ).to(self.device)

        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"✅ Loaded NIRT model from {load_dir}")

        return self

    @torch.no_grad()
    def predict(self, embedding: np.ndarray) -> np.ndarray:
        """
        Predict LLM performance scores (standard sklearn pattern).

        Args:
            embedding: Query embeddings to predict on, shape (n_queries, embedding_dim)

        Returns:
            predictions: Predicted quality scores, shape (n_queries, n_llms)
        """
        assert self.model is not None, "Call fit() or load() first"

        # Generate relevance vectors for new embeddings
        r = self._generate_relevance_vectors(embedding)

        self.model.eval()
        X_t = torch.from_numpy(embedding.astype(np.float32)).to(self.device)
        R_t = torch.from_numpy(r).to(self.device)
        logits = self.model(X_t, self.Z_llms, R_t)
        preds = torch.sigmoid(logits).cpu().numpy()
        return preds


# Backward compatibility - keep old class names
MIRTBaseline = IRTBaseline


if __name__ == "__main__":
    import pandas as pd
    from dataset_manager import DatasetManager
    from sentence_transformers import SentenceTransformer

    # Initialize centralized dataset manager
    print("=" * 80)
    print("INITIALIZING DATASET MANAGER")
    print("=" * 80)
    dataset_manager = DatasetManager(
        embeddings_path="data/prompt_embeddings.pkl",
        train_ratio=0.8,
        seed=42
    )

    embeddings = dataset_manager.get_embeddings()
    train_idx, test_idx = dataset_manager.get_split_indices()

    # Define models to load
    models_to_load = [
        {"name": "GLM-4.5-Air", "csv": "data/GLM-4.5-Air.csv"},
        {"name": "GLM-4.6", "csv": "data/GLM-4.6.csv"},
        {"name": "gemma-3-4b-it", "csv": "data/gemma-3-4b-it.csv"},
        {"name": "Llama-3.1-70B-Instruct", "csv": "data/Llama-3.1-70B-Instruct.csv"},
        {"name": "Llama-3.2-3B-Instruct", "csv": "data/Llama-3.2-3B-Instruct.csv"},
        {"name": "Qwen2.5-Math-1.5B-Instruct", "csv": "data/Qwen2.5-Math-1.5B-Instruct.csv"},
        {"name": "Qwen2.5-Math-7B-Instruct", "csv": "data/Qwen2.5-Math-7B-Instruct.csv"},
        {"name": "Qwen3-0.6B", "csv": "data/Qwen3-0.6B.csv"},
        {"name": "Qwen3-235B-A22B-Instruct-2507", "csv": "data/Qwen3-235B-A22B-Instruct-2507.csv"},
        {"name": "Qwen3-Next-80B-A3B-Instruct", "csv": "data/Qwen3-Next-80B-A3B-Instruct.csv"},
    ]

    # Load unlimited scores from all models (IRT only needs quality, not token counts)
    print("\n" + "=" * 80)
    print("LOADING MODEL DATA (unlimited quality only)")
    print("=" * 80)

    quality_train_list = []
    quality_test_list = []
    model_names_loaded = []

    for model_config in models_to_load:
        model_name = model_config["name"]
        csv_path = model_config["csv"]

        if not os.path.exists(csv_path):
            print(f"⚠️  Skipping {model_name}: CSV file not found")
            continue

        print(f"Loading {model_name}...")

        # Load CSV directly and extract unlimited quality
        df = pd.read_csv(csv_path)

        # Get unlimited quality scores
        train_quality = df.loc[train_idx, 'unlimited_score'].values
        test_quality = df.loc[test_idx, 'unlimited_score'].values

        quality_train_list.append(train_quality)
        quality_test_list.append(test_quality)
        model_names_loaded.append(model_name)

    # Stack into matrices (n_queries, n_models)
    embedding_train = embeddings[train_idx]
    embedding_test = embeddings[test_idx]
    quality_train = np.column_stack(quality_train_list)
    quality_test = np.column_stack(quality_test_list)

    print(f"\n✅ Loaded {len(model_names_loaded)} models")
    print(f"Training embeddings shape: {embedding_train.shape}")
    print(f"Training quality shape: {quality_train.shape}")

    # Generate LLM embeddings using sentence transformers
    print("\n" + "=" * 80)
    print("GENERATING LLM EMBEDDINGS")
    print("=" * 80)

    llm_texts = {name: f"Language model {name}" for name in model_names_loaded}
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    llm_embeddings = encoder.encode(
        [llm_texts[name] for name in model_names_loaded],
        convert_to_numpy=True,
        normalize_embeddings=False
    ).astype(np.float32)

    print(f"LLM embeddings shape: {llm_embeddings.shape}")

    # Train MIRT
    print("\n" + "=" * 80)
    print("TRAINING MIRT (latent_dim=32)")
    print("=" * 80)

    irt = IRTBaseline(latent_dim=32, device="cuda" if torch.cuda.is_available() else "cpu")
    irt.fit(
        embedding_train=embedding_train,
        quality_train=quality_train,
        llm_embeddings=llm_embeddings,
        llm_names=model_names_loaded,
        embedding_test=embedding_test,
        quality_test=quality_test,
        lr=3e-3,
        batch_size=128,
        epochs=50,  # Reduced for faster training
        save_dir="./checkpoints/irt_mirt",
        plot_dir="./plots/irt_mirt"
    )

    # Test prediction
    Y_hat_score = irt.predict()
    print(f"✅ MIRT prediction shape: {Y_hat_score.shape}")

    # Train NIRT
    print("\n" + "=" * 80)
    print("TRAINING NIRT (latent_dim=32)")
    print("=" * 80)

    nirt = NIRTBaseline(latent_dim=32, device="cuda" if torch.cuda.is_available() else "cpu")
    nirt.fit(
        embedding_train=embedding_train,
        quality_train=quality_train,
        llm_embeddings=llm_embeddings,
        llm_names=model_names_loaded,
        embedding_test=embedding_test,
        quality_test=quality_test,
        lr=3e-3,
        batch_size=128,
        epochs=50,  # Reduced for faster training
        save_dir="./checkpoints/irt_nirt",
        plot_dir="./plots/irt_nirt"
    )

    # Test prediction
    Y_hat_score2 = nirt.predict()
    print(f"✅ NIRT prediction shape: {Y_hat_score2.shape}")

    print("\n" + "=" * 80)
    print("ALL IRT BASELINES TRAINED SUCCESSFULLY!")
    print("=" * 80)
