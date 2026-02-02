import pickle
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import WeightedRandomSampler


def create_confusion_matrix_plot(y_true, y_pred, epoch, target_token, model_type="Model"):
    # 分桶 (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
    n_buckets = 10
    y_true_bucket = np.clip((y_true * n_buckets).astype(int), 0, n_buckets-1)
    y_pred_bucket = np.clip((y_pred * n_buckets).astype(int), 0, n_buckets-1)
    bucket_names = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(n_buckets)]

    cm = confusion_matrix(y_true_bucket, y_pred_bucket, labels=range(n_buckets))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=bucket_names, yticklabels=bucket_names, ax=ax)

    ax.set_xlabel('Predicted Bucket')
    ax.set_ylabel('True Bucket')
    ax.set_title(f'{model_type} - Epoch {epoch} - Token {target_token} - Confusion Matrix')

    plt.tight_layout()
    return fig

class _TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x): 
        return self.net(x).squeeze(-1)

class TokenPerformancePredictor:
    def __init__(self, embeddings_path = "data/prompt_embeddings.pkl", score_df_path = "data/Qwen3-235B.csv",
                 target_tokens = None, hidden_dims = [128, 64], dropout = 0.5,
                 train_ratio = 0.8, device = None, load_path = None):
        self.hidden_dims, self.dropout = hidden_dims, dropout
        self.train_ratio = train_ratio
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)["embeddings"]
        self.score_df = pd.read_csv(score_df_path)
        
        assert target_tokens is not None
        self.target_tokens = target_tokens

        rng = np.random.default_rng(42)
        self.permutation = rng.permutation(len(self.embeddings))
        assert len(self.embeddings) == len(self.score_df)

        self.train_set, self.test_set = self._construct_dataset()
        self.models: Dict[str, _TinyMLP] = {}

        if load_path:
            checkpoint = torch.load(load_path, map_location=self.device)
            for token in self.target_tokens:
                model = _TinyMLP(self.embeddings.shape[1], self.hidden_dims, self.dropout).to(self.device)
                model.load_state_dict(checkpoint["models"][token])
                self.models[token] = model.eval()
            print(f"Loaded models from {load_path}")

    def _construct_dataset(self):
        split = int(len(self.embeddings) * self.train_ratio)
        train_idx, test_idx = self.permutation[:split], self.permutation[split:]

        train = {"X": self.embeddings[train_idx].astype(np.float32), "y": {}}
        test = {"X": self.embeddings[test_idx].astype(np.float32), "y": {}}

        for token in self.target_tokens:
            y = self.score_df[token].values.astype(np.float32)
            train["y"][token] = y[train_idx]
            test["y"][token] = y[test_idx]
        
        return train, test

    def fit(self, save_path, lr=1e-4, batch_size=128, epochs=50, weight_decay=1e-8, 
            use_sampler=False, plot_dir="./plots"):
        os.makedirs(plot_dir, exist_ok=True)
        Xtr, Xte = self.train_set["X"], self.test_set["X"]

        for token in tqdm(self.target_tokens, desc="Training MLPs"):
            ytr, yte = self.train_set["y"][token], self.test_set["y"][token]

            # === 构造 WeightedRandomSampler (只配平 0 桶和 10 桶) ===
            ytr_cls = (np.round(ytr * 10)).astype(int)   # 离散成 11 桶 [0..10]
            class_counts = np.bincount(ytr_cls, minlength=11)
            class_weights = np.ones_like(class_counts, dtype=np.float32)

            # 只调整桶 0 和 桶 10 的权重，其余桶保持 1.0
            if class_counts[0] > 0 and class_counts[10] > 0:
                max_count = max(class_counts[0], class_counts[10])
                class_weights[0] = max_count / (class_counts[0] + 1e-6)
                class_weights[10] = max_count / (class_counts[10] + 1e-6)

            sample_weights = class_weights[ytr_cls]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            train_dataset = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
            if use_sampler:
                loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
            else:
                loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # === 模型和优化器 ===
            model = _TinyMLP(Xtr.shape[1], self.hidden_dims, self.dropout).to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.BCEWithLogitsLoss()
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

            for epoch in tqdm(range(epochs), desc=f"Token {token}", leave=False):
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

                # === 验证 ===
                model.eval()
                with torch.no_grad():
                    preds = torch.sigmoid(model(torch.from_numpy(Xte).to(self.device))).cpu().numpy()
                mse = mean_squared_error(yte, preds)
                mae = mean_absolute_error(yte, preds)
                r2 = r2_score(yte, preds)

                scheduler.step(mse)

                if epoch == epochs - 1:
                    fig = create_confusion_matrix_plot(
                        yte, preds, epoch+1, target_token=token, model_type="TokenPredictor"
                    )
                    fig.savefig(f"{plot_dir}/confmat_token{token}.png")
                    plt.close(fig)

                tqdm.write(f"[{token}] Epoch {epoch+1}: LR={optimizer.param_groups[0]['lr']:.6f}, "
                        f"MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

            self.models[token] = model.eval()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "models": {t: m.state_dict() for t, m in self.models.items()},
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "tokens": self.target_tokens
            }, save_path)
            print(f"Models saved to {save_path}")

    @torch.no_grad()
    def predict(self, indices = None, X = None) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Call fit() or load weights before predict().")

        if X is None:
            if indices is not None:
                X, index = self.embeddings[indices], indices
            else:
                X, index = self.test_set["X"], np.arange(len(self.test_set["X"]))
        else:
            index = np.arange(len(X))

        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        out = {}
        for token, model in self.models.items():
            out[token] = torch.sigmoid(model(X_tensor)).cpu().numpy()

        return pd.DataFrame(out, index=index)

if __name__ == "__main__":
    target_tokens = ['10_score', '20_score', '30_score', '40_score', '50_score', '80_score', '100_score', '150_score', '200_score', '300_score', '500_score', '800_score', '1200_score', '2000_score', '4000_score', '8000_score']
    Llama3_3B_predictor = TokenPerformancePredictor(
        dropout=0.5,
        hidden_dims=[256, 128, 64],
        embeddings_path="data/prompt_embeddings.pkl",
        score_df_path="data/Llama-3.2-3B-Instruct.csv",
        target_tokens=target_tokens,
        # load_path="./checkpoints/Llama-3.2-3B-Instruct.pt"
    )
    Llama3_3B_predictor.fit(
        lr=1e-4, batch_size=128, epochs=200, weight_decay=1e-5, use_sampler=True, 
        save_path="./checkpoints/Llama-3.2-3B-Instruct_1e4_sampler.pt",
        plot_dir="./plots/Llama-3.2-3B-Instruct_1e4_sampler"
    )
    print(Llama3_3B_predictor.predict())

    Qwen3_235B_predictor = TokenPerformancePredictor(
        dropout=0.5,
        hidden_dims=[256, 128, 64],
        embeddings_path="data/prompt_embeddings.pkl",
        score_df_path="data/Qwen3-235B-A22B-Instruct-2507.csv",
        target_tokens=target_tokens,
        # load_path="./checkpoints/Qwen3-235B-A22B-Instruct-2507.pt"
    )
    Qwen3_235B_predictor.fit(
        lr=1e-4, batch_size=128, epochs=200, weight_decay=1e-5, use_sampler=True, 
        save_path="./checkpoints/Qwen3-235B-A22B-Instruct-2507_1e4_sampler.pt",
        plot_dir="./plots/Qwen3-235B-A22B-Instruct-2507_1e4_sampler"
    )
    print(Qwen3_235B_predictor.predict())

    Qwen3_Next_80B_A3B_Instruct_predictor = TokenPerformancePredictor(
        dropout=0.5,
        hidden_dims=[256, 128, 64],
        embeddings_path="data/prompt_embeddings.pkl",
        score_df_path="data/Qwen3-Next-80B-A3B-Instruct.csv",
        target_tokens=target_tokens,
        # load_path="./checkpoints/Qwen3-235B-A22B-Instruct-2507.pt"
    )
    Qwen3_Next_80B_A3B_Instruct_predictor.fit(
        lr=1e-4, batch_size=128, epochs=200, weight_decay=1e-5, use_sampler=True, 
        save_path="./checkpoints/Qwen3-Next-80B-A3B-Instruct_1e4_sampler.pt",
        plot_dir="./plots/Qwen3-Next-80B-A3B-Instruct_1e4_sampler"
    )
    print(Qwen3_Next_80B_A3B_Instruct_predictor.predict())


    Qwen3_0_6B_predictor = TokenPerformancePredictor(
        dropout=0.5,
        hidden_dims=[256, 128, 64],
        embeddings_path="data/prompt_embeddings.pkl",
        score_df_path="data/Qwen3-0.6B.csv",
        target_tokens=target_tokens,
        # load_path="./checkpoints/Qwen3-235B-A22B-Instruct-2507.pt"
    )
    Qwen3_0_6B_predictor.fit(
        lr=1e-4, batch_size=128, epochs=200, weight_decay=1e-5, use_sampler=True, 
        save_path="./checkpoints/Qwen3-0.6B_1e4_sampler.pt",
        plot_dir="./plots/Qwen3-0.6B_1e4_sampler"
    )
    print(Qwen3_0_6B_predictor.predict())