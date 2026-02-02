import pickle
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Union
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
from ..shared.router_dataset import RouterDataset
from sklearn.linear_model import LinearRegression
from joblib import dump, load


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


def route_scores(llms, lamb_range):
    router_cost, router_perf = [], []
    llm_names = list(llms.keys())
    
    n_queries = llms[llm_names[0]]["pred_test_score"].shape[0]
    
    for lam in lamb_range:
        chosen_perf, chosen_cost = [], []
        
        for i in range(n_queries):
            best_score = -float('inf')
            best_cost = float('inf')
            best_perf = 0.0
            
            # For each LLM, find the best token limit for this query
            for llm_name in llm_names:
                llm_data = llms[llm_name]
                pred_scores = llm_data["pred_test_score"][i]  # Shape: (num_token_limits,)
                pred_tokens = llm_data["pred_test_token"][i]  # Shape: (num_token_limits,)
                
                # Calculate cost for each token limit option
                costs = pred_tokens * llm_data["size"]  # Shape: (num_token_limits,)

                # Calculate routing score for each token limit option
                # Formula: score = (1-λ)*quality - λ*cost
                # λ ∈ [0,1]: λ=0 pure quality, λ=1 pure cost minimization
                risks = (1 - lam) * pred_scores - lam * costs  # Shape: (num_token_limits,)
                
                # Find best token limit for this LLM
                best_token_idx = risks.argmax()
                
                # Update global best if this LLM+token_limit combination is better
                if risks[best_token_idx] > best_score:
                    best_score = risks[best_token_idx]
                    # Use true cost for evaluation (actual token count * model size)
                    true_tokens = llm_data["true_test_count"][i, best_token_idx]
                    best_cost = true_tokens * llm_data["size"]
                    # Get the actual performance from true_test_score
                    best_perf = llm_data["true_test_score"][i, best_token_idx]
            
            chosen_perf.append(best_perf)
            chosen_cost.append(best_cost)
        
        router_perf.append(np.mean(chosen_perf))
        router_cost.append(np.mean(chosen_cost))
    
    return np.array(router_cost), np.array(router_perf)

class TokenPerformancePredictor:
    def __init__(self, hidden_dims = [128, 64], dropout = 0.5, device = None, load_dir = None, dataset: RouterDataset = None):
        self.hidden_dims, self.dropout = hidden_dims, dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        assert dataset is not None, "dataset must be provided (paths removed)."
        self.train_set_score = dataset.get_train_set_score()
        self.test_set_score = dataset.get_test_set_score()
        self.target_tokens_score = dataset.get_target_tokens_score()

        self.train_token_unlimited_count = dataset.get_train_token_unlimited_count()
        self.test_token_unlimited_count = dataset.get_test_token_unlimited_count()

        self.models: Dict[str, _TinyMLP] = {}

        self.token_predictor = LinearRegression()

        if load_dir:
            score_path = os.path.join(load_dir, "score_predictor.pt")
            token_path = os.path.join(load_dir, "token_predictor.joblib")

            if os.path.isfile(score_path):
                ckpt = torch.load(score_path, map_location=self.device)
                for token in self.target_tokens_score:
                    model = _TinyMLP(self.train_set_score["X"].shape[1], self.hidden_dims, self.dropout).to(self.device)
                    model.load_state_dict(ckpt["models"][token])
                    self.models[token] = model.eval()
                print(f"✅ Loaded {len(self.models)} token models from {score_path}")
            else:
                print(f"⚠️ Missing: {score_path}")

            self.token_predictor = load(token_path) if os.path.isfile(token_path) else None
            print(f"{'✅ Loaded' if self.token_predictor else '⚠️ Missing'} token predictor from {token_path}")


    def fit_token_predictor(self, plot_dir="./plots"):
        self.token_predictor.fit(self.train_set_score["X"], self.train_token_unlimited_count)
        self.token_predictor.predict(self.test_set_score["X"])
        plt.figure(figsize=(8, 6))
        sns.kdeplot(self.test_token_unlimited_count, label="True", color="blue", fill=True, alpha=0.4)
        sns.kdeplot(self.token_predictor.predict(self.test_set_score["X"]), label="Pred", color="orange", fill=True, alpha=0.4)
        plt.title("Token 8000 Count Predictor")
        plt.xlabel("Token 8000 Count")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/token_8000_count_predictor.png")
        plt.close()

    def fit(self, save_dir, lr=1e-4, batch_size=128, epochs=50, weight_decay=1e-8, 
            use_sampler=False, plot_dir="./plots"):
        os.makedirs(plot_dir, exist_ok=True)
        Xtr, Xte = self.train_set_score["X"], self.test_set_score["X"]

        for token in tqdm(self.target_tokens_score, desc="Training MLPs"):
            ytr, yte = self.train_set_score["y"][token], self.test_set_score["y"][token]

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

        self.fit_token_predictor(plot_dir=plot_dir)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                "models": {t: m.state_dict() for t, m in self.models.items()},
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "tokens": self.target_tokens_score
            }, os.path.join(save_dir, "score_predictor.pt"))
            dump(self.token_predictor, os.path.join(save_dir, "token_predictor.joblib"))

    @torch.no_grad()
    def predict(self, X):
        if not self.models:
            raise RuntimeError("Call fit() or load weights before predict().")

        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        score_pred = []
        count = None
        for token in self.target_tokens_score:
            pred = torch.sigmoid(self.models[token](X_tensor)).cpu().numpy()
            score_pred.append(pred)
        count = self.token_predictor.predict(X.astype(np.float32))
        
        return np.column_stack(score_pred), count

if __name__ == "__main__":
    from dataset_manager import DatasetManager

    # Initialize centralized dataset manager
    dataset_manager = DatasetManager(
        embeddings_path="data/prompt_embeddings.pkl",
        train_ratio=0.8,
        seed=42
    )

    # Get shared embeddings and split indices
    embeddings = dataset_manager.get_embeddings()
    train_idx, test_idx = dataset_manager.get_split_indices()

    token_limits_score = ['10_score', '20_score', '30_score', '40_score', '50_score', '80_score', '100_score', '150_score', '200_score', '300_score', '500_score', '800_score', '1200_score', '2000_score', '4000_score', 'unlimited_score']

    # Create dataset with pre-computed split
    dataset = RouterDataset(
        embeddings=embeddings,
        score_df_path="data/GLM-4.6.csv",
        target_tokens_score=token_limits_score,
        train_idx=train_idx,
        test_idx=test_idx
    )

    predictor = TokenPerformancePredictor(
        hidden_dims=[256, 128, 64],
        dropout=0.5,
        dataset=dataset,
        # load_dir="./checkpoints/GLM-4.6_1e3_sampler"
    )

    predictor.fit(save_dir="./checkpoints/GLM-4.6_1e4", lr=1e-4, epochs=100, use_sampler=False, plot_dir='./plots/GLM-4.5-Air_1e4')
    predictions, count = predictor.predict(dataset.get_test_set_score()["X"])
    print(count)
    # print(f"Shape: {predictions.shape}")