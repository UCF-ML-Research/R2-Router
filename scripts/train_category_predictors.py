#!/usr/bin/env python3
"""
Train per-category quality and token count predictors for category-aware R2-Router.

Compares 3 architectures per (category x model) using K-fold CV:
  A) Independent Ridge: one Ridge(alpha=10) per budget head + LinearRegression for tokens
  B) MultiHeadMLP (PyTorch): shared backbone -> N quality heads + 1 token head
  C) KNN: KNeighborsRegressor(cosine) per budget head + KNeighborsRegressor for tokens

Selects best architecture per (category x model) based on mean quality R2 across folds.
Then trains final model on full training set.

Requires: training_data.pkl from build_category_training_data.py

Usage:
    .venv/bin/python scripts/train_category_predictors.py [--epochs 200] [--hidden 128] [--k 5]
"""

import os
import sys
import json
import pickle
import numpy as np
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from category_config import (
    TRAINING_DATA_PATH, CHECKPOINT_DIR,
    CATEGORY_NAMES, NUM_CATEGORIES,
    MODELS, get_budgets,
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Architecture B: MultiHeadMLP (PyTorch)
# ============================================================================

class MultiHeadMLP(nn.Module):
    """Shared backbone -> N quality heads + 1 token head."""
    def __init__(self, input_dim: int, hidden_dim: int, num_quality_heads: int,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Quality heads: one per budget
        self.quality_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_quality_heads)
        ])
        # Token head: one for predicting output tokens
        self.token_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared = self.backbone(x)
        quality_out = torch.cat([h(shared) for h in self.quality_heads], dim=1)
        token_out = self.token_head(shared)
        return quality_out, token_out


def train_mlp_fold(X_train, Y_quality_train, y_token_train,
                   X_val, Y_quality_val, y_token_val,
                   hidden_dim=128, epochs=200, lr=1e-3, batch_size=256,
                   device="cpu"):
    """Train MultiHeadMLP on one fold and return validation metrics.

    Returns: quality_r2_per_head (list), token_r2 (float)
    """
    input_dim = X_train.shape[1]
    num_heads = Y_quality_train.shape[1]

    model = MultiHeadMLP(input_dim, hidden_dim, num_heads, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Normalize token targets for balanced loss
    token_scale = max(y_token_train.std(), 1.0)
    y_token_norm_train = y_token_train / token_scale

    # Tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    Yq_tr = torch.tensor(Y_quality_train, dtype=torch.float32)
    Yt_tr = torch.tensor(y_token_norm_train, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tr, Yq_tr, Yt_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    quality_loss_fn = nn.MSELoss()
    token_loss_fn = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(epochs):
        for xb, yqb, ytb in loader:
            xb, yqb, ytb = xb.to(device), yqb.to(device), ytb.to(device)
            q_pred, t_pred = model(xb)
            loss = quality_loss_fn(q_pred, yqb) + 0.1 * token_loss_fn(t_pred, ytb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluate
    model.eval()
    X_te = torch.tensor(X_val, dtype=torch.float32).to(device)
    with torch.no_grad():
        q_preds, t_pred = model(X_te)
        q_preds = q_preds.cpu().numpy()
        t_pred = t_pred.cpu().numpy().squeeze() * token_scale

    quality_r2s = []
    for h in range(num_heads):
        y_true = Y_quality_val[:, h]
        y_pred = q_preds[:, h]
        r2 = r2_score(y_true, y_pred) if y_true.var() > 1e-10 else 0.0
        quality_r2s.append(r2)

    token_r2 = r2_score(y_token_val, t_pred) if y_token_val.var() > 1e-10 else 0.0

    return quality_r2s, token_r2, model, token_scale


def train_mlp_final(X_train, Y_quality_train, y_token_train,
                    hidden_dim=128, epochs=200, lr=1e-3, batch_size=256,
                    device="cpu"):
    """Train MultiHeadMLP on full training set. Returns model + token_scale."""
    input_dim = X_train.shape[1]
    num_heads = Y_quality_train.shape[1]

    model = MultiHeadMLP(input_dim, hidden_dim, num_heads, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    token_scale = max(y_token_train.std(), 1.0)
    y_token_norm = y_token_train / token_scale

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    Yq_tr = torch.tensor(Y_quality_train, dtype=torch.float32)
    Yt_tr = torch.tensor(y_token_norm, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tr, Yq_tr, Yt_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    quality_loss_fn = nn.MSELoss()
    token_loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for xb, yqb, ytb in loader:
            xb, yqb, ytb = xb.to(device), yqb.to(device), ytb.to(device)
            q_pred, t_pred = model(xb)
            loss = quality_loss_fn(q_pred, yqb) + 0.1 * token_loss_fn(t_pred, ytb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model, token_scale


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--k", type=int, default=5, help="K for K-fold CV")
    parser.add_argument("--pca", type=int, default=0,
                        help="PCA components per category (0=disabled, e.g. 64)")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Ridge regularization alpha (default: 10.0)")
    parser.add_argument("--ridge-only", action="store_true",
                        help="Force Ridge only (skip MLP/KNN architecture search)")
    parser.add_argument("--knn-only", action="store_true",
                        help="Force KNN only (skip Ridge/MLP architecture search)")
    parser.add_argument("--logistic", action="store_true",
                        help="Use LogisticRegression instead of Ridge for quality prediction (binary targets)")
    parser.add_argument("--full", action="store_true",
                        help="Train final model on ALL data (no held-out test set)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for held-out test (default: 0.2)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"=== Category-Aware Predictor Training (K={args.k}-fold CV) ===")
    log(f"Device: {device}, Epochs: {args.epochs}, Hidden: {args.hidden}, Ridge alpha: {args.alpha}")
    if args.ridge_only:
        log(f"RIDGE-ONLY MODE: skipping MLP/KNN architecture search")
    if args.knn_only:
        log(f"KNN-ONLY MODE: skipping Ridge/MLP architecture search")
    if args.logistic:
        log(f"LOGISTIC MODE: using LogisticRegression for quality (binary classification)")
    if args.pca > 0:
        log(f"PCA: {args.pca} components per category")
    if args.full:
        log(f"FULL MODE: training final model on ALL data (no held-out test)")
    log("")

    # 1. Load training data
    log("Loading training data...")
    with open(TRAINING_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"]      # (8400, 1024)
    categories = data["categories"]       # (8400,) int
    models_data = data["models"]          # {model: {budget: {accuracy, output_tokens}}}
    n_queries = embeddings.shape[0]
    log(f"  Queries: {n_queries}, Dim: {embeddings.shape[1]}")
    log(f"  Models: {list(models_data.keys())}")

    # 2. Global train/test split (80/20, stratified, seed=42)
    # K-fold CV is done within the training set.
    # --full mode: use all data for final training, but still CV for arch selection.
    all_idx = np.arange(n_queries)
    if args.full:
        # In full mode, all data is "training" for the final model.
        # We still do CV within it for architecture selection.
        train_idx = all_idx.copy()
        test_idx = np.array([], dtype=int)
        train_set = set(train_idx.tolist())
        log(f"  FULL MODE: {len(train_idx)} train, 0 test (all data)")
    else:
        train_idx, test_idx = train_test_split(
            all_idx, test_size=args.test_size, random_state=42, stratify=categories,
        )
        train_set = set(train_idx.tolist())
        log(f"  Split: {len(train_idx)} train, {len(test_idx)} test (held-out, test_size={args.test_size})")

    # Save split
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    split_path = os.path.join(CHECKPOINT_DIR, "train_test_split.pkl")
    with open(split_path, "wb") as f:
        pickle.dump({"train_idx": train_idx, "test_idx": test_idx}, f)

    # 3. K-fold CV + final training per (category x model)
    K = args.k
    results_summary = {}
    best_choices = {}  # (cat, model) -> best architecture name
    global_ridge_wins = 0
    global_mlp_wins = 0
    global_knn_wins = 0

    for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
        log(f"\n{'='*70}")
        log(f"Category {cat_idx}: {cat_name}")
        log(f"{'='*70}")

        cat_all = np.where(categories == cat_idx)[0]
        cat_train = np.array([i for i in cat_all if i in train_set])
        cat_test = np.array([i for i in cat_all if i not in train_set])
        log(f"  Train: {len(cat_train)}, Test: {len(cat_test)}")

        if len(cat_train) < 20:
            log(f"  SKIP: too few samples")
            continue

        X_cat_train = embeddings[cat_train]
        X_cat_test = embeddings[cat_test] if len(cat_test) > 0 else np.zeros((0, embeddings.shape[1]))

        # Optional PCA dimensionality reduction
        if args.pca > 0:
            n_components = min(args.pca, X_cat_train.shape[0] - 1, X_cat_train.shape[1])
            pca = PCA(n_components=n_components, random_state=42)
            X_cat_train = pca.fit_transform(X_cat_train)
            if len(X_cat_test) > 0:
                X_cat_test = pca.transform(X_cat_test)
            log(f"  PCA: {embeddings.shape[1]} -> {n_components} (explained var: {pca.explained_variance_ratio_.sum():.3f})")

        cat_dir = os.path.join(CHECKPOINT_DIR, "predictors", cat_name)
        os.makedirs(cat_dir, exist_ok=True)

        # Save PCA transformer if used
        if args.pca > 0:
            dump(pca, os.path.join(cat_dir, "pca.joblib"))

        results_summary[cat_name] = {
            "n_train": len(cat_train), "n_test": len(cat_test), "models": {}
        }

        # Category-level labels for stratified K-fold within category
        # Use a dummy label (all same category) — just use KFold, not StratifiedKFold
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=min(K, len(cat_train)), shuffle=True, random_state=42)

        for model_name in MODELS:
            if model_name not in models_data:
                continue

            model_budgets_data = models_data[model_name]
            budgets = get_budgets(model_name)

            # Collect valid budgets (have accuracy data)
            valid_budgets = []
            for b in budgets:
                if b in model_budgets_data:
                    y = model_budgets_data[b]["accuracy"][cat_train]
                    if y.sum() > 0:
                        valid_budgets.append(b)

            if not valid_budgets:
                continue

            num_heads = len(valid_budgets)

            # Build quality target matrices for this category subset
            Y_quality_train = np.column_stack([
                model_budgets_data[b]["accuracy"][cat_train] for b in valid_budgets
            ])
            if len(cat_test) > 0:
                Y_quality_test = np.column_stack([
                    model_budgets_data[b]["accuracy"][cat_test] for b in valid_budgets
                ])
            else:
                Y_quality_test = np.zeros((0, num_heads))

            # Token targets (from concise setting)
            token_budget = "concise"
            has_token = (token_budget in model_budgets_data and
                         model_budgets_data[token_budget]["output_tokens"][cat_train].sum() > 0)
            if has_token:
                y_token_train = model_budgets_data[token_budget]["output_tokens"][cat_train]
                y_token_test = model_budgets_data[token_budget]["output_tokens"][cat_test] if len(cat_test) > 0 else np.zeros(0, dtype=np.float32)
            else:
                y_token_train = np.zeros(len(cat_train), dtype=np.float32)
                y_token_test = np.zeros(len(cat_test), dtype=np.float32)

            # KNN K value: adapt to category training size
            # Use sqrt(n_train_fold) capped at 256, minimum 3
            knn_k = min(256, max(3, int(np.sqrt(len(cat_train) * (K - 1) / K))))

            # --- K-fold CV ---
            ridge_cv_r2s = []
            mlp_cv_r2s = []
            knn_cv_r2s = []
            ridge_token_cv_r2s = []
            mlp_token_cv_r2s = []
            knn_token_cv_r2s = []

            for fold_idx, (fold_train, fold_val) in enumerate(kf.split(X_cat_train)):
                Xf_train = X_cat_train[fold_train]
                Xf_val = X_cat_train[fold_val]
                Yqf_train = Y_quality_train[fold_train]
                Yqf_val = Y_quality_train[fold_val]
                ytf_train = y_token_train[fold_train]
                ytf_val = y_token_train[fold_val]

                # Ensure KNN K doesn't exceed fold train size
                fold_knn_k = min(knn_k, len(fold_train) - 1)

                # Ridge/Logistic: per-budget quality + LinearRegression token
                if not args.knn_only:
                    fold_ridge_r2s = []
                    for h in range(num_heads):
                        if args.logistic:
                            y_bin_train = (Yqf_train[:, h] > 0.5).astype(int)
                            y_bin_val = (Yqf_val[:, h] > 0.5).astype(int)
                            # Need both classes in training
                            if y_bin_train.sum() == 0 or y_bin_train.sum() == len(y_bin_train):
                                fold_ridge_r2s.append(0.0)
                                continue
                            clf = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs')
                            clf.fit(Xf_train, y_bin_train)
                            y_prob = clf.predict_proba(Xf_val)[:, 1]
                            # Use AUC-ROC as CV metric (more appropriate for classification)
                            if y_bin_val.sum() > 0 and y_bin_val.sum() < len(y_bin_val):
                                auc = roc_auc_score(y_bin_val, y_prob)
                                fold_ridge_r2s.append(auc)
                            else:
                                fold_ridge_r2s.append(0.5)
                        else:
                            reg = Ridge(alpha=args.alpha)
                            reg.fit(Xf_train, Yqf_train[:, h])
                            y_pred = reg.predict(Xf_val)
                            r2 = r2_score(Yqf_val[:, h], y_pred) if Yqf_val[:, h].var() > 1e-10 else 0.0
                            fold_ridge_r2s.append(r2)
                    ridge_cv_r2s.append(np.mean(fold_ridge_r2s))

                    if has_token and ytf_train.sum() > 0:
                        treg = LinearRegression()
                        treg.fit(Xf_train, ytf_train)
                        tp = treg.predict(Xf_val)
                        tr2 = r2_score(ytf_val, tp) if ytf_val.var() > 1e-10 else 0.0
                        ridge_token_cv_r2s.append(tr2)

                if not args.ridge_only:
                    # KNN: per-budget quality + KNN token
                    fold_knn_r2s = []
                    for h in range(num_heads):
                        knn = KNeighborsRegressor(
                            n_neighbors=fold_knn_k, metric="cosine", weights="distance"
                        )
                        knn.fit(Xf_train, Yqf_train[:, h])
                        y_pred = knn.predict(Xf_val)
                        r2 = r2_score(Yqf_val[:, h], y_pred) if Yqf_val[:, h].var() > 1e-10 else 0.0
                        fold_knn_r2s.append(r2)
                    knn_cv_r2s.append(np.mean(fold_knn_r2s))

                    if has_token and ytf_train.sum() > 0:
                        tknn = KNeighborsRegressor(
                            n_neighbors=fold_knn_k, metric="cosine", weights="distance"
                        )
                        tknn.fit(Xf_train, ytf_train)
                        tp = tknn.predict(Xf_val)
                        tr2 = r2_score(ytf_val, tp) if ytf_val.var() > 1e-10 else 0.0
                        knn_token_cv_r2s.append(tr2)

                if not args.ridge_only and not args.knn_only:
                    # MLP: shared backbone + quality heads + token head
                    q_r2s, t_r2, _, _ = train_mlp_fold(
                        Xf_train, Yqf_train, ytf_train,
                        Xf_val, Yqf_val, ytf_val,
                        hidden_dim=args.hidden, epochs=args.epochs, device=device,
                    )
                    mlp_cv_r2s.append(np.mean(q_r2s))
                    mlp_token_cv_r2s.append(t_r2)

            ridge_mean_r2 = np.mean(ridge_cv_r2s) if ridge_cv_r2s else 0.0
            mlp_mean_r2 = np.mean(mlp_cv_r2s) if mlp_cv_r2s else 0.0
            knn_mean_r2 = np.mean(knn_cv_r2s) if knn_cv_r2s else 0.0

            # Pick best architecture
            if args.ridge_only:
                best_arch = "Ridge"
            elif args.knn_only:
                best_arch = "KNN"
            else:
                arch_scores = {"Ridge": ridge_mean_r2, "MLP": mlp_mean_r2, "KNN": knn_mean_r2}
                best_arch = max(arch_scores, key=arch_scores.get)
            best_choices[(cat_name, model_name)] = best_arch

            if best_arch == "Ridge":
                global_ridge_wins += 1
            elif best_arch == "MLP":
                global_mlp_wins += 1
            else:
                global_knn_wins += 1

            if args.ridge_only:
                metric_name = "AUC" if args.logistic else "R2"
                arch_name = "Logistic" if args.logistic else "Ridge"
                log(f"  {model_name:<15} heads={num_heads:>2}  {arch_name} {metric_name}={ridge_mean_r2:.4f}  -> {arch_name}")
            elif args.knn_only:
                log(f"  {model_name:<15} heads={num_heads:>2}  k={knn_k:>3}  KNN R2={knn_mean_r2:.4f}  -> KNN")
            else:
                log(f"  {model_name:<15} heads={num_heads:>2}  k={knn_k:>3}  "
                    f"Ridge={ridge_mean_r2:.4f}  "
                    f"MLP={mlp_mean_r2:.4f}  "
                    f"KNN={knn_mean_r2:.4f}  "
                    f"-> {best_arch}")

            # --- Train final model on full category training set ---
            final_knn_k = min(knn_k, len(cat_train) - 1)

            if best_arch == "Ridge":
                arch_label = "Logistic" if args.logistic else "Ridge"
                for h, b in enumerate(valid_budgets):
                    if args.logistic:
                        y_bin = (Y_quality_train[:, h] > 0.5).astype(int)
                        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                            # Degenerate case: all same class, fall back to Ridge
                            reg = Ridge(alpha=args.alpha)
                            reg.fit(X_cat_train, Y_quality_train[:, h])
                        else:
                            reg = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs')
                            reg.fit(X_cat_train, y_bin)
                    else:
                        reg = Ridge(alpha=args.alpha)
                        reg.fit(X_cat_train, Y_quality_train[:, h])
                    dump(reg, os.path.join(cat_dir, f"{model_name}_{b}_quality.joblib"))

                if has_token:
                    token_reg = LinearRegression()
                    token_reg.fit(X_cat_train, y_token_train)
                    dump(token_reg, os.path.join(cat_dir, f"{model_name}_token.joblib"))

                meta = {"architecture": arch_label, "budgets": valid_budgets,
                        "cv_mean_r2": float(ridge_mean_r2)}
                with open(os.path.join(cat_dir, f"{model_name}_quality_meta.json"), "w") as f:
                    json.dump(meta, f)

            elif best_arch == "KNN":
                for h, b in enumerate(valid_budgets):
                    knn = KNeighborsRegressor(
                        n_neighbors=final_knn_k, metric="cosine", weights="distance"
                    )
                    knn.fit(X_cat_train, Y_quality_train[:, h])
                    dump(knn, os.path.join(cat_dir, f"{model_name}_{b}_quality.joblib"))

                if has_token:
                    token_knn = KNeighborsRegressor(
                        n_neighbors=final_knn_k, metric="cosine", weights="distance"
                    )
                    token_knn.fit(X_cat_train, y_token_train)
                    dump(token_knn, os.path.join(cat_dir, f"{model_name}_token.joblib"))

                meta = {"architecture": "KNN", "budgets": valid_budgets,
                        "knn_k": final_knn_k, "cv_mean_r2": float(knn_mean_r2)}
                with open(os.path.join(cat_dir, f"{model_name}_quality_meta.json"), "w") as f:
                    json.dump(meta, f)

            else:  # MLP
                mlp_model, token_scale = train_mlp_final(
                    X_cat_train, Y_quality_train, y_token_train,
                    hidden_dim=args.hidden, epochs=args.epochs, device=device,
                )
                torch.save({
                    "state_dict": mlp_model.state_dict(),
                    "input_dim": X_cat_train.shape[1],
                    "hidden_dim": args.hidden,
                    "num_quality_heads": num_heads,
                    "budgets": valid_budgets,
                    "architecture": "MLP",
                    "token_scale": float(token_scale),
                    "cv_mean_r2": float(mlp_mean_r2),
                }, os.path.join(cat_dir, f"{model_name}_quality_mlp.pt"))

                meta = {"architecture": "MLP", "budgets": valid_budgets,
                        "hidden_dim": args.hidden, "cv_mean_r2": float(mlp_mean_r2)}
                with open(os.path.join(cat_dir, f"{model_name}_quality_meta.json"), "w") as f:
                    json.dump(meta, f)

            # --- Evaluate on held-out test set (skip in --full mode) ---
            if len(cat_test) == 0:
                test_mean_r2 = 0.0
                test_token_r2 = 0.0
            elif best_arch == "Ridge":
                test_r2s = []
                for h, b in enumerate(valid_budgets):
                    if args.logistic:
                        y_bin_train = (Y_quality_train[:, h] > 0.5).astype(int)
                        y_bin_test = (Y_quality_test[:, h] > 0.5).astype(int)
                        if y_bin_train.sum() == 0 or y_bin_train.sum() == len(y_bin_train):
                            test_r2s.append(0.0)
                            continue
                        clf = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs')
                        clf.fit(X_cat_train, y_bin_train)
                        y_prob = clf.predict_proba(X_cat_test)[:, 1]
                        if y_bin_test.sum() > 0 and y_bin_test.sum() < len(y_bin_test):
                            auc = roc_auc_score(y_bin_test, y_prob)
                            test_r2s.append(auc)
                        else:
                            test_r2s.append(0.5)
                    else:
                        reg = Ridge(alpha=args.alpha)
                        reg.fit(X_cat_train, Y_quality_train[:, h])
                        y_pred = reg.predict(X_cat_test)
                        r2 = r2_score(Y_quality_test[:, h], y_pred) if Y_quality_test[:, h].var() > 1e-10 else 0.0
                        test_r2s.append(r2)
                test_mean_r2 = np.mean(test_r2s)

                test_token_r2 = 0.0
                if has_token:
                    treg = LinearRegression()
                    treg.fit(X_cat_train, y_token_train)
                    tp = treg.predict(X_cat_test)
                    test_token_r2 = r2_score(y_token_test, tp) if y_token_test.var() > 1e-10 else 0.0

            elif best_arch == "KNN":
                test_r2s = []
                for h, b in enumerate(valid_budgets):
                    knn = KNeighborsRegressor(
                        n_neighbors=final_knn_k, metric="cosine", weights="distance"
                    )
                    knn.fit(X_cat_train, Y_quality_train[:, h])
                    y_pred = knn.predict(X_cat_test)
                    r2 = r2_score(Y_quality_test[:, h], y_pred) if Y_quality_test[:, h].var() > 1e-10 else 0.0
                    test_r2s.append(r2)
                test_mean_r2 = np.mean(test_r2s)

                test_token_r2 = 0.0
                if has_token:
                    tknn = KNeighborsRegressor(
                        n_neighbors=final_knn_k, metric="cosine", weights="distance"
                    )
                    tknn.fit(X_cat_train, y_token_train)
                    tp = tknn.predict(X_cat_test)
                    test_token_r2 = r2_score(y_token_test, tp) if y_token_test.var() > 1e-10 else 0.0

            else:  # MLP
                mlp_model.eval()
                X_te = torch.tensor(X_cat_test, dtype=torch.float32).to(device)
                with torch.no_grad():
                    q_preds, t_pred = mlp_model(X_te)
                    q_preds = q_preds.cpu().numpy()
                    t_pred = t_pred.cpu().numpy().squeeze() * token_scale

                test_r2s = []
                for h in range(num_heads):
                    r2 = r2_score(Y_quality_test[:, h], q_preds[:, h]) if Y_quality_test[:, h].var() > 1e-10 else 0.0
                    test_r2s.append(r2)
                test_mean_r2 = np.mean(test_r2s)
                test_token_r2 = r2_score(y_token_test, t_pred) if y_token_test.var() > 1e-10 else 0.0

            log(f"    Test R2: quality={test_mean_r2:.4f}, token={test_token_r2:.4f}")

            results_summary[cat_name]["models"][model_name] = {
                "best_architecture": best_arch,
                "n_budgets": num_heads,
                "knn_k": knn_k,
                "Ridge_cv_r2": float(ridge_mean_r2),
                "MLP_cv_r2": float(mlp_mean_r2),
                "KNN_cv_r2": float(knn_mean_r2),
                "test_quality_r2": float(test_mean_r2),
                "test_token_r2": float(test_token_r2),
                "ridge_token_cv_r2": float(np.mean(ridge_token_cv_r2s)) if ridge_token_cv_r2s else 0.0,
                "mlp_token_cv_r2": float(np.mean(mlp_token_cv_r2s)) if mlp_token_cv_r2s else 0.0,
                "knn_token_cv_r2": float(np.mean(knn_token_cv_r2s)) if knn_token_cv_r2s else 0.0,
            }

    # 3b. Compute and save category means for shrinkage
    log("\nComputing category means for shrinkage...")
    category_means = {}  # {cat_name: {model_name: {budget: mean_accuracy}}}
    category_token_means = {}  # {cat_name: {model_name: mean_tokens}}
    for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
        cat_train = np.array([i for i in np.where(categories == cat_idx)[0] if i in train_set])
        if len(cat_train) == 0:
            continue
        category_means[cat_name] = {}
        category_token_means[cat_name] = {}
        for model_name in MODELS:
            if model_name not in models_data:
                continue
            model_budgets_data = models_data[model_name]
            budgets = get_budgets(model_name)
            category_means[cat_name][model_name] = {}
            for b in budgets:
                if b in model_budgets_data:
                    y = model_budgets_data[b]["accuracy"][cat_train]
                    category_means[cat_name][model_name][b] = float(y.mean())
            # Token mean (from concise or first available budget)
            for b in ["concise"] + budgets:
                if b in model_budgets_data:
                    t = model_budgets_data[b]["output_tokens"][cat_train]
                    category_token_means[cat_name][model_name] = float(t.mean())
                    break

    means_path = os.path.join(CHECKPOINT_DIR, "category_means.pkl")
    with open(means_path, "wb") as f:
        pickle.dump({"quality": category_means, "tokens": category_token_means}, f)
    log(f"Saved category means to {means_path}")

    # 4. Overall summary
    log(f"\n{'='*70}")
    log(f"ARCHITECTURE SELECTION SUMMARY")
    log(f"{'='*70}")
    log(f"  Ridge wins: {global_ridge_wins}")
    log(f"  MLP wins:   {global_mlp_wins}")
    log(f"  KNN wins:   {global_knn_wins}")

    log(f"\nPer category:")
    for cat_name in CATEGORY_NAMES:
        if cat_name not in results_summary:
            continue
        cat_archs = defaultdict(int)
        cat_r2s = []
        for model_name, minfo in results_summary[cat_name]["models"].items():
            cat_archs[minfo["best_architecture"]] += 1
            cat_r2s.append(minfo["test_quality_r2"])
        arch_str = ", ".join(f"{a}={c}" for a, c in sorted(cat_archs.items(), key=lambda x: -x[1]))
        mean_test_r2 = np.mean(cat_r2s) if cat_r2s else 0.0
        log(f"  {cat_name:<12} {arch_str:<25} mean_test_R2={mean_test_r2:.4f}")

    # Save results
    summary_path = os.path.join(CHECKPOINT_DIR, "predictor_results.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    log(f"\nSaved results to {summary_path}")

    log("\nDone!")


if __name__ == "__main__":
    main()
