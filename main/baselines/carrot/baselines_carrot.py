"""
CARROT Baselines: KNN and Linear Regression

Two baseline methods for LLM routing:
1. CARROT-KNN: K-Nearest Neighbors baseline
2. CARROT-Linear: Linear Regression baseline

Both predict:
- Quality scores for unlimited setting
- Token counts for unlimited setting
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from joblib import dump, load


def route_baseline(Y_hat_score: np.ndarray,
                   Y_hat_count: np.ndarray,
                   Y_score_true: np.ndarray,
                   Y_count_true: np.ndarray,
                   lamb_range: np.ndarray,
                   sizes_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Route queries to optimal models based on predicted scores and costs.

    Args:
        Y_hat_score: Predicted quality scores, shape (n_queries, n_models)
        Y_hat_count: Predicted token counts, shape (n_queries, n_models)
        Y_score_true: True quality scores, shape (n_queries, n_models)
        Y_count_true: True token counts, shape (n_queries, n_models)
        lamb_range: Array of lambda values for cost-performance tradeoff
        sizes_vec: Model sizes for cost calculation, shape (n_models,)

    Returns:
        Tuple of (router_cost, router_perf) arrays
    """
    pred_cost_mat = Y_hat_count * sizes_vec
    true_cost_mat = Y_count_true * sizes_vec

    router_cost, router_perf = [], []
    n_queries = Y_hat_score.shape[0]

    for lam in lamb_range:
        risk = (1 - lam) * Y_hat_score - lam * pred_cost_mat
        chosen = risk.argmax(axis=1)

        chosen_perf = Y_score_true[np.arange(n_queries), chosen]
        chosen_cost = true_cost_mat[np.arange(n_queries), chosen]

        router_perf.append(chosen_perf.mean())
        router_cost.append(chosen_cost.mean())

    return np.array(router_cost), np.array(router_perf)


def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray,
                                 llm_name: str, plot_dir: str):
    """
    Create and save confusion matrix plot for quality predictions.

    Args:
        y_true: True quality scores for one LLM
        y_pred: Predicted quality scores for one LLM
        llm_name: Name of the LLM
        plot_dir: Directory to save plot
    """
    n_buckets = 10
    y_true_bucket = np.clip((y_true * n_buckets).astype(int), 0, n_buckets - 1)
    y_pred_bucket = np.clip((y_pred * n_buckets).astype(int), 0, n_buckets - 1)
    labels = [f"{i/10:.1f}" for i in range(n_buckets)]

    cm = confusion_matrix(y_true_bucket, y_pred_bucket, labels=range(n_buckets))

    # Compute metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_xlabel("Predicted Quality Bucket", fontsize=11)
    ax.set_ylabel("True Quality Bucket", fontsize=11)
    ax.set_title(f"Quality Confusion Matrix: {llm_name}\nMSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}",
                 fontsize=12)

    os.makedirs(plot_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"confmat_quality_{llm_name}.png"), dpi=150)
    plt.close(fig)


def create_token_distribution_plot(y_true: np.ndarray, y_pred: np.ndarray,
                                   llm_name: str, plot_dir: str):
    """
    Create and save token count distribution plot.

    Args:
        y_true: True token counts for one LLM
        y_pred: Predicted token counts for one LLM
        llm_name: Name of the LLM
        plot_dir: Directory to save plot
    """
    # Compute metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Distribution plot
    sns.kdeplot(y_true, label="True", fill=True, color="blue", alpha=0.4, ax=ax1)
    sns.kdeplot(y_pred, label="Predicted", fill=True, color="orange", alpha=0.4, ax=ax1)
    ax1.set_xlabel("Token Count", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title(f"Token Count Distribution: {llm_name}", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.3, s=10)

    # Add diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax2.set_xlabel("True Token Count", fontsize=11)
    ax2.set_ylabel("Predicted Token Count", fontsize=11)
    ax2.set_title(f"True vs Predicted: {llm_name}\nMSE={mse:.1f}, MAE={mae:.1f}, R²={r2:.4f}",
                  fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    os.makedirs(plot_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"token_count_{llm_name}.png"), dpi=150)
    plt.close(fig)


class CarrotBaseline:
    """
    Base class for CARROT baselines.

    CARROT baselines predict unlimited quality scores and token counts
    using either KNN or Linear Regression.

    Attributes:
        regressor_type: Type of regressor ('knn' or 'linear')
        score_model: Regression model for quality prediction
        count_model: Regression model for token count prediction
        X_test: Stored test embeddings (optional)
        Y_score_test_true: Stored test quality scores (optional)
        Y_count_test_true: Stored test token counts (optional)
    """

    def __init__(self,
                 regressor_type: str = 'knn',
                 n_neighbors_score: int = 256,
                 n_neighbors_count: int = 256,
                 metric: str = "cosine",
                 fit_intercept: bool = True,
                 load_dir: Optional[str] = None):
        """
        Initialize CARROT baseline.

        Args:
            regressor_type: Type of regressor ('knn' or 'linear')
            n_neighbors_score: Number of neighbors for quality KNN (only used if regressor_type='knn')
            n_neighbors_count: Number of neighbors for count KNN (only used if regressor_type='knn')
            metric: Distance metric for KNN (only used if regressor_type='knn')
            fit_intercept: Whether to fit intercept for linear regression (only used if regressor_type='linear')
            load_dir: Optional directory to load pre-trained models
        """
        self.regressor_type = regressor_type
        self.n_neighbors_score = n_neighbors_score
        self.n_neighbors_count = n_neighbors_count
        self.metric = metric
        self.fit_intercept = fit_intercept

        # Initialize models based on type
        if regressor_type == 'knn':
            self.score_model = KNeighborsRegressor(n_neighbors=n_neighbors_score, metric=metric)
            self.count_model = KNeighborsRegressor(n_neighbors=n_neighbors_count, metric=metric)
        elif regressor_type == 'linear':
            self.score_model = LinearRegression(fit_intercept=fit_intercept)
            self.count_model = LinearRegression(fit_intercept=fit_intercept)
        else:
            raise ValueError(f"Unknown regressor_type: {regressor_type}. Must be 'knn' or 'linear'.")

        if load_dir:
            self.load(load_dir)

    def fit(self,
            embedding_train: np.ndarray,
            quality_train: np.ndarray,
            token_count_train: np.ndarray,
            save_dir: Optional[str] = None) -> 'CarrotBaseline':
        """
        Train CARROT models (standard sklearn pattern).

        Args:
            embedding_train: Training query embeddings, shape (n_train, embedding_dim)
            quality_train: Training quality scores, shape (n_train, n_models)
            token_count_train: Training token counts, shape (n_train, n_models)
            save_dir: Optional directory to save trained models

        Returns:
            self (for method chaining)
        """
        model_name = "CARROT-KNN" if self.regressor_type == 'knn' else "CARROT-Linear"
        print(f"🚀 Training {model_name}...")

        # Train quality and token count models
        self.score_model.fit(embedding_train, quality_train)
        self.count_model.fit(embedding_train, token_count_train)

        print(f"✅ {model_name} training complete")

        # Save models if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            dump(self.score_model, os.path.join(save_dir, f"{self.regressor_type}_score.joblib"))
            dump(self.count_model, os.path.join(save_dir, f"{self.regressor_type}_count.joblib"))
            print(f"💾 Saved {model_name} models to {save_dir}")

        return self

    def predict(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict quality scores and token counts (standard sklearn pattern).

        Args:
            embedding: Query embeddings to predict on, shape (n_queries, embedding_dim)

        Returns:
            Tuple of (Y_hat_score, Y_hat_count), each shape (n_queries, n_models)
        """

        Y_hat_score = self.score_model.predict(embedding)
        Y_hat_count = self.count_model.predict(embedding)
        return Y_hat_score, Y_hat_count

    def load(self, load_dir: str) -> 'CarrotBaseline':
        """
        Load pre-trained models.

        Args:
            load_dir: Directory containing saved models

        Returns:
            self
        """
        score_path = os.path.join(load_dir, f"{self.regressor_type}_score.joblib")
        count_path = os.path.join(load_dir, f"{self.regressor_type}_count.joblib")

        if os.path.isfile(score_path):
            self.score_model = load(score_path)
            print(f"✅ Loaded quality model from {score_path}")
        else:
            print(f"⚠️  Missing: {score_path}")

        if os.path.isfile(count_path):
            self.count_model = load(count_path)
            print(f"✅ Loaded token count model from {count_path}")
        else:
            print(f"⚠️  Missing: {count_path}")

        return self


class CarrotKNNBaseline(CarrotBaseline):
    """CARROT-KNN: K-Nearest Neighbors baseline."""

    def __init__(self,
                 n_neighbors_score: int = 256,
                 n_neighbors_count: int = 256,
                 metric: str = "cosine",
                 load_dir: Optional[str] = None):
        """
        Initialize CARROT-KNN baseline.

        Args:
            n_neighbors_score: Number of neighbors for quality prediction
            n_neighbors_count: Number of neighbors for token count prediction
            metric: Distance metric (default: cosine)
            load_dir: Optional directory to load pre-trained models
        """
        super().__init__(
            regressor_type='knn',
            n_neighbors_score=n_neighbors_score,
            n_neighbors_count=n_neighbors_count,
            metric=metric,
            load_dir=load_dir
        )


class CarrotLinearBaseline(CarrotBaseline):
    """CARROT-Linear: Linear Regression baseline."""

    def __init__(self, fit_intercept: bool = True, load_dir: Optional[str] = None):
        """
        Initialize CARROT-Linear baseline.

        Args:
            fit_intercept: Whether to calculate intercept for linear regression
            load_dir: Optional directory to load pre-trained models
        """
        super().__init__(
            regressor_type='linear',
            fit_intercept=fit_intercept,
            load_dir=load_dir
        )
