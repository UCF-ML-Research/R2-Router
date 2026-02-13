import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from joblib import dump, load
from ..shared.router_dataset import RouterDataset


def route_scores(llms, lamb_range):
    """
    Route queries to optimal (LLM, token_limit) combinations based on risk function.

    Args:
        llms: Dictionary of LLM data (from llm_loader.load_llm)
        lamb_range: Array of lambda values for cost-performance tradeoff

    Returns:
        Tuple of (costs, performances) arrays
    """
    router_cost, router_perf = [], []
    llm_names = list(llms.keys())
    n_queries = llms[llm_names[0]]["pred_test_score"].shape[0]

    for lam in lamb_range:
        chosen_perf, chosen_cost = [], []
        for i in range(n_queries):
            best_score = -float('inf')
            best_cost = float('inf')
            best_perf = 0.0

            for llm_name in llm_names:
                llm_data = llms[llm_name]
                pred_scores = llm_data["pred_test_score"][i]
                pred_tokens = llm_data["pred_test_token"][i]

                # Use true costs for evaluation
                costs = pred_tokens * llm_data["size"]

                # Calculate routing score: (1-λ)*quality - λ*cost
                # λ ∈ [0,1]: λ=0 pure quality, λ=1 pure cost minimization
                risks = (1 - lam) * pred_scores - lam * costs
                best_token_idx = risks.argmax()

                if risks[best_token_idx] > best_score:
                    best_score = risks[best_token_idx]
                    true_tokens = llm_data["true_test_count"][i, best_token_idx]
                    best_cost = true_tokens * llm_data["size"]
                    best_perf = llm_data["true_test_score"][i, best_token_idx]

            chosen_perf.append(best_perf)
            chosen_cost.append(best_cost)

        router_perf.append(np.mean(chosen_perf))
        router_cost.append(np.mean(chosen_cost))

    return np.array(router_cost), np.array(router_perf)


def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray,
                                 token_name: str, plot_dir: str):
    """
    Create and save confusion matrix plot for score predictions.

    Args:
        y_true: True scores
        y_pred: Predicted scores
        token_name: Token limit name (e.g., '10_score')
        plot_dir: Directory to save plot
    """
    n_buckets = 10
    y_true_bucket = np.clip((y_true * n_buckets).astype(int), 0, n_buckets - 1)
    y_pred_bucket = np.clip((y_pred * n_buckets).astype(int), 0, n_buckets - 1)
    labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(n_buckets)]

    cm = confusion_matrix(y_true_bucket, y_pred_bucket, labels=range(n_buckets))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_xlabel("Predicted Bucket")
    ax.set_ylabel("True Bucket")
    ax.set_title(f"Confusion Matrix - {token_name}")

    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, f"confmat_{token_name}.png"))
    plt.close(fig)


class TokenPerformancePredictor:
    """
    Linear regression-based predictor for LLM performance scores across token limits.

    This predictor trains three separate components:
    1. Score predictors for limited token budgets (15 models: 10, 20, ..., 4000)
    2. Score predictor for unlimited setting (1 model)
    3. Token count predictor for unlimited setting (1 model)

    It follows sklearn's design pattern where data is passed to fit() and predict()
    methods rather than stored as class attributes.

    Architecture:
        - 15 limited budget score predictors (10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000)
        - 1 unlimited quality predictor (predicts quality for unlimited setting)
        - 1 unlimited token count predictor (predicts actual tokens used in unlimited setting)

    Attributes:
        limited_score_predictors: Dict mapping limited token names to LinearRegression models
                                 Keys: '10_score', '20_score', ..., '4000_score' (15 total)
        unlimited_score_predictor: LinearRegression model for unlimited quality prediction
        unlimited_token_predictor: LinearRegression model for unlimited token count prediction
        token_limits: List of token limit column names (e.g., ['10_score', ..., 'unlimited_score'])
    """

    def __init__(self,
                 token_limits: Optional[List[str]] = None,
                 load_dir: Optional[str] = None,
                 model_type: str = "linear",
                 random_state: int = 42,
                 **model_kwargs):
        """
        Initialize the predictor.

        Args:
            token_limits: List of token limit column names (required for training, optional for loading)
                         Should include both limited and 'unlimited_score'
            load_dir: Directory to load pre-trained models from (optional)
            model_type: Type of regression model to use:
                       - "linear": Linear Regression (no regularization)
                       - "random_forest": Random Forest Regressor
                       - "gradient_boosting": Gradient Boosting Regressor
            random_state: Random seed for reproducibility
            **model_kwargs: Additional keyword arguments for the model
        """
        self.token_limits = token_limits
        self.model_type = model_type
        self.random_state = random_state
        self.model_kwargs = model_kwargs

        self.limited_score_predictors: Dict = {}
        self.unlimited_score_predictor = None
        self.unlimited_token_predictor = None

        if load_dir:
            self._load_models(load_dir)

    def _create_model(self):
        """Create a new model instance based on model_type."""
        if self.model_type == "linear":
            return LinearRegression(**self.model_kwargs)
        elif self.model_type == "random_forest":
            return RandomForestRegressor(random_state=self.random_state, **self.model_kwargs)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(random_state=self.random_state, **self.model_kwargs)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _load_models(self, load_dir: str):
        """Load pre-trained models from directory."""
        limited_path = os.path.join(load_dir, "limited_score_predictors.joblib")
        unlimited_score_path = os.path.join(load_dir, "unlimited_score_predictor.joblib")
        unlimited_token_path = os.path.join(load_dir, "unlimited_token_predictor.joblib")

        # Load limited budget score predictors (15 models)
        if os.path.isfile(limited_path):
            self.limited_score_predictors = load(limited_path)
            print(f"✅ Loaded {len(self.limited_score_predictors)} limited budget score predictors from {limited_path}")
        else:
            print(f"⚠️  Missing: {limited_path}")

        # Load unlimited quality predictor
        if os.path.isfile(unlimited_score_path):
            self.unlimited_score_predictor = load(unlimited_score_path)
            print(f"✅ Loaded unlimited quality predictor from {unlimited_score_path}")
        else:
            print(f"⚠️  Missing: {unlimited_score_path}")

        # Load unlimited token count predictor
        if os.path.isfile(unlimited_token_path):
            self.unlimited_token_predictor = load(unlimited_token_path)
            print(f"✅ Loaded unlimited token count predictor from {unlimited_token_path}")
        else:
            print(f"⚠️  Missing: {unlimited_token_path}")

        # Extract token_limits from loaded models if not provided
        if self.token_limits is None and self.limited_score_predictors:
            self.token_limits = list(self.limited_score_predictors.keys()) + ['unlimited_score']

    def fit(self,
            embedding_train: np.ndarray,
            quality_train_limited: Dict[str, np.ndarray],
            quality_train_unlimited: np.ndarray,
            token_count_train: np.ndarray,
            embedding_test: Optional[np.ndarray] = None,
            quality_test_limited: Optional[Dict[str, np.ndarray]] = None,
            quality_test_unlimited: Optional[np.ndarray] = None,
            token_count_test: Optional[np.ndarray] = None,
            save_dir: Optional[str] = None,
            plot_dir: Optional[str] = None) -> 'TokenPerformancePredictor':
        """
        Train linear regression models for the three components.

        Args:
            embedding_train: Training query embeddings, shape (n_train, embedding_dim)
            quality_train_limited: Dict mapping LIMITED token limit names to training quality scores
                                  Keys: '10_score', '20_score', ..., '4000_score' (15 total)
                                  Values: (n_train,) arrays of quality scores [0, 1]
            quality_train_unlimited: Training quality scores for unlimited setting, shape (n_train,)
            token_count_train: Training token counts for unlimited setting, shape (n_train,)
            embedding_test: Optional test query embeddings for evaluation
            quality_test_limited: Optional test quality scores for limited budgets
            quality_test_unlimited: Optional test quality scores for unlimited setting
            token_count_test: Optional test token counts for evaluation
            save_dir: Optional directory to save trained models
            plot_dir: Optional directory to save evaluation plots

        Returns:
            self (for method chaining)
        """
        if self.token_limits is None:
            # Infer token_limits from quality_train_limited keys + unlimited
            self.token_limits = list(quality_train_limited.keys()) + ['unlimited_score']

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)

        # Extract limited token names from token_limits
        limited_tokens = [t for t in self.token_limits if t != 'unlimited_score']

        # ============================================================
        # 1. Train limited budget score predictors (15 models)
        # ============================================================
        print(f"🚀 [1/3] Training limited budget score predictors (15 models) with {self.model_type}...")
        limited_preds_dict, limited_true_dict = {}, {}

        for token in tqdm(limited_tokens, desc="Limited budgets"):
            if token not in quality_train_limited:
                raise KeyError(f"Token '{token}' not found in quality_train_limited")

            reg = self._create_model()
            reg.fit(embedding_train, quality_train_limited[token])
            self.limited_score_predictors[token] = reg

            # Evaluate on test set if provided
            if embedding_test is not None and quality_test_limited is not None and token in quality_test_limited:
                quality_pred = reg.predict(embedding_test)
                limited_preds_dict[token] = quality_pred
                limited_true_dict[token] = quality_test_limited[token]

                if plot_dir:
                    create_confusion_matrix_plot(quality_test_limited[token], quality_pred, token, plot_dir)

        # Evaluate aggregated limited budget predictions
        if limited_preds_dict and limited_true_dict:
            quality_true = np.column_stack([limited_true_dict[t] for t in limited_tokens])
            quality_pred = np.column_stack([limited_preds_dict[t] for t in limited_tokens])
            print(f"[Limited Score Predictors] MSE={mean_squared_error(quality_true, quality_pred):.4f}, "
                  f"MAE={mean_absolute_error(quality_true, quality_pred):.4f}, "
                  f"R2={r2_score(quality_true, quality_pred):.4f}")

        # ============================================================
        # 2. Train unlimited quality predictor (1 model)
        # ============================================================
        print(f"🚀 [2/3] Training unlimited quality predictor with {self.model_type}...")
        self.unlimited_score_predictor = self._create_model()
        self.unlimited_score_predictor.fit(embedding_train, quality_train_unlimited)

        if embedding_test is not None and quality_test_unlimited is not None:
            quality_pred_unlimited = self.unlimited_score_predictor.predict(embedding_test)

            print(f"[Unlimited Quality Predictor] MSE={mean_squared_error(quality_test_unlimited, quality_pred_unlimited):.4f}, "
                  f"MAE={mean_absolute_error(quality_test_unlimited, quality_pred_unlimited):.4f}, "
                  f"R2={r2_score(quality_test_unlimited, quality_pred_unlimited):.4f}")

            if plot_dir:
                create_confusion_matrix_plot(quality_test_unlimited, quality_pred_unlimited,
                                            'unlimited_score', plot_dir)

        # ============================================================
        # 3. Train unlimited token count predictor (1 model)
        # ============================================================
        print(f"🚀 [3/3] Training unlimited token count predictor with {self.model_type}...")
        self.unlimited_token_predictor = self._create_model()
        self.unlimited_token_predictor.fit(embedding_train, token_count_train)

        if embedding_test is not None and token_count_test is not None:
            count_pred = self.unlimited_token_predictor.predict(embedding_test)
            print(f"[Unlimited Token Count Predictor] MSE={mean_squared_error(token_count_test, count_pred):.4f}, "
                  f"MAE={mean_absolute_error(token_count_test, count_pred):.4f}, "
                  f"R2={r2_score(token_count_test, count_pred):.4f}")

            # Visualize token count predictions
            if plot_dir:
                plt.figure(figsize=(6, 5))
                sns.kdeplot(token_count_test, label="True", fill=True, color="blue", alpha=0.3)
                sns.kdeplot(count_pred, label="Pred", fill=True, color="orange", alpha=0.3)
                plt.title("Unlimited Token Count Predictor")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "unlimited_token_count.png"))
                plt.close()

        # ============================================================
        # Save all three model components
        # ============================================================
        if save_dir:
            dump(self.limited_score_predictors, os.path.join(save_dir, "limited_score_predictors.joblib"))
            dump(self.unlimited_score_predictor, os.path.join(save_dir, "unlimited_score_predictor.joblib"))
            dump(self.unlimited_token_predictor, os.path.join(save_dir, "unlimited_token_predictor.joblib"))
            print(f"💾 Saved 3 model components to {save_dir}:")
            print(f"   - limited_score_predictors.joblib ({len(self.limited_score_predictors)} models)")
            print(f"   - unlimited_score_predictor.joblib (1 model)")
            print(f"   - unlimited_token_predictor.joblib (1 model)")

        return self

    def predict(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict quality scores and token counts for given query embeddings.

        Returns three separate predictions matching the three-component architecture:
        1. Quality predictions for limited budgets (15 token limits)
        2. Quality prediction for unlimited setting
        3. Token count prediction for unlimited setting

        Args:
            embedding: Query embeddings, shape (n_queries, embedding_dim)

        Returns:
            Tuple of (quality_pred_limited, quality_pred_unlimited, token_count_pred):
                - quality_pred_limited: shape (n_queries, 15) - quality scores for limited budgets
                - quality_pred_unlimited: shape (n_queries,) - quality scores for unlimited
                - token_count_pred: shape (n_queries,) - predicted token counts for unlimited
        """
        if not self.limited_score_predictors:
            raise RuntimeError("No models loaded. Call fit() or provide load_dir in __init__.")

        if self.token_limits is None:
            raise RuntimeError("token_limits not set. This should not happen if models are loaded.")

        # Separate limited and unlimited token limits
        limited_tokens = [t for t in self.token_limits if t != 'unlimited_score']

        # 1. Predict quality scores for limited budgets (15 models)
        limited_quality_preds = [self.limited_score_predictors[t].predict(embedding) for t in limited_tokens]
        quality_pred_limited = np.column_stack(limited_quality_preds)  # (n_queries, 15)

        # 2. Predict quality score for unlimited (1 model)
        quality_pred_unlimited = self.unlimited_score_predictor.predict(embedding)  # (n_queries,)

        # 3. Predict token counts for unlimited
        token_count_pred = self.unlimited_token_predictor.predict(embedding)  # (n_queries,)

        return quality_pred_limited, quality_pred_unlimited, token_count_pred

    def predict_combined(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with combined quality scores (backward compatibility).

        This combines limited and unlimited quality predictions into a single array
        for compatibility with code expecting (n_queries, 16) output.

        Args:
            embedding: Query embeddings, shape (n_queries, embedding_dim)

        Returns:
            Tuple of (quality_predictions, token_count_predictions):
                - quality_predictions: shape (n_queries, 16) - all 16 token limits combined
                - token_count_predictions: shape (n_queries,) - predicted token counts for unlimited
        """
        quality_limited, quality_unlimited, token_count = self.predict(embedding)

        # Combine: [15 limited + 1 unlimited] = 16 total
        quality_combined = np.column_stack([quality_limited, quality_unlimited])

        return quality_combined, token_count


def train_predictor_from_dataset(dataset: RouterDataset, save_dir: str, plot_dir: str) -> TokenPerformancePredictor:
    """
    Convenience function to train predictor from RouterDataset.

    This maintains backward compatibility with the old API where dataset was passed
    to the constructor.

    Args:
        dataset: RouterDataset instance
        save_dir: Directory to save trained models
        plot_dir: Directory to save plots

    Returns:
        Trained TokenPerformancePredictor
    """
    train_data = dataset.get_train_set_score()
    test_data = dataset.get_test_set_score()
    token_limits = dataset.get_target_tokens_score()

    # Split into limited and unlimited quality scores
    quality_train_limited = {k: v for k, v in train_data["y"].items() if k != 'unlimited_score'}
    quality_train_unlimited = train_data["y"]['unlimited_score']

    quality_test_limited = {k: v for k, v in test_data["y"].items() if k != 'unlimited_score'}
    quality_test_unlimited = test_data["y"]['unlimited_score']

    predictor = TokenPerformancePredictor(token_limits=token_limits)
    predictor.fit(
        embedding_train=train_data["X"],
        quality_train_limited=quality_train_limited,
        quality_train_unlimited=quality_train_unlimited,
        token_count_train=dataset.get_train_token_unlimited_count(),
        embedding_test=test_data["X"],
        quality_test_limited=quality_test_limited,
        quality_test_unlimited=quality_test_unlimited,
        token_count_test=dataset.get_test_token_unlimited_count(),
        save_dir=save_dir,
        plot_dir=plot_dir
    )

    return predictor


if __name__ == "__main__":
    from dataset_manager import DatasetManager

    # ============================================================================
    # HYPERPARAMETER CONFIGURATION - Edit this section to tune model performance
    # ============================================================================

    # Model type: "linear", "random_forest", "gradient_boosting"
    MODEL_TYPE = "linear"

    # Random Forest / Gradient Boosting specific parameters
    N_ESTIMATORS = 100  # Number of trees (default: 100)
    MAX_DEPTH = None  # Maximum tree depth (None = unlimited)

    # Define token limits (shared across all models)
    token_limits_score = [
        '10_score', '20_score', '30_score', '40_score', '50_score',
        '80_score', '100_score', '150_score', '200_score', '300_score',
        '500_score', '800_score', '1200_score', '2000_score', '4000_score',
        'unlimited_score'
    ]

    # Initialize centralized dataset manager
    dataset_manager = DatasetManager(
        embeddings_path="data/prompt_embeddings.pkl",
        train_ratio=0.8,
        seed=42
    )

    embeddings = dataset_manager.get_embeddings()
    train_idx, test_idx = dataset_manager.get_split_indices()

    # Define models to train
    models_to_train = [
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

    # Prepare model-specific kwargs based on model type
    if MODEL_TYPE == "random_forest":
        model_kwargs = {"n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH}
    elif MODEL_TYPE == "gradient_boosting":
        model_kwargs = {"n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH}
    else:
        model_kwargs = {}

    print("=" * 80)
    print(f"TRAINING WITH MODEL TYPE: {MODEL_TYPE}")
    if MODEL_TYPE in ["random_forest", "gradient_boosting"]:
        print(f"  N Estimators: {N_ESTIMATORS}")
        print(f"  Max Depth: {MAX_DEPTH}")
    print("=" * 80)

    # Train each model
    for model_config in models_to_train:
        model_name = model_config["name"]
        csv_path = model_config["csv"]

        print("\n" + "=" * 80)
        print(f"Training model: {model_name}")
        print("=" * 80)

        # Check if CSV file exists
        if not os.path.exists(csv_path):
            print(f"⚠️  Skipping {model_name}: CSV file not found at {csv_path}")
            continue

        # Create dataset with centralized split
        dataset = RouterDataset(
            embeddings=embeddings,
            score_df_path=csv_path,
            target_tokens_score=token_limits_score,
            train_idx=train_idx,
            test_idx=test_idx
        )

        # Create predictor with hyperparameters
        predictor = TokenPerformancePredictor(
            token_limits=token_limits_score,
            model_type=MODEL_TYPE,
            **model_kwargs
        )

        # Split data for training
        train_data = dataset.get_train_set_score()
        test_data = dataset.get_test_set_score()

        quality_train_limited = {k: v for k, v in train_data["y"].items() if k != 'unlimited_score'}
        quality_train_unlimited = train_data["y"]['unlimited_score']
        quality_test_limited = {k: v for k, v in test_data["y"].items() if k != 'unlimited_score'}
        quality_test_unlimited = test_data["y"]['unlimited_score']

        # Train predictor
        predictor.fit(
            embedding_train=train_data["X"],
            quality_train_limited=quality_train_limited,
            quality_train_unlimited=quality_train_unlimited,
            token_count_train=dataset.get_train_token_unlimited_count(),
            embedding_test=test_data["X"],
            quality_test_limited=quality_test_limited,
            quality_test_unlimited=quality_test_unlimited,
            token_count_test=dataset.get_test_token_unlimited_count(),
            save_dir=f"./checkpoints/{model_name}_multi",
            plot_dir=f"./plots/{model_name}_multi"
        )

        # Test prediction with new API (3 returns)
        quality_limited, quality_unlimited, token_count = predictor.predict(dataset.get_test_set_score()["X"])
        print(f"✅ {model_name}: Prediction done.")
        print(f"   - Limited quality shape: {quality_limited.shape}")
        print(f"   - Unlimited quality shape: {quality_unlimited.shape}")
        print(f"   - Token count shape: {token_count.shape}")

    print("\n" + "=" * 80)
    print("All models trained successfully!")
    print("=" * 80)
