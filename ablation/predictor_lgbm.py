#!/usr/bin/env python3
"""
Random Forest-based predictor for LLM performance scores across token limits.
This is an ablation study variant using sklearn's RandomForestRegressor instead of Linear Regression.
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
from typing import Dict, List, Optional, Tuple


class TokenPerformancePredictor:
    """
    Random Forest-based predictor for LLM performance scores across token limits.

    Architecture:
        - 15 limited budget score predictors (10, 20, 30, ..., 4000)
        - 1 unlimited quality predictor (predicts quality for unlimited setting)
        - 1 unlimited token count predictor (predicts actual tokens used in unlimited setting)

    Attributes:
        limited_score_predictors: Dict mapping limited token names to RandomForestRegressor models
        unlimited_score_predictor: RandomForestRegressor model for unlimited quality prediction
        unlimited_token_predictor: RandomForestRegressor model for unlimited token count prediction
        token_limits: List of token limit column names
        rf_params: RandomForest hyperparameters
    """

    def __init__(self,
                 token_limits: Optional[List[str]] = None,
                 load_dir: Optional[str] = None,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 n_jobs: int = -1,
                 random_state: int = 42,
                 **rf_kwargs):
        """
        Initialize the RandomForest predictor.

        Args:
            token_limits: List of token limit column names (required for training, optional for loading)
            load_dir: Directory to load pre-trained models from (optional)
            n_estimators: Number of trees in the forest
            max_depth: Maximum tree depth (None for unlimited)
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
            **rf_kwargs: Additional RandomForestRegressor parameters
        """
        self.token_limits = token_limits
        self.rf_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'n_jobs': n_jobs,
            'random_state': random_state,
            'verbose': 0,  # Suppress warnings
            **rf_kwargs
        }

        self.limited_score_predictors: Dict = {}
        self.unlimited_score_predictor = None
        self.unlimited_token_predictor = None

        if load_dir:
            self._load_models(load_dir)

    def _create_model(self):
        """Create a new RandomForestRegressor instance."""
        return RandomForestRegressor(**self.rf_params)

    def _load_models(self, load_dir: str):
        """Load pre-trained models from directory."""
        limited_path = os.path.join(load_dir, "limited_score_predictors.joblib")
        unlimited_score_path = os.path.join(load_dir, "unlimited_score_predictor.joblib")
        unlimited_token_path = os.path.join(load_dir, "unlimited_token_predictor.joblib")

        # Load limited budget score predictors (15 models)
        if os.path.isfile(limited_path):
            self.limited_score_predictors = load(limited_path)
            print(f"✅ Loaded {len(self.limited_score_predictors)} limited budget RandomForest predictors from {limited_path}")
        else:
            print(f"⚠️  Missing: {limited_path}")

        # Load unlimited quality predictor (1 model)
        if os.path.isfile(unlimited_score_path):
            self.unlimited_score_predictor = load(unlimited_score_path)
            print(f"✅ Loaded unlimited quality RandomForest predictor from {unlimited_score_path}")
        else:
            print(f"⚠️  Missing: {unlimited_score_path}")

        # Load unlimited token count predictor (1 model)
        if os.path.isfile(unlimited_token_path):
            self.unlimited_token_predictor = load(unlimited_token_path)
            print(f"✅ Loaded unlimited token count RandomForest predictor from {unlimited_token_path}")
        else:
            print(f"⚠️  Missing: {unlimited_token_path}")

        # Extract token_limits from loaded models if not provided
        if self.token_limits is None and self.limited_score_predictors:
            self.token_limits = list(self.limited_score_predictors.keys()) + ['unlimited_score']

    def fit(self, embedding_train: np.ndarray, quality_train: Dict[str, np.ndarray],
            token_count_train: np.ndarray, save_dir: Optional[str] = None):
        """
        Train RandomForest predictors for all token limits.

        Args:
            embedding_train: Training embeddings, shape (n_train, embedding_dim)
            quality_train: Dict mapping token limit names to quality scores
                          Keys: '10_score', '20_score', ..., '4000_score', 'unlimited_score'
                          Values: numpy arrays of shape (n_train,)
            token_count_train: Training token counts for unlimited setting, shape (n_train,)
            save_dir: Optional directory to save trained models
        """
        if self.token_limits is None:
            raise ValueError("token_limits must be provided for training")

        print(f"🚀 Training RandomForest predictors for {len(self.token_limits)} token limits...")
        print(f"   RF params: n_estimators={self.rf_params['n_estimators']}, "
              f"max_depth={self.rf_params['max_depth']}, n_jobs={self.rf_params['n_jobs']}")

        # Separate limited and unlimited token limits
        limited_tokens = [t for t in self.token_limits if t != 'unlimited_score']

        # 1. Train score predictors for limited budgets (15 models)
        print(f"   Training {len(limited_tokens)} limited budget score predictors...")
        for token_name in limited_tokens:
            model = self._create_model()
            model.fit(embedding_train, quality_train[token_name])
            self.limited_score_predictors[token_name] = model

        # 2. Train score predictor for unlimited (1 model)
        print("   Training unlimited quality predictor...")
        self.unlimited_score_predictor = self._create_model()
        self.unlimited_score_predictor.fit(embedding_train, quality_train['unlimited_score'])

        # 3. Train token count predictor for unlimited
        print("   Training unlimited token count predictor...")
        self.unlimited_token_predictor = self._create_model()
        self.unlimited_token_predictor.fit(embedding_train, token_count_train)

        print("✅ RandomForest training complete")

        # Save models if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            dump(self.limited_score_predictors, os.path.join(save_dir, "limited_score_predictors.joblib"))
            dump(self.unlimited_score_predictor, os.path.join(save_dir, "unlimited_score_predictor.joblib"))
            dump(self.unlimited_token_predictor, os.path.join(save_dir, "unlimited_token_predictor.joblib"))
            print(f"💾 Saved RandomForest models to {save_dir}")

    def predict(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict quality scores and token counts for given query embeddings.

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
            raise RuntimeError("token_limits not set.")

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


def route_scores(predicted_scores_dict, predicted_count, true_scores_dict, true_counts_dict,
                lambda_range, model_size):
    """
    Route queries based on predicted scores and costs.

    Args:
        predicted_scores_dict: Dict mapping token limit names to predicted scores
        predicted_count: Predicted token counts for unlimited setting
        true_scores_dict: Dict mapping token limit names to true scores
        true_counts_dict: Dict mapping token limit names to true token counts
        lambda_range: Array of lambda values (cost-performance tradeoff)
        model_size: Model size in billions

    Returns:
        Tuple of (costs, performances) for each lambda value
    """
    costs = []
    performances = []

    token_limit_names = list(predicted_scores_dict.keys())
    n_test = len(predicted_scores_dict[token_limit_names[0]])

    for lam in lambda_range:
        # Compute risk for each token limit
        risks = []
        for token_name in token_limit_names:
            pred_score = predicted_scores_dict[token_name]
            if token_name == 'unlimited_score':
                pred_cost = predicted_count * model_size
            else:
                # For limited budgets, use true token counts (assumption: known budget)
                pred_cost = true_counts_dict[token_name] * model_size

            risk = (1 - lam) * pred_score - lam * pred_cost
            risks.append(risk)

        # Stack and find best option for each query
        risks_matrix = np.column_stack(risks)  # (n_test, n_limits)
        best_idx = np.argmax(risks_matrix, axis=1)

        # Get actual performance and cost for selected options
        selected_scores = []
        selected_costs = []
        for i in range(n_test):
            token_name = token_limit_names[best_idx[i]]
            selected_scores.append(true_scores_dict[token_name][i])
            selected_costs.append(true_counts_dict[token_name][i] * model_size)

        # Average across test set
        costs.append(np.mean(selected_costs))
        performances.append(np.mean(selected_scores))

    return np.array(costs), np.array(performances)
