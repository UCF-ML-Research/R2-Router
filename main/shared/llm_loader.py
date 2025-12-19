import pandas as pd
import numpy as np

from .router_dataset import RouterDataset


def load_llm(name, size, score_df_path, load_dir, embeddings, train_idx, test_idx, token_limits_score, token_limits_count, predictor_class=None):
    """
    Load a single LLM's predictor and data using centralized train/test split.

    Args:
        name: LLM name
        size: Model size (for cost calculation)
        score_df_path: Path to CSV with performance scores
        load_dir: Directory containing trained model checkpoints
        embeddings: Full embeddings array (from DatasetManager)
        train_idx: Training indices (from DatasetManager)
        test_idx: Test indices (from DatasetManager)
        token_limits_score: List of score column names
        token_limits_count: List of count column names
        predictor_class: Predictor class to use (default: TokenPerformancePredictor from predictor.py)

    Returns:
        Dictionary containing predictions, ground truth, and metadata for the LLM
    """
    # Default to PyTorch predictor if not specified
    if predictor_class is None:
        from predictor import TokenPerformancePredictor
        predictor_class = TokenPerformancePredictor

    # Create dataset with pre-computed split
    dataset = RouterDataset(
        embeddings=embeddings,
        score_df_path=score_df_path,
        target_tokens_score=token_limits_score,
        train_idx=train_idx,
        test_idx=test_idx
    )

    # Load predictor - handle different predictor types
    # PyTorch predictor needs hidden_dims and dropout
    try:
        predictor = predictor_class(
            hidden_dims=[256, 128, 64],
            dropout=0.5,
            load_dir=load_dir,
            dataset=dataset
        )
    except TypeError:
        # Sklearn predictor - try new API first (token_limits + load_dir)
        try:
            predictor = predictor_class(
                token_limits=token_limits_score,
                load_dir=load_dir
            )
        except TypeError:
            # Fall back to old API (dataset + load_dir) for backward compatibility
            predictor = predictor_class(
                dataset=dataset,
                load_dir=load_dir
            )

    # Predict on both train and test splits
    # Handle both new API (3 returns) and old API (2 returns)
    test_pred = predictor.predict(dataset.get_test_set_score()["X"])
    train_pred = predictor.predict(dataset.get_train_set_score()["X"])

    if len(test_pred) == 3:
        # New API: (quality_limited, quality_unlimited, token_count)
        pred_test_score_limited, pred_test_score_unlimited, pred_test_unlimited_count = test_pred
        pred_train_score_limited, pred_train_score_unlimited, pred_train_unlimited_count = train_pred

        # Combine limited and unlimited quality scores
        pred_test_score = np.column_stack([pred_test_score_limited, pred_test_score_unlimited])
        pred_train_score = np.column_stack([pred_train_score_limited, pred_train_score_unlimited])
    else:
        # Old API: (quality_combined, token_count)
        pred_test_score, pred_test_unlimited_count = test_pred
        pred_train_score, pred_train_unlimited_count = train_pred

    # Load ground truth
    df = pd.read_csv(score_df_path)
    gt_train_df = df.loc[train_idx, token_limits_score + token_limits_count].reset_index(drop=True)
    gt_test_df = df.loc[test_idx, token_limits_score + token_limits_count].reset_index(drop=True)

    # Create predicted token arrays
    # For limited budgets: use the limit directly (not min with predicted count)
    #   Router uses the limit as the cost estimate for budgeted inference
    # For unlimited: use predicted token count
    token_limits_values = np.array([int(s.replace('_score', '')) for s in token_limits_score[:-1]])

    # Use limits directly for test set
    pred_test_token_limited = np.tile(
        token_limits_values[None, :],  # Broadcast limits across queries
        (len(test_idx), 1)
    )
    pred_test_token = np.column_stack((pred_test_token_limited, pred_test_unlimited_count))

    # Use limits directly for train set
    pred_train_token_limited = np.tile(
        token_limits_values[None, :],  # Broadcast limits across queries
        (len(train_idx), 1)
    )
    pred_train_token = np.column_stack((pred_train_token_limited, pred_train_unlimited_count))

    # Extract ground truth arrays
    true_train_score = np.array(gt_train_df.loc[:, token_limits_score])
    true_train_count = np.array(gt_train_df.loc[:, token_limits_count])
    true_test_score = np.array(gt_test_df.loc[:, token_limits_score])
    true_test_count = np.array(gt_test_df.loc[:, token_limits_count])

    return {
        "name": name,
        "size": size,

        # Test predictions
        "pred_test_score": pred_test_score,
        "pred_test_token": pred_test_token,
        "pred_test_unlimited_score": pred_test_score[:, -1],
        "pred_test_unlimited_count": pred_test_token[:, -1],

        # Train predictions
        "pred_train_score": pred_train_score,
        "pred_train_token": pred_train_token,
        "pred_train_unlimited_score": pred_train_score[:, -1],
        "pred_train_unlimited_count": pred_train_token[:, -1],

        # Test ground truth
        "true_test_score": true_test_score,
        "true_test_count": true_test_count,
        "true_test_unlimited_score": true_test_score[:, -1],
        "true_test_unlimited_count": true_test_count[:, -1],

        # Train ground truth
        "true_train_score": true_train_score,
        "true_train_count": true_train_count,
        "true_train_unlimited_score": true_train_score[:, -1],
        "true_train_unlimited_count": true_train_count[:, -1],

        # Additional data
        "gt": gt_test_df,
        "train_embeddings": dataset.get_train_set_score()["X"],
        "test_embeddings": dataset.get_test_set_score()["X"],
        "dataset": dataset,
    }