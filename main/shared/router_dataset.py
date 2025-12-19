import numpy as np
import pandas as pd


class RouterDataset:
    def __init__(self,
                 embeddings: np.ndarray,
                 score_df_path: str,
                 target_tokens_score,
                 train_idx: np.ndarray,
                 test_idx: np.ndarray):
        """
        Initialize RouterDataset with pre-computed train/test split.

        Args:
            embeddings: Full embeddings array
            score_df_path: Path to CSV file with scores
            target_tokens_score: List of token limit columns to use
            train_idx: Indices for training set
            test_idx: Indices for test set
        """
        self.embeddings = embeddings
        self.score_df = pd.read_csv(score_df_path)
        assert len(self.embeddings) == len(self.score_df), \
            f"Embeddings ({len(self.embeddings)}) and score_df ({len(self.score_df)}) length mismatch"

        self.target_tokens_score = list(target_tokens_score)
        self.train_idx = train_idx
        self.test_idx = test_idx

        # Build supervised splits as private attributes
        self._train_set_score, self._test_set_score, \
        self._train_token_unlimited_count, self._test_token_unlimited_count = \
            self._build_supervised(self.target_tokens_score)

    def _build_supervised(self, target_tokens_score):
        train = {"X": self.embeddings[self.train_idx].astype(np.float32), "y": {}}
        test = {"X": self.embeddings[self.test_idx].astype(np.float32), "y": {}}

        for token in target_tokens_score:
            y = self.score_df[token].values.astype(np.float32)
            train["y"][token] = y[self.train_idx]
            test["y"][token] = y[self.test_idx]

        train_unlimited_count = self.score_df["unlimited_count"].values.astype(np.float32)[self.train_idx]
        test_unlimited_count = self.score_df["unlimited_count"].values.astype(np.float32)[self.test_idx]

        return train, test, train_unlimited_count, test_unlimited_count

    def get_train_set_score(self):
        return self._train_set_score

    def get_test_set_score(self):
        return self._test_set_score

    def get_target_tokens_score(self):
        return self.target_tokens_score.copy()

    def get_train_token_unlimited_count(self):
        return self._train_token_unlimited_count

    def get_test_token_unlimited_count(self):
        return self._test_token_unlimited_count
