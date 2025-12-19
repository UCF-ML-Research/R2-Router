"""
CARROT Baseline Routing Module
Implements CARROT-KNN and CARROT-Linear baseline routing
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Ensure demo directory is first in path
demo_dir = Path(__file__).parent.absolute()
if str(demo_dir) not in sys.path:
    sys.path.insert(0, str(demo_dir))

# Add parent directory to path to import from main codebase (for baselines_carrot)
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import config

try:
    from main.baselines.carrot.baselines_carrot import CarrotKNNBaseline, CarrotLinearBaseline
    BASELINES_AVAILABLE = True
except ImportError:
    BASELINES_AVAILABLE = False
    print("⚠️  baselines_carrot not available")


class CARROTRouter:
    """Base class for CARROT routing"""

    def __init__(self, method: str, llm_pool: Dict = None):
        """
        Initialize CARROT router

        Args:
            method: "knn" or "linear"
            llm_pool: LLM pool configuration (uses config.LLM_POOL if None)
        """
        self.method = method.lower()
        self.llm_pool = llm_pool or config.LLM_POOL
        self.model = None
        self.token_limits = config.TOKEN_LIMITS

        if self.method not in ["knn", "linear"]:
            raise ValueError(f"Invalid method: {method}. Must be 'knn' or 'linear'")

        print(f"Initializing CARROT-{method.upper()} router...")

    def load_model(self):
        """Load CARROT baseline model from checkpoint"""
        if not BASELINES_AVAILABLE:
            raise ImportError("baselines_carrot not available. Cannot load CARROT models.")

        if self.method == "knn":
            checkpoint_dir = config.CARROT_KNN_DIR
            self.model = CarrotKNNBaseline(load_dir=str(checkpoint_dir))
        else:  # linear
            checkpoint_dir = config.CARROT_LINEAR_DIR
            self.model = CarrotLinearBaseline(load_dir=str(checkpoint_dir))

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"CARROT checkpoint not found: {checkpoint_dir}")

        print(f"✅ CARROT-{self.method.upper()} model loaded")

    def route(
        self,
        embedding: np.ndarray,
        lambda_val: float = 0.5,
        budget: Optional[float] = None
    ) -> Dict:
        """
        Route a query using CARROT baseline

        Args:
            embedding: Query embedding vector
            lambda_val: Tradeoff parameter (0=quality, 1=cost)
            budget: Maximum cost budget (None=unlimited)

        Returns:
            Dictionary with routing decision
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Get predictions for unlimited setting for all LLMs
        # CARROT returns: (Y_hat_score, Y_hat_count) of shape (1, n_llms)
        # NOTE: CARROT was trained ONLY on unlimited setting, not all token limits
        Y_hat_score, Y_hat_count = self.model.predict(embedding)

        # DEBUG: Print CARROT prediction details
        print(f"\n{'='*60}")
        print(f"DEBUG: CARROT Router - route() method")
        print(f"{'='*60}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Y_hat_score shape: {Y_hat_score.shape}")
        print(f"Y_hat_count shape: {Y_hat_count.shape}")
        print(f"Y_hat_score: {Y_hat_score[0]}")
        print(f"Y_hat_count: {Y_hat_count[0]}")

        # Safety: Clip quality scores to [0, 1] to handle sklearn unbounded predictions
        Y_hat_score = np.clip(Y_hat_score, 0, 1)
        # Safety: Ensure token counts are non-negative
        Y_hat_count = np.maximum(Y_hat_count, 0)

        # Build routing options
        # CARROT only predicts unlimited setting (single quality-cost pair per LLM)
        # Unlike CoRE which represents each LLM as a quality-cost curve across token limits
        llm_names = list(self.llm_pool.keys())
        print(f"LLM pool: {llm_names} (count={len(llm_names)})")
        print(f"{'='*60}\n")
        all_options = []

        for idx, llm_name in enumerate(llm_names):
            llm_size = self.llm_pool[llm_name]["size"]

            # Get predictions for unlimited setting
            predicted_score_unlimited = Y_hat_score[0, idx]
            predicted_count_unlimited = Y_hat_count[0, idx]

            # CARROT only routes to unlimited (no per-limit predictions)
            predicted_score = predicted_score_unlimited
            predicted_count = predicted_count_unlimited
            predicted_cost = predicted_count * llm_size

            # Check budget constraint
            if budget is not None and predicted_cost > budget:
                continue

            # Calculate risk
            risk = (1 - lambda_val) * predicted_score - lambda_val * predicted_cost

            all_options.append({
                "llm_name": llm_name,
                "token_limit": "unlimited",  # CARROT always routes to unlimited
                "predicted_score": float(predicted_score),
                "predicted_count": float(predicted_count),
                "predicted_cost": float(predicted_cost),
                "risk": float(risk)
            })

        if len(all_options) == 0:
            raise RuntimeError("No valid routing options (check budget)")

        # Select option with maximum risk
        best_option = max(all_options, key=lambda x: x["risk"])
        best_option["all_options"] = all_options

        return best_option


class MockCARROTRouter:
    """Mock CARROT router for testing without checkpoints"""

    def __init__(self, method: str, llm_pool: Dict = None):
        self.method = method.lower()
        self.llm_pool = llm_pool or config.LLM_POOL
        self.token_limits = config.TOKEN_LIMITS
        print(f"⚠️  Using Mock CARROT-{method.upper()} router")

    def load_model(self):
        """No-op for mock"""
        print(f"✅ Mock CARROT-{self.method.upper()} ready")

    def route(
        self,
        embedding: np.ndarray,
        lambda_val: float = 0.5,
        budget: Optional[float] = None
    ) -> Dict:
        """
        Mock CARROT routing

        Args:
            embedding: Query embedding
            lambda_val: Tradeoff parameter
            budget: Cost budget

        Returns:
            Mock routing decision
        """
        # Use embedding hash + method for deterministic selection
        seed = abs(hash(embedding.tobytes() + self.method.encode())) % (2**32)
        rng = np.random.RandomState(seed)

        all_options = []

        for llm_name, llm_config in self.llm_pool.items():
            llm_size = llm_config["size"]

            for token_limit in self.token_limits:
                # Generate mock predictions
                # KNN tends to be slightly better than Linear
                if self.method == "knn":
                    base_score = 0.65 + (llm_size * 0.25) + rng.uniform(-0.08, 0.08)
                else:  # linear
                    base_score = 0.60 + (llm_size * 0.28) + rng.uniform(-0.10, 0.10)

                base_score = np.clip(base_score, 0, 1)

                # Token count
                if token_limit == "unlimited":
                    predicted_count = rng.uniform(120, 550)
                else:
                    predicted_count = min(token_limit * 0.75, token_limit)

                predicted_cost = predicted_count * llm_size

                # Check budget
                if budget is not None and predicted_cost > budget:
                    continue

                # Calculate risk
                risk = (1 - lambda_val) * base_score - lambda_val * predicted_cost

                all_options.append({
                    "llm_name": llm_name,
                    "token_limit": token_limit,
                    "predicted_score": float(base_score),
                    "predicted_count": float(predicted_count),
                    "predicted_cost": float(predicted_cost),
                    "risk": float(risk)
                })

        if len(all_options) == 0:
            raise RuntimeError("No valid routing options (check budget)")

        best_option = max(all_options, key=lambda x: x["risk"])
        best_option["all_options"] = all_options

        return best_option


# ============================================================================
# Factory Functions
# ============================================================================

def get_carrot_router(method: str, mock: bool = None) -> CARROTRouter:
    """
    Get a CARROT router instance

    Args:
        method: "knn" or "linear"
        mock: Use mock router (uses config.ENABLE_MOCK_MODE if None)

    Returns:
        CARROTRouter or MockCARROTRouter
    """
    if mock is None:
        mock = config.ENABLE_MOCK_MODE

    if mock:
        return MockCARROTRouter(method)
    else:
        return CARROTRouter(method)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing CARROT Routers...")
    print()

    embedding = np.random.randn(384)

    # Test both methods
    for method in ["knn", "linear"]:
        print(f"=== Mock CARROT-{method.upper()} ===")
        router = MockCARROTRouter(method)
        router.load_model()

        # Test routing with different lambda values
        for lambda_val in [0.0, 0.5, 1.0]:
            result = router.route(embedding, lambda_val=lambda_val)
            print(f"\nLambda = {lambda_val}:")
            print(f"  Selected: {result['llm_name']} @ {result['token_limit']} tokens")
            print(f"  Predicted Score: {result['predicted_score']:.3f}")
            print(f"  Predicted Cost: {result['predicted_cost']:.1f}")
            print(f"  Risk: {result['risk']:.3f}")

        print()

    # Test real routers (only if checkpoints available)
    if not config.ENABLE_MOCK_MODE and BASELINES_AVAILABLE:
        print("=== Real CARROT Routers ===")
        for method in ["knn", "linear"]:
            try:
                print(f"\nTesting CARROT-{method.upper()}...")
                router = CARROTRouter(method)
                router.load_model()

                result = router.route(embedding, lambda_val=0.5)
                print(f"  Selected: {result['llm_name']} @ {result['token_limit']} tokens")
                print(f"  Predicted Score: {result['predicted_score']:.3f}")
                print(f"  Predicted Cost: {result['predicted_cost']:.1f}")

            except Exception as e:
                print(f"  ⚠️  Test failed: {e}")
    else:
        print("\n⚠️  Skipping real router tests (mock mode or no baselines)")
