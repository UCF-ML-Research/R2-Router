"""
CARROT Baseline Routing Module
Implements CARROT baseline routing using unlimited predictors from each LLM
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Ensure demo directory is first in path
demo_dir = Path(__file__).parent.absolute()
if str(demo_dir) not in sys.path:
    sys.path.insert(0, str(demo_dir))

# Add parent directory to path to import from main codebase
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import config

try:
    from main.core.predictor_sklearn import TokenPerformancePredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    print("⚠️  TokenPerformancePredictor not available")


class CARROTRouter:
    """CARROT router using unlimited predictors from each LLM"""

    def __init__(self, llm_pool: Dict = None):
        """
        Initialize CARROT router

        Args:
            llm_pool: LLM pool configuration (uses config.LLM_POOL if None)
        """
        self.llm_pool = llm_pool or config.LLM_POOL
        self.predictors = {}
        self.token_limits = config.TOKEN_LIMITS

        print(f"Initializing CARROT router...")

    def load_model(self):
        """Load unlimited predictors from each LLM checkpoint"""
        if not PREDICTOR_AVAILABLE:
            raise ImportError("TokenPerformancePredictor not available. Cannot load CARROT models.")

        print(f"Loading unlimited predictors for {len(self.llm_pool)} LLMs...")

        for llm_name, llm_config in self.llm_pool.items():
            checkpoint_path = llm_config["checkpoint"]

            if not checkpoint_path.exists():
                print(f"⚠️  Checkpoint not found for {llm_name}: {checkpoint_path}")
                continue

            try:
                # Load predictor (which includes unlimited_score_predictor and unlimited_token_predictor)
                predictor = TokenPerformancePredictor(
                    token_limits=None,  # Will be loaded from checkpoint
                    load_dir=str(checkpoint_path)
                )
                self.predictors[llm_name] = predictor
                print(f"  ✅ Loaded predictor for {llm_name}")

            except Exception as e:
                print(f"  ⚠️  Failed to load {llm_name}: {e}")

        if len(self.predictors) == 0:
            raise RuntimeError("No predictors loaded successfully")

        print(f"✅ CARROT router ready with {len(self.predictors)} LLMs")

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
        if len(self.predictors) == 0:
            raise RuntimeError("No predictors loaded. Call load_model() first.")

        # Ensure embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Build routing options using unlimited predictors
        all_options = []

        for llm_name, predictor in self.predictors.items():
            llm_size = self.llm_pool[llm_name]["size"]

            # Get predictions using unlimited predictors
            # predict() returns (quality_limited, quality_unlimited, token_count)
            _, quality_unlimited, token_count_unlimited = predictor.predict(embedding)

            # Use unlimited predictions
            predicted_score = float(quality_unlimited[0])  # Shape: (1,)
            predicted_count = float(token_count_unlimited[0])  # Shape: (1,)

            # Safety: Clip quality scores to [0, 1]
            predicted_score = np.clip(predicted_score, 0, 1)
            # Safety: Ensure token counts are non-negative
            predicted_count = max(predicted_count, 0)

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

    def __init__(self, llm_pool: Dict = None):
        self.llm_pool = llm_pool or config.LLM_POOL
        self.token_limits = config.TOKEN_LIMITS
        print(f"⚠️  Using Mock CARROT router")

    def load_model(self):
        """No-op for mock"""
        print(f"✅ Mock CARROT ready")

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
        # Use embedding hash for deterministic selection
        seed = abs(hash(embedding.tobytes())) % (2**32)
        rng = np.random.RandomState(seed)

        all_options = []

        for llm_name, llm_config in self.llm_pool.items():
            llm_size = llm_config["size"]

            # Generate mock predictions for unlimited setting
            base_score = 0.65 + (llm_size * 0.25) + rng.uniform(-0.08, 0.08)
            base_score = np.clip(base_score, 0, 1)

            # Token count for unlimited
            predicted_count = rng.uniform(120, 550)
            predicted_cost = predicted_count * llm_size

            # Check budget
            if budget is not None and predicted_cost > budget:
                continue

            # Calculate risk
            risk = (1 - lambda_val) * base_score - lambda_val * predicted_cost

            all_options.append({
                "llm_name": llm_name,
                "token_limit": "unlimited",
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

def get_carrot_router(mock: bool = None) -> CARROTRouter:
    """
    Get a CARROT router instance

    Args:
        mock: Use mock router (uses config.ENABLE_MOCK_MODE if None)

    Returns:
        CARROTRouter or MockCARROTRouter
    """
    if mock is None:
        mock = config.ENABLE_MOCK_MODE

    if mock:
        return MockCARROTRouter()
    else:
        return CARROTRouter()


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing CARROT Router...")
    print()

    embedding = np.random.randn(1024)  # Use 1024-dim for main experiment embeddings

    # Test mock router
    print("=== Mock CARROT ===")
    router = MockCARROTRouter()
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

    # Test real router (only if checkpoints available)
    if not config.ENABLE_MOCK_MODE and PREDICTOR_AVAILABLE:
        print("=== Real CARROT Router ===")
        try:
            router = CARROTRouter()
            router.load_model()

            result = router.route(embedding, lambda_val=0.5)
            print(f"  Selected: {result['llm_name']} @ {result['token_limit']} tokens")
            print(f"  Predicted Score: {result['predicted_score']:.3f}")
            print(f"  Predicted Cost: {result['predicted_cost']:.1f}")

        except Exception as e:
            print(f"  ⚠️  Test failed: {e}")
    else:
        print("\n⚠️  Skipping real router test (mock mode or no predictor)")
