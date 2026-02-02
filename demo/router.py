"""
R2-Router Module
Implements the R2-Router routing logic
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Ensure demo directory is first in path
demo_dir = Path(__file__).parent.absolute()
if str(demo_dir) not in sys.path:
    sys.path.insert(0, str(demo_dir))

# Add parent directory to path to import from main codebase (for predictor_sklearn)
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import config

try:
    from main.r2.predictor_sklearn import TokenPerformancePredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    print("⚠️  predictor_sklearn not available")


class R2Router:
    """R2-Router routing system that selects best (LLM, token_limit) combination"""

    def __init__(self, llm_pool: Dict = None):
        """
        Initialize R2-Router

        Args:
            llm_pool: LLM pool configuration (uses config.LLM_POOL if None)
        """
        self.llm_pool = llm_pool or config.LLM_POOL
        self.predictors = {}
        self.token_limits = config.TOKEN_LIMITS

        print(f"Initializing R2-Router with {len(self.llm_pool)} LLMs...")

    def load_models(self):
        """Load all R2-Router predictors from checkpoints"""
        if not PREDICTOR_AVAILABLE:
            raise ImportError("predictor_sklearn not available. Cannot load R2-Router models.")

        for llm_name, llm_config in self.llm_pool.items():
            checkpoint_dir = llm_config["checkpoint"]

            if not checkpoint_dir.exists():
                print(f"  ⚠️  Checkpoint not found for {llm_name}: {checkpoint_dir}")
                continue

            try:
                predictor = TokenPerformancePredictor(load_dir=str(checkpoint_dir))
                self.predictors[llm_name] = predictor
                print(f"  ✓ Loaded R2-Router model for {llm_name}")
            except Exception as e:
                print(f"  ✗ Failed to load {llm_name}: {e}")

        if len(self.predictors) == 0:
            raise RuntimeError("No R2-Router models loaded. Check checkpoint paths.")

        print(f"✅ R2-Router ready ({len(self.predictors)} models loaded)")

    def route(
        self,
        embedding: np.ndarray,
        lambda_val: float = 0.5,
        budget: Optional[float] = None
    ) -> Dict:
        """
        Route a query to the best (LLM, token_limit)

        Args:
            embedding: Query embedding vector
            lambda_val: Tradeoff parameter (0=quality, 1=cost)
            budget: Maximum cost budget (None=unlimited)

        Returns:
            Dictionary with routing decision:
            {
                "llm_name": str,
                "token_limit": int or "unlimited",
                "predicted_score": float,
                "predicted_cost": float,
                "risk": float,
                "all_options": List[Dict]  # All evaluated options
            }
        """
        if len(self.predictors) == 0:
            raise RuntimeError("No models loaded. Call load_models() first.")

        # Ensure embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        all_options = []

        # Evaluate all (LLM, token_limit) combinations
        for llm_name, predictor in self.predictors.items():
            llm_size = self.llm_pool[llm_name]["size"]

            try:
                # Get all predictions at once using the actual API
                # Returns: (quality_limited, quality_unlimited, token_count_unlimited)
                quality_limited, quality_unlimited, token_count_unlimited = predictor.predict(embedding)

                # quality_limited has shape (1, 15) for 15 limited token limits
                # quality_unlimited has shape (1,)
                # token_count_unlimited has shape (1,)

                # Safety: Clip quality scores to [0, 1] to handle sklearn unbounded predictions
                quality_limited = np.clip(quality_limited, 0, 1)
                quality_unlimited = np.clip(quality_unlimited, 0, 1)
                # Safety: Ensure token counts are non-negative
                token_count_unlimited = np.maximum(token_count_unlimited, 0)

                # Improved cost prediction logic:
                # Compare quality scores across all limits (including unlimited)
                # to determine which setting would actually be chosen

                # Collect all score predictions
                limited_token_limits = [t for t in self.token_limits if t != "unlimited"]
                all_scores = []

                # Add limited token limit scores
                for idx, token_limit in enumerate(limited_token_limits):
                    all_scores.append({
                        "limit": token_limit,
                        "score": quality_limited[0, idx],
                        "is_unlimited": False
                    })

                # Add unlimited score
                all_scores.append({
                    "limit": "unlimited",
                    "score": quality_unlimited[0],
                    "is_unlimited": True
                })

                # Find best quality score
                best_score_option = max(all_scores, key=lambda x: x["score"])

                # Determine predicted token count based on best quality option
                if best_score_option["is_unlimited"]:
                    # If unlimited gives best quality, use predicted token count
                    predicted_best_token_count = token_count_unlimited[0]
                else:
                    # If a limited setting gives best quality, use that limit
                    # (model will likely use up to the limit)
                    predicted_best_token_count = best_score_option["limit"]

                # Now process each option with improved cost prediction
                for idx, token_limit in enumerate(limited_token_limits):
                    predicted_score = quality_limited[0, idx]

                    # Improved cost prediction:
                    # If this limit's score equals the best score, use the predicted token count
                    # Otherwise, assume the model will use the full limit (trying to maximize quality)
                    if predicted_score >= best_score_option["score"] - 0.01:  # Close to best
                        predicted_count = min(token_limit, predicted_best_token_count)
                    else:
                        # Model will likely use full limit trying to reach better quality
                        predicted_count = token_limit

                    predicted_cost = predicted_count * llm_size

                    # Check budget constraint
                    if budget is not None and predicted_cost > budget:
                        continue

                    # Calculate risk: (1-λ) * score - λ * cost
                    risk = (1 - lambda_val) * predicted_score - lambda_val * predicted_cost

                    all_options.append({
                        "llm_name": llm_name,
                        "token_limit": token_limit,
                        "predicted_score": float(predicted_score),
                        "predicted_count": float(predicted_count),
                        "predicted_cost": float(predicted_cost),
                        "risk": float(risk)
                    })

                # Process unlimited token limit
                predicted_score = quality_unlimited[0]
                predicted_count = token_count_unlimited[0]
                predicted_cost = predicted_count * llm_size

                # Check budget constraint
                if budget is None or predicted_cost <= budget:
                    risk = (1 - lambda_val) * predicted_score - lambda_val * predicted_cost

                    all_options.append({
                        "llm_name": llm_name,
                        "token_limit": "unlimited",
                        "predicted_score": float(predicted_score),
                        "predicted_count": float(predicted_count),
                        "predicted_cost": float(predicted_cost),
                        "risk": float(risk)
                    })

            except Exception as e:
                print(f"  ⚠️  Error predicting for {llm_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(all_options) == 0:
            raise RuntimeError("No valid routing options found (check budget constraints)")

        # Select option with maximum risk
        best_option = max(all_options, key=lambda x: x["risk"])
        best_option["all_options"] = all_options

        return best_option


class MockR2Router:
    """Mock R2-Router for testing without checkpoints"""

    def __init__(self, llm_pool: Dict = None):
        self.llm_pool = llm_pool or config.LLM_POOL
        self.token_limits = config.TOKEN_LIMITS
        print(f"⚠️  Using MockR2Router ({len(self.llm_pool)} LLMs)")

    def load_models(self):
        """No-op for mock router"""
        print("✅ Mock R2-Router ready")

    def route(
        self,
        embedding: np.ndarray,
        lambda_val: float = 0.5,
        budget: Optional[float] = None
    ) -> Dict:
        """
        Mock routing decision based on heuristics

        Args:
            embedding: Query embedding (used for deterministic randomness)
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

            for token_limit in self.token_limits:
                # Generate mock predictions
                # Larger models tend to have higher scores
                base_score = 0.6 + (llm_size * 0.3) + rng.uniform(-0.1, 0.1)
                base_score = np.clip(base_score, 0, 1)

                # Token count depends on limit
                if token_limit == "unlimited":
                    predicted_count = rng.uniform(100, 500)
                else:
                    predicted_count = min(token_limit * 0.8, token_limit)

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
# Factory Function
# ============================================================================

def get_r2_router(mock: bool = None) -> R2Router:
    """
    Get a R2-Router instance

    Args:
        mock: Use mock router (uses config.ENABLE_MOCK_MODE if None)

    Returns:
        R2Router or MockR2Router
    """
    if mock is None:
        mock = config.ENABLE_MOCK_MODE

    if mock:
        return MockR2Router()
    else:
        return R2Router()


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing R2-Router...")
    print()

    # Test mock router
    print("=== Mock Router ===")
    mock_router = MockR2Router()
    mock_router.load_models()

    # Create dummy embedding
    embedding = np.random.randn(384)

    # Test routing with different lambda values
    for lambda_val in [0.0, 0.5, 1.0]:
        result = mock_router.route(embedding, lambda_val=lambda_val)
        print(f"\nLambda = {lambda_val}:")
        print(f"  Selected: {result['llm_name']} @ {result['token_limit']} tokens")
        print(f"  Predicted Score: {result['predicted_score']:.3f}")
        print(f"  Predicted Cost: {result['predicted_cost']:.1f}")
        print(f"  Risk: {result['risk']:.3f}")

    # Test with budget constraint
    print(f"\nWith budget = 5.0:")
    result = mock_router.route(embedding, lambda_val=0.5, budget=5.0)
    print(f"  Selected: {result['llm_name']} @ {result['token_limit']} tokens")
    print(f"  Predicted Cost: {result['predicted_cost']:.1f} (within budget)")

    # Test real router (only if checkpoints available)
    if not config.ENABLE_MOCK_MODE and PREDICTOR_AVAILABLE:
        print("\n=== Real Router ===")
        try:
            real_router = R2Router()
            real_router.load_models()

            result = real_router.route(embedding, lambda_val=0.5)
            print(f"\nRouting result:")
            print(f"  Selected: {result['llm_name']} @ {result['token_limit']} tokens")
            print(f"  Predicted Score: {result['predicted_score']:.3f}")
            print(f"  Predicted Cost: {result['predicted_cost']:.1f}")

        except Exception as e:
            print(f"⚠️  Real router test failed: {e}")
    else:
        print("\n⚠️  Skipping real router test (mock mode or no predictor)")
