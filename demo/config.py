"""
Configuration for CoRE Router Demo
"""

import os
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================

# Root directory of the main project
ROOT_DIR = Path(__file__).parent.parent.absolute()
DEMO_DIR = Path(__file__).parent.absolute()

# Checkpoint directories
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints/main"
DATA_DIR = ROOT_DIR / "data"


# ============================================================================
# LLM Pool Configuration
# ============================================================================

# Available LLMs for routing
# Format: {name: {size: model_size_for_cost, openrouter_id: API_model_id}}
LLM_POOL = {
    "GLM-4.5-Air": {
        "size": 0.85,
        "openrouter_id": "z-ai/glm-4.5-air",  # Replace with actual GLM model ID
        "checkpoint": CHECKPOINTS_DIR / f"GLM-4.5-Air_ridge_alpha10.0",
        "csv": DATA_DIR / "GLM-4.5-Air.csv"
    },
    "Llama-3.1-70B-Instruct": {
        "size": 0.40,
        "openrouter_id": "meta-llama/llama-3.1-70b-instruct",
        "checkpoint": CHECKPOINTS_DIR / f"Llama-3.1-70B-Instruct_ridge_alpha10.0",
        "csv": DATA_DIR / "Llama-3.1-70B-Instruct.csv"
    },
    "Llama-3.2-3B-Instruct": {
        "size": 0.06,
        "openrouter_id": "meta-llama/llama-3.2-3b-instruct",
        "checkpoint": CHECKPOINTS_DIR / f"Llama-3.2-3B-Instruct_ridge_alpha10.0",
        "csv": DATA_DIR / "Llama-3.2-3B-Instruct.csv"
    },
    # "Qwen3-235B-A22B-Instruct-2507": {
    #     "size": 0.88,
    #     "openrouter_id": "qwen/qwen3-235b-a22b-2507",
    #     "checkpoint": CHECKPOINTS_DIR / f"Qwen3-235B-A22B-Instruct-2507_ridge_alpha10.0",
    #     "csv": DATA_DIR / "Qwen3-235B-A22B-Instruct-2507.csv"
    # },
    "Qwen3-0.6B": {
        "size": 0.0173,
        "openrouter_id": "qwen/qwen-2.5-7b-instruct",  # Use closest available
        "checkpoint": CHECKPOINTS_DIR / f"Qwen3-0.6B_ridge_alpha10.0",
        "csv": DATA_DIR / "Qwen3-0.6B.csv"
    }
}

# Token limits available (must match training)
TOKEN_LIMITS = [10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000, "unlimited"]

# ============================================================================
# CARROT Baseline Paths
# ============================================================================

CARROT_KNN_DIR = CHECKPOINTS_DIR / "carrot_knn"
CARROT_LINEAR_DIR = CHECKPOINTS_DIR / "carrot_linear"

# ============================================================================
# Embedding Configuration
# ============================================================================

# Embedding model to use
# Options: "vllm", "sentence-transformers", "openai", "huggingface"
EMBEDDING_MODEL = "vllm"

# vLLM Embedding Configuration (for Qwen3-0.6B)
VLLM_ENCODER_NAME = "Qwen/Qwen3-0.6B"  # Model name or path
VLLM_TENSOR_PARALLEL_SIZE = 2  # Number of GPUs for tensor parallelism
VLLM_MAX_MODEL_LEN = 1024  # Maximum sequence length
VLLM_BATCH_SIZE = 32  # Batch size for embedding generation
VLLM_DTYPE = "bfloat16"  # Data type for model

# Sentence-Transformers model name (fallback)
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # Fast and good quality
# Alternative: "all-mpnet-base-v2" (better quality, slower)

# Embedding dimension (will be auto-detected from model)
EMBEDDING_DIM = None  # Auto-detect (896 for Qwen2.5-Embedding-0.6B)

# ============================================================================
# Judge LLM Configuration
# ============================================================================

# Which LLM to use as judge
JUDGE_MODEL = "qwen/qwen3-next-80b-a3b-instruct"  # Qwen judge
# Alternatives:
# - "openai/gpt-4o-mini" (cost-effective)
# - "openai/gpt-4o" (best quality, more expensive)
# - "anthropic/claude-3.5-sonnet" (good balance)
# - "qwen/qwen-2.5-72b-instruct" (cheaper Qwen option)

# Judge prompt template
# Note: {golden_answer} is optional - if not provided, judge evaluates without reference
JUDGE_PROMPT_TEMPLATE = """You are a strict judge.
Compare the candidate answer with the golden answer(s).

Query: {query}

Golden Answer(s): {golden_answer}

Candidate Answer: {response}

Give a correctness score from {{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}}, and a brief justification.

Output strictly in JSON:
{{
    "correctness_score": <your_score>,
    "justification": "<your_reason>"
}}
"""

# Alternative: Judge without golden answer (for queries without ground truth)
JUDGE_PROMPT_TEMPLATE_NO_GOLDEN = """You are a strict judge.
Evaluate the quality and correctness of the candidate answer to the query.

Query: {query}

Candidate Answer: {response}

Give a correctness score from {{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}}, and a brief justification.

Output strictly in JSON:
{{
    "correctness_score": <your_score>,
    "justification": "<your_reason>"
}}
"""

# ============================================================================
# API Configuration
# ============================================================================

# OpenRouter API key (set via environment variable)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")

# API endpoints
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Request timeouts (seconds)
LLM_TIMEOUT = 30
JUDGE_TIMEOUT = 20

# ============================================================================
# Demo UI Configuration
# ============================================================================

# Default values for UI
DEFAULT_LAMBDA = 0.5
DEFAULT_BUDGET = None  # None = unlimited
DEFAULT_METHODS = ["CoRE", "CARROT-KNN"]  # Default selected methods

# Available methods
AVAILABLE_METHODS = ["CoRE", "CARROT-KNN", "CARROT-Linear"]

# UI theme
GRADIO_THEME = "default"  # "default", "soft", "glass", etc.

# Maximum queries in batch mode
MAX_BATCH_QUERIES = 5

# Example queries to show users
EXAMPLE_QUERIES = [
    "Solve the equation: 2x + 5 = 13",
    "What is the capital of France?",
    "Explain what photosynthesis is in simple terms.",
    "Write a Python function to calculate factorial.",
    "What are the main causes of climate change?"
]

# ============================================================================
# Feature Flags
# ============================================================================

# Enable/disable features
ENABLE_BATCH_MODE = True
ENABLE_CACHING = False  # Cache embeddings and routing results
ENABLE_LOGGING = True  # Log queries and results
ENABLE_MOCK_MODE = False  # Use real vLLM embeddings and API (set to True for testing without GPU/API)

# ============================================================================
# Cost Calculation
# ============================================================================

# Cost per token per unit of model size
# Adjust based on actual pricing
COST_PER_TOKEN_PER_SIZE = 0.000001  # Example: $1e-6 per token per size unit

def calculate_cost(token_count: int, model_size: float) -> float:
    """Calculate cost for a given token count and model size"""
    return token_count * model_size * COST_PER_TOKEN_PER_SIZE

# ============================================================================
# Validation
# ============================================================================

def validate_config():
    """Validate configuration on startup"""
    errors = []

    # Check API key
    if not OPENROUTER_API_KEY and not ENABLE_MOCK_MODE:
        errors.append("OPENROUTER_API_KEY environment variable not set")

    # Check checkpoint directories exist
    if not CHECKPOINTS_DIR.exists():
        errors.append(f"Checkpoints directory not found: {CHECKPOINTS_DIR}")

    # Check CARROT checkpoints
    if not CARROT_KNN_DIR.exists():
        errors.append(f"CARROT-KNN checkpoint not found: {CARROT_KNN_DIR}")
    if not CARROT_LINEAR_DIR.exists():
        errors.append(f"CARROT-Linear checkpoint not found: {CARROT_LINEAR_DIR}")

    # Check CoRE checkpoints for each LLM
    for llm_name, llm_config in LLM_POOL.items():
        checkpoint = llm_config["checkpoint"]
        if not checkpoint.exists():
            errors.append(f"CoRE checkpoint for {llm_name} not found: {checkpoint}")

    if errors:
        print("⚠️  Configuration Warnings:")
        for error in errors:
            print(f"  - {error}")

        if not ENABLE_MOCK_MODE:
            print("\n💡 Set ENABLE_MOCK_MODE = True in config.py to test without checkpoints/API")
    else:
        print("✅ Configuration validated successfully")

    return len(errors) == 0

if __name__ == "__main__":
    print("Configuration:")
    print(f"  Root directory: {ROOT_DIR}")
    print(f"  Checkpoints: {CHECKPOINTS_DIR}")
    print(f"  Training scheme: {TRAINING_SCHEME}")
    print(f"  LLM pool: {len(LLM_POOL)} models")
    print(f"  Token limits: {len(TOKEN_LIMITS)} limits")
    print(f"  Judge model: {JUDGE_MODEL}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  Mock mode: {ENABLE_MOCK_MODE}")
    print()
    validate_config()
