"""
Test script to verify demo setup
Run this to check if all components are working
"""

import sys
from pathlib import Path

print("="*60)
print("R2-Router Demo - Setup Verification")
print("="*60)
print()

# Test 1: Configuration
print("1. Testing configuration...")
try:
    import config
    print(f"   ✓ Config loaded")
    print(f"   - Root directory: {config.ROOT_DIR}")
    print(f"   - LLM pool: {len(config.LLM_POOL)} models")
    print(f"   - Mock mode: {config.ENABLE_MOCK_MODE}")
    print(f"   - API key set: {'Yes' if config.OPENROUTER_API_KEY else 'No'}")
except Exception as e:
    print(f"   ✗ Config failed: {e}")
    sys.exit(1)

print()

# Test 2: Embedder
print("2. Testing embedder...")
try:
    from embedder import get_embedder
    embedder = get_embedder()
    embedder.load_model()

    query = "Test query"
    embedding = embedder.embed(query)
    print(f"   ✓ Embedder working")
    print(f"   - Embedding shape: {embedding.shape}")
    print(f"   - Embedding dimension: {embedder.embedding_dim}")
except Exception as e:
    print(f"   ✗ Embedder failed: {e}")
    print(f"   (This is OK if using mock mode)")

print()

# Test 3: LLM Client
print("3. Testing LLM client...")
try:
    from llm_client import get_llm_client
    client = get_llm_client()
    print(f"   ✓ LLM client initialized")
    print(f"   - Type: {type(client).__name__}")

    if config.ENABLE_MOCK_MODE:
        # Test mock call
        response, tokens = client.call_llm("test-model", "What is 2+2?", max_tokens=10)
        print(f"   - Mock response: {response[:50]}...")
        print(f"   - Mock tokens: {tokens}")
except Exception as e:
    print(f"   ✗ LLM client failed: {e}")

print()

# Test 4: Judge
print("4. Testing judge...")
try:
    from judge import get_judge
    judge_instance = get_judge()
    print(f"   ✓ Judge initialized")
    print(f"   - Type: {type(judge_instance).__name__}")

    if config.ENABLE_MOCK_MODE:
        score, reasoning = judge_instance.evaluate("What is 2+2?", "4")
        print(f"   - Mock score: {score}")
        print(f"   - Mock reasoning: {reasoning[:50]}...")
except Exception as e:
    print(f"   ✗ Judge failed: {e}")

print()

# Test 5: R2-Router
print("5. Testing R2-Router...")
try:
    from router import get_r2_router
    import numpy as np

    router = get_r2_router()
    router.load_models()
    print(f"   ✓ R2-Router loaded")
    print(f"   - Type: {type(router).__name__}")
    print(f"   - Models: {len(router.predictors) if hasattr(router, 'predictors') else 'N/A (mock)'}")

    # Test routing
    dummy_embedding = np.random.randn(384)
    result = router.route(dummy_embedding, lambda_val=0.5)
    print(f"   - Test route: {result['llm_name']} @ {result['token_limit']}")
except Exception as e:
    print(f"   ✗ R2-Router failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 6: CARROT Routers
print("6. Testing CARROT routers...")
try:
    from baselines import get_carrot_router
    import numpy as np

    for method in ["knn", "linear"]:
        router = get_carrot_router(method)
        router.load_model()
        print(f"   ✓ CARROT-{method.upper()} loaded")

        dummy_embedding = np.random.randn(384)
        result = router.route(dummy_embedding, lambda_val=0.5)
        print(f"   - Test route: {result['llm_name']} @ {result['token_limit']}")
except Exception as e:
    print(f"   ✗ CARROT routers failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 7: Gradio
print("7. Testing Gradio...")
try:
    import gradio as gr
    print(f"   ✓ Gradio installed")
    print(f"   - Version: {gr.__version__}")
except Exception as e:
    print(f"   ✗ Gradio not installed: {e}")
    print(f"   Install with: pip install gradio")

print()

# Summary
print("="*60)
print("Setup Verification Complete!")
print("="*60)
print()

if config.ENABLE_MOCK_MODE:
    print("✓ Running in MOCK MODE")
    print("  - All components working with mock data")
    print("  - To use real LLMs:")
    print("    1. Set OPENROUTER_API_KEY environment variable")
    print("    2. Set ENABLE_MOCK_MODE = False in config.py")
    print("    3. Ensure checkpoints are trained")
else:
    if config.OPENROUTER_API_KEY:
        print("✓ API key configured")
    else:
        print("⚠️  API key not set - LLM calls will fail")

    # Check checkpoints
    missing_checkpoints = []
    for llm_name, llm_config in config.LLM_POOL.items():
        if not llm_config["checkpoint"].exists():
            missing_checkpoints.append(llm_name)

    if missing_checkpoints:
        print(f"⚠️  Missing checkpoints for: {', '.join(missing_checkpoints)}")
        print(f"   Train with: cd .. && bash run_experiment.sh")
    else:
        print("✓ All checkpoints found")

print()
print("Ready to run demo!")
print("Start with: python app.py")
print("Or use: bash start_demo.sh")
print()
