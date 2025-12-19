"""
Basic test of demo functionality without Gradio
Tests the core routing pipeline
"""

import sys
from pathlib import Path

# Add demo directory to path
demo_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(demo_dir))

print("="*60)
print("Testing CoRE Router Demo (Mock Mode)")
print("="*60)
print()

# Test 1: Configuration
print("1. Loading configuration...")
import config
config.ENABLE_MOCK_MODE = True  # Force mock mode
print(f"   ✓ Mock mode enabled")
print(f"   LLM pool: {len(config.LLM_POOL)} models")
print()

# Test 2: Embedder
print("2. Testing embedder...")
from embedder import get_embedder
embedder = get_embedder()
embedder.load_model()

query = "What is the capital of France?"
embedding = embedder.embed(query)
print(f"   ✓ Query embedded: {embedding.shape}")
print()

# Test 3: Routers
print("3. Testing routers...")
from router import get_core_router
from baselines import get_carrot_router

# CoRE
core_router = get_core_router()
core_router.load_models()
print(f"   ✓ CoRE router loaded")

# CARROT-KNN
knn_router = get_carrot_router("knn")
knn_router.load_model()
print(f"   ✓ CARROT-KNN loaded")

# CARROT-Linear
linear_router = get_carrot_router("linear")
linear_router.load_model()
print(f"   ✓ CARROT-Linear loaded")
print()

# Test 4: Routing
print("4. Testing routing (lambda=0.5)...")
lambda_val = 0.5

core_result = core_router.route(embedding, lambda_val=lambda_val)
print(f"   CoRE: {core_result['llm_name']} @ {core_result['token_limit']} tokens")
print(f"         score={core_result['predicted_score']:.3f}, cost={core_result['predicted_cost']:.1f}")

knn_result = knn_router.route(embedding, lambda_val=lambda_val)
print(f"   KNN:  {knn_result['llm_name']} @ {knn_result['token_limit']} tokens")
print(f"         score={knn_result['predicted_score']:.3f}, cost={knn_result['predicted_cost']:.1f}")

linear_result = linear_router.route(embedding, lambda_val=lambda_val)
print(f"   LIN:  {linear_result['llm_name']} @ {linear_result['token_limit']} tokens")
print(f"         score={linear_result['predicted_score']:.3f}, cost={linear_result['predicted_cost']:.1f}")
print()

# Test 5: LLM Client
print("5. Testing LLM client...")
from llm_client import get_llm_client

client = get_llm_client()

llm_name = core_result['llm_name']
token_limit = core_result['token_limit']

response, actual_tokens = client.call_llm_by_name(llm_name, query, token_limit)
print(f"   ✓ LLM response: {response[:50]}...")
print(f"   Actual tokens: {actual_tokens}")
print()

# Test 6: Judge
print("6. Testing judge...")
from judge import get_judge

judge_instance = get_judge()
score, reasoning = judge_instance.evaluate(query, response)
print(f"   ✓ Judge score: {score:.3f}")
print(f"   Reasoning: {reasoning[:60]}...")
print()

# Test 7: Full Pipeline
print("7. Testing full pipeline...")
print(f"   Query: {query}")
print(f"   Lambda: {lambda_val}")
print()

for method_name, router in [("CoRE", core_router), ("CARROT-KNN", knn_router)]:
    print(f"   === {method_name} ===")

    # Route
    result = router.route(embedding, lambda_val=lambda_val)
    llm = result['llm_name']
    limit = result['token_limit']

    print(f"   Selected: {llm} @ {limit} tokens")
    print(f"   Predicted: score={result['predicted_score']:.3f}, cost={result['predicted_cost']:.1f}")

    # Call LLM
    resp, tokens = client.call_llm_by_name(llm, query, limit)

    # Evaluate
    actual_score, reason = judge_instance.evaluate(query, resp)

    llm_size = config.LLM_POOL[llm]["size"]
    actual_cost = tokens * llm_size

    print(f"   Actual: score={actual_score:.3f}, cost={actual_cost:.1f}")
    print(f"   Error: score={abs(actual_score - result['predicted_score']):.3f}, "
          f"cost={abs(actual_cost - result['predicted_cost']):.1f}")
    print()

print("="*60)
print("✅ All tests passed!")
print("="*60)
print()
print("Demo components are working correctly in mock mode.")
print("To run the full Gradio web interface:")
print("  1. Install Gradio: pip install gradio")
print("  2. Run: python app.py")
print()
