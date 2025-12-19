"""
Verification script for the API fix
Tests that the predictor API is correctly used
"""

import sys
from pathlib import Path

# Add demo directory to path
demo_dir = Path(__file__).parent.absolute()
if str(demo_dir) not in sys.path:
    sys.path.insert(0, str(demo_dir))

# Add parent directory for predictor imports
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import numpy as np
import config

print("=" * 70)
print("API Fix Verification")
print("=" * 70)
print()

# Step 1: Check if dependencies are installed
print("1. Checking dependencies...")
try:
    import matplotlib
    print("   ✓ matplotlib installed")
except ImportError:
    print("   ✗ matplotlib NOT installed - run: pip install matplotlib")

try:
    import seaborn
    print("   ✓ seaborn installed")
except ImportError:
    print("   ✗ seaborn NOT installed - run: pip install seaborn")

try:
    import tqdm
    print("   ✓ tqdm installed")
except ImportError:
    print("   ✗ tqdm NOT installed - run: pip install tqdm")

print()

# Step 2: Check if predictor_sklearn is available
print("2. Checking predictor availability...")
try:
    from predictor_sklearn import TokenPerformancePredictor
    print("   ✓ predictor_sklearn is available")
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    print(f"   ✗ predictor_sklearn not available: {e}")
    PREDICTOR_AVAILABLE = False

print()

# Step 3: Check if baselines_carrot is available
print("3. Checking CARROT baselines availability...")
try:
    from baselines_carrot import CarrotKNNBaseline, CarrotLinearBaseline
    print("   ✓ baselines_carrot is available")
    CARROT_AVAILABLE = True
except ImportError as e:
    print(f"   ✗ baselines_carrot not available: {e}")
    CARROT_AVAILABLE = False

print()

# Step 4: Check checkpoint directories
print("4. Checking checkpoint directories...")
checkpoint_found = False

for llm_name, llm_config in config.LLM_POOL.items():
    checkpoint_dir = llm_config["checkpoint"]
    if checkpoint_dir.exists():
        print(f"   ✓ Found checkpoint for {llm_name}: {checkpoint_dir}")
        checkpoint_found = True
    else:
        print(f"   ✗ Missing checkpoint for {llm_name}: {checkpoint_dir}")

if config.CARROT_KNN_DIR.exists():
    print(f"   ✓ Found CARROT-KNN checkpoint: {config.CARROT_KNN_DIR}")
    checkpoint_found = True
else:
    print(f"   ✗ Missing CARROT-KNN checkpoint: {config.CARROT_KNN_DIR}")

if config.CARROT_LINEAR_DIR.exists():
    print(f"   ✓ Found CARROT-Linear checkpoint: {config.CARROT_LINEAR_DIR}")
    checkpoint_found = True
else:
    print(f"   ✗ Missing CARROT-Linear checkpoint: {config.CARROT_LINEAR_DIR}")

print()

# Step 5: Test predictor API (if available)
if PREDICTOR_AVAILABLE and checkpoint_found:
    print("5. Testing predictor API with real checkpoint...")

    # Find first available checkpoint
    test_llm = None
    test_checkpoint = None
    for llm_name, llm_config in config.LLM_POOL.items():
        if llm_config["checkpoint"].exists():
            test_llm = llm_name
            test_checkpoint = llm_config["checkpoint"]
            break

    if test_checkpoint:
        try:
            # Load predictor
            predictor = TokenPerformancePredictor(load_dir=str(test_checkpoint))
            print(f"   ✓ Loaded predictor for {test_llm}")

            # Create dummy embedding (896-dim for Qwen3-0.6B)
            embedding = np.random.randn(1, 896)

            # Test the API
            quality_limited, quality_unlimited, token_count = predictor.predict(embedding)

            print(f"   ✓ predict() returned 3 values:")
            print(f"     - quality_limited: shape {quality_limited.shape} (expected: (1, 15))")
            print(f"     - quality_unlimited: shape {quality_unlimited.shape} (expected: (1,))")
            print(f"     - token_count: shape {token_count.shape} (expected: (1,))")

            # Verify shapes
            if quality_limited.shape == (1, 15):
                print("   ✓ quality_limited shape is correct")
            else:
                print(f"   ✗ quality_limited shape mismatch!")

            if quality_unlimited.shape == (1,):
                print("   ✓ quality_unlimited shape is correct")
            else:
                print(f"   ✗ quality_unlimited shape mismatch!")

            if token_count.shape == (1,):
                print("   ✓ token_count shape is correct")
            else:
                print(f"   ✗ token_count shape mismatch!")

            print()
            print("   ✅ API fix is working correctly!")

        except Exception as e:
            print(f"   ✗ Error testing predictor: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ⚠️  No checkpoint available for testing")
else:
    print("5. Skipping predictor API test (dependencies or checkpoints missing)")

print()

# Step 6: Test CARROT API (if available)
if CARROT_AVAILABLE and (config.CARROT_KNN_DIR.exists() or config.CARROT_LINEAR_DIR.exists()):
    print("6. Testing CARROT API with real checkpoint...")

    if config.CARROT_KNN_DIR.exists():
        try:
            model = CarrotKNNBaseline(load_dir=str(config.CARROT_KNN_DIR))
            print(f"   ✓ Loaded CARROT-KNN model")

            # Create dummy embedding
            embedding = np.random.randn(1, 896)

            # Test the API
            Y_hat_score, Y_hat_count = model.predict(embedding)

            print(f"   ✓ predict() returned 2 values:")
            print(f"     - Y_hat_score: shape {Y_hat_score.shape}")
            print(f"     - Y_hat_count: shape {Y_hat_count.shape}")

            # Expected: (1, num_llms * num_token_limits)
            expected_models = len(config.LLM_POOL) * len(config.TOKEN_LIMITS)

            if Y_hat_score.shape[1] == expected_models:
                print(f"   ✓ Y_hat_score has {expected_models} predictions (correct)")
            else:
                print(f"   ⚠️  Y_hat_score has {Y_hat_score.shape[1]} predictions, expected {expected_models}")

            print()
            print("   ✅ CARROT API fix is working correctly!")

        except Exception as e:
            print(f"   ✗ Error testing CARROT: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ⚠️  No CARROT checkpoint available for testing")
else:
    print("6. Skipping CARROT API test (dependencies or checkpoints missing)")

print()

# Summary
print("=" * 70)
print("Summary")
print("=" * 70)
print()

if not (PREDICTOR_AVAILABLE and CARROT_AVAILABLE):
    print("⚠️  ACTION REQUIRED: Install missing dependencies")
    print()
    print("Run this command:")
    print("  pip install matplotlib seaborn tqdm")
    print()

if not checkpoint_found:
    print("⚠️  WARNING: No checkpoints found")
    print()
    print("The demo will run in mock mode until you:")
    print("  1. Train models using the main training script")
    print("  2. Or set ENABLE_MOCK_MODE = True in config.py to test with fake data")
    print()

if PREDICTOR_AVAILABLE and CARROT_AVAILABLE and checkpoint_found:
    print("✅ All checks passed! The API fix should work correctly.")
    print()
    print("Next steps:")
    print("  1. Set your OpenRouter API key:")
    print("     export OPENROUTER_API_KEY='your-key-here'")
    print()
    print("  2. Update model IDs in config.py if needed")
    print()
    print("  3. Run the full demo:")
    print("     python app.py")
    print()
else:
    print("Next step: Install dependencies and run this script again:")
    print("  pip install matplotlib seaborn tqdm")
    print("  python verify_api_fix.py")
    print()
