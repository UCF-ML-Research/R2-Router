# -*- coding: utf-8 -*-
"""
Simple API test for refactored predictor_sklearn.py (no dependencies required).
Tests that the API signatures are correct without running actual training.
"""

import inspect
from predictor_sklearn import TokenPerformancePredictor, train_predictor_from_dataset

print("=" * 80)
print("Testing Refactored predictor_sklearn.py API")
print("=" * 80)

# Test 1: Check __init__ signature
print("\nTest 1: Check __init__ signature")
init_sig = inspect.signature(TokenPerformancePredictor.__init__)
init_params = list(init_sig.parameters.keys())
print(f"__init__ parameters: {init_params}")
assert 'self' in init_params
assert 'token_limits' in init_params
assert 'load_dir' in init_params
# Should NOT have 'dataset' as a required parameter
assert init_sig.parameters['token_limits'].default == inspect.Parameter.empty or \
       init_sig.parameters['token_limits'].default is None
print("[OK] __init__ has correct signature (token_limits, load_dir)")

# Test 2: Check fit() signature
print("\nTest 2: Check fit() signature")
fit_sig = inspect.signature(TokenPerformancePredictor.fit)
fit_params = list(fit_sig.parameters.keys())
print(f"fit() parameters: {fit_params}")
assert 'X_train' in fit_params
assert 'y_train' in fit_params
assert 'token_count_train' in fit_params
assert 'X_test' in fit_params
assert 'save_dir' in fit_params
print("[OK] fit() has correct signature (follows sklearn pattern)")

# Test 3: Check predict() signature
print("\nTest 3: Check predict() signature")
predict_sig = inspect.signature(TokenPerformancePredictor.predict)
predict_params = list(predict_sig.parameters.keys())
print(f"predict() parameters: {predict_params}")
assert 'X' in predict_params
assert len(predict_params) == 2  # self, X
print("[OK] predict() has correct signature")

# Test 4: Check backward compatibility function exists
print("\nTest 4: Check train_predictor_from_dataset() exists")
compat_sig = inspect.signature(train_predictor_from_dataset)
compat_params = list(compat_sig.parameters.keys())
print(f"train_predictor_from_dataset() parameters: {compat_params}")
assert 'dataset' in compat_params
assert 'save_dir' in compat_params
print("[OK] Backward compatibility function exists")

# Test 5: Check that TokenPerformancePredictor can be instantiated
print("\nTest 5: Check instantiation")
try:
    # Should work with just token_limits
    predictor1 = TokenPerformancePredictor(token_limits=['10_score', '20_score'])
    print("[OK] Can instantiate with token_limits only")
except Exception as e:
    print(f"[FAIL] Cannot instantiate: {e}")
    raise

try:
    # Should work with load_dir only
    predictor2 = TokenPerformancePredictor(load_dir='./nonexistent')
    print("[OK] Can instantiate with load_dir only (warnings expected)")
except Exception as e:
    print(f"[FAIL] Cannot instantiate with load_dir: {e}")
    raise

print("\n" + "=" * 80)
print("ALL API TESTS PASSED!")
print("=" * 80)
print("\nKey improvements:")
print("1. Dataset is NOT an attribute of the model class")
print("2. Data is passed to fit() and predict() methods (sklearn pattern)")
print("3. Backward compatibility maintained via train_predictor_from_dataset()")
print("4. Cleaner separation of concerns: model vs data")
