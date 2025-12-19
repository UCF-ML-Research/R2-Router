#!/usr/bin/env python3
"""
Test script to verify dual metric calculation (normalized vs actual cost).

Verifies that:
1. Normalized metrics are in [0,1] range
2. Actual metrics use raw cost values
3. Both metrics correctly identify 90% peak performance
"""

import numpy as np

# Standalone implementations to test the logic
def calculate_audc(cost_curve, perf_curve, normalize=True):
    """Area Under Deferral Curve."""
    sorted_indices = np.argsort(cost_curve)
    sorted_cost = cost_curve[sorted_indices]
    sorted_perf = perf_curve[sorted_indices]

    if normalize:
        if sorted_cost.max() > sorted_cost.min():
            x_values = (sorted_cost - sorted_cost.min()) / (sorted_cost.max() - sorted_cost.min())
        else:
            x_values = np.zeros_like(sorted_cost)
    else:
        x_values = sorted_cost

    return np.trapz(sorted_perf, x_values)

def calculate_qnc(cost_curve, perf_curve, normalize=True):
    """Query-Normalized Cost at 90% peak performance."""
    peak_perf = np.max(perf_curve)
    target_perf = 0.9 * peak_perf

    sorted_indices = np.argsort(cost_curve)
    sorted_cost = cost_curve[sorted_indices]
    sorted_perf = perf_curve[sorted_indices]

    idx = np.where(sorted_perf >= target_perf)[0]
    if len(idx) == 0:
        if normalize:
            return 1.0
        else:
            return sorted_cost[-1]

    cost_at_target = sorted_cost[idx[0]]

    if normalize:
        if sorted_cost.max() > sorted_cost.min():
            normalized_cost = (cost_at_target - sorted_cost.min()) / (sorted_cost.max() - sorted_cost.min())
        else:
            normalized_cost = 0.0
        return normalized_cost
    else:
        return cost_at_target

def test_metrics():
    """Test metric calculations with synthetic data."""
    print("=" * 80)
    print("TESTING DUAL METRIC CALCULATION")
    print("=" * 80)

    # Synthetic data: increasing cost, increasing then plateauing performance
    cost_curve = np.array([10, 50, 100, 500, 1000, 5000, 10000])
    perf_curve = np.array([0.3, 0.5, 0.7, 0.85, 0.95, 0.97, 0.98])

    print("\nTest Data:")
    print(f"Cost curve:        {cost_curve}")
    print(f"Performance curve: {perf_curve}")
    print(f"Peak performance:  {perf_curve.max()}")
    print(f"90% of peak:       {0.9 * perf_curve.max():.3f}")

    # Test normalized metrics
    print("\n" + "-" * 80)
    print("NORMALIZED COST METRICS:")
    print("-" * 80)

    audc_norm = calculate_audc(cost_curve, perf_curve, normalize=True)
    qnc_norm = calculate_qnc(cost_curve, perf_curve, normalize=True)

    print(f"AUDC (normalized): {audc_norm:.4f}")
    print(f"QNC (normalized):  {qnc_norm:.4f}")

    # Verify normalized cost is in [0,1]
    assert 0 <= audc_norm <= 1, f"AUDC normalized should be in [0,1], got {audc_norm}"
    assert 0 <= qnc_norm <= 1, f"QNC normalized should be in [0,1], got {qnc_norm}"
    print("✓ Normalized metrics are in [0,1] range")

    # Test actual cost metrics
    print("\n" + "-" * 80)
    print("ACTUAL COST METRICS:")
    print("-" * 80)

    audc_actual = calculate_audc(cost_curve, perf_curve, normalize=False)
    qnc_actual = calculate_qnc(cost_curve, perf_curve, normalize=False)

    print(f"AUDC (actual):     {audc_actual:.2f}")
    print(f"QNC (actual):      {qnc_actual:.2f}")

    # Verify actual cost uses raw values
    assert qnc_actual in cost_curve, f"QNC actual should be one of the cost values: {cost_curve}"
    print(f"✓ QNC actual cost ({qnc_actual}) is from the cost curve")

    # Find which cost point achieves 90% of peak
    target_perf = 0.9 * perf_curve.max()
    idx = np.where(perf_curve >= target_perf)[0][0]
    expected_qnc_actual = cost_curve[idx]

    assert qnc_actual == expected_qnc_actual, \
        f"QNC actual should be {expected_qnc_actual}, got {qnc_actual}"
    print(f"✓ QNC correctly identifies cost at 90% peak: index {idx}, cost {qnc_actual}")

    # Verify AUDC values are different
    print("\n" + "-" * 80)
    print("COMPARISON:")
    print("-" * 80)
    print(f"AUDC ratio (actual/normalized): {audc_actual/audc_norm:.2f}x")
    print(f"QNC ratio (actual/normalized):  {qnc_actual/qnc_norm:.2f}x")
    assert audc_actual != audc_norm, "Actual and normalized AUDC should differ"
    assert qnc_actual != qnc_norm, "Actual and normalized QNC should differ"
    print("✓ Normalized and actual metrics produce different values")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)

    # Test edge case: constant cost
    print("\nTesting edge case: constant cost...")
    const_cost = np.array([100, 100, 100, 100])
    const_perf = np.array([0.2, 0.4, 0.6, 0.8])

    audc_norm_edge = calculate_audc(const_cost, const_perf, normalize=True)
    audc_actual_edge = calculate_audc(const_cost, const_perf, normalize=False)

    print(f"AUDC normalized (constant cost): {audc_norm_edge:.4f}")
    print(f"AUDC actual (constant cost):     {audc_actual_edge:.2f}")
    assert audc_norm_edge == 0.0, "AUDC normalized should be 0 when cost is constant"
    print("✓ Edge case handled correctly")

    print("\n" + "=" * 80)
    print("COMPREHENSIVE TESTING COMPLETE ✓")
    print("=" * 80)

if __name__ == "__main__":
    test_metrics()
