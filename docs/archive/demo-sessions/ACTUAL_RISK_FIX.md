# Actual Risk Value Fix

## Issue Identified

The demo was displaying **predicted risk values** (based on predicted scores/costs) instead of **actual risk values** (based on actual scores/costs from the judge evaluation). This made it unclear which method actually performed best in reality.

## Problem

**Before:**
- Risk value shown = `(1 - λ) × predicted_score - λ × predicted_cost`
- Trophy awarded based on predicted risk (routing decision)
- Does not reflect actual performance after LLM execution and judge evaluation

**After:**
- Risk value shown = `(1 - λ) × actual_score - λ × actual_cost`
- Trophy awarded based on actual risk (true performance)
- Reflects real-world performance with ground truth evaluation

## Changes Made

### 1. Calculate Actual Risk Value

**File: `demo/app.py`** (lines 154-158)

```python
# Calculate actual risk value using actual score and cost
# Risk = (1 - λ) × actual_score - λ × actual_cost
actual_risk = (1 - lambda_val) * actual_score - lambda_val * actual_cost

print(f"   ✓ Actual: score={actual_score:.3f}, cost={actual_cost:.1f}, risk={actual_risk:.3f}")
```

### 2. Store Both Predicted and Actual Risk

**File: `demo/app.py`** (lines 161-176)

```python
results.append({
    "method": method_name,
    "llm_name": llm_name,
    "token_limit": token_limit,
    "response": response,
    "actual_prompt": actual_prompt,
    "original_query": query,
    "predicted_score": routing_result["predicted_score"],
    "predicted_cost": routing_result["predicted_cost"],
    "predicted_risk": routing_result["risk"],  # Keep predicted risk for reference
    "actual_score": actual_score,
    "actual_cost": actual_cost,
    "actual_risk": actual_risk,  # Use actual risk for comparison
    "actual_tokens": actual_tokens,
    "reasoning": reasoning
})
```

### 3. Update Comparison Table

**File: `demo/app.py`** (lines 393-414)

**Changed:**
- Line 396: `max(r["risk"]...)` → `max(r["actual_risk"]...)`
- Line 402: `result["risk"] == best_risk` → `result["actual_risk"] == best_risk`
- Line 412: `result["risk"]:.3f` → `result["actual_risk"]:.3f`

**Result:**
- Trophy 🏆 now awarded to method with highest **actual risk**
- Risk Value column shows **actual risk** based on real performance

### 4. Update Detailed Results Card

**File: `demo/app.py`** (lines 449-450)

**Changed:**
- Label: "Risk Value" → "Actual Risk Value"
- Display: `result["risk"]` → `result["actual_risk"]`

## Risk Formula

```
Actual Risk = (1 - λ) × actual_score - λ × actual_cost

Where:
- actual_score: Judge's evaluation of the response (0.0 to 1.0)
- actual_cost: actual_tokens × llm_size
- λ (lambda): Cost-performance tradeoff parameter (0 to 1)
  - λ = 0: Maximize quality (ignore cost)
  - λ = 1: Minimize cost (ignore quality)
  - λ = 0.5: Balance quality and cost equally
```

## Why This Matters

### Before (Predicted Risk)
- Trophy could go to a method that made a good **prediction** but poor **execution**
- Example: R2-Router predicts score=0.9, but actual score=0.6 → predicted risk is high, but actual performance is poor

### After (Actual Risk)
- Trophy goes to method with best **real-world performance**
- Example: CARROT-KNN predicts score=0.7, actual score=0.9 → lower predicted risk, but wins due to better actual performance

## Interpretation

**Highest Actual Risk = Best Method** because:
1. Higher actual_score is better (more correct)
2. Lower actual_cost is better (more efficient)
3. Risk maximizes score while penalizing cost
4. The method with highest actual risk achieved the best quality-cost tradeoff **in reality**

## Example

Given λ = 0.3:

| Method | Actual Score | Actual Cost | Actual Risk | Winner? |
|--------|-------------|-------------|-------------|---------|
| R2-Router | 0.8 | 100 | (1-0.3)×0.8 - 0.3×100 = 0.56 - 30 = **-29.44** | ❌ |
| CARROT-KNN | 0.9 | 80 | (1-0.3)×0.9 - 0.3×80 = 0.63 - 24 = **-23.37** | 🏆 |

CARROT-KNN wins because it achieved higher quality (0.9 vs 0.8) at lower cost (80 vs 100), resulting in higher actual risk!

## Benefits

✅ **Truthful Comparison**: Shows which method actually performed best
✅ **Fair Evaluation**: Based on ground truth from judge, not predictions
✅ **Debugging**: Can compare predicted_risk vs actual_risk to assess routing accuracy
✅ **Research Value**: True performance metrics for paper/analysis

---
*This fix ensures the demo awards the trophy and highlights the winner based on actual performance, not predicted performance.*
