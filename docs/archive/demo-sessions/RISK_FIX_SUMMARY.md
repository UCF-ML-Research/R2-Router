# Actual Risk Fix - Quick Summary

## What Was Wrong

The demo was showing **predicted risk** (based on predicted score/cost) instead of **actual risk** (based on actual score/cost from judge evaluation). Trophy was awarded incorrectly.

## What Changed

### File: `demo/app.py`

1. **Line 156**: Calculate actual risk after judge evaluation
   ```python
   actual_risk = (1 - lambda_val) * actual_score - lambda_val * actual_cost
   ```

2. **Lines 170-173**: Store both predicted and actual risk
   ```python
   "predicted_risk": routing_result["risk"],  # For reference
   "actual_risk": actual_risk,  # For comparison
   ```

3. **Line 396**: Find best actual risk (not predicted)
   ```python
   best_risk = max(r["actual_risk"] for r in valid_results)
   ```

4. **Line 402**: Award trophy based on actual risk
   ```python
   is_winner = result["actual_risk"] == best_risk
   ```

5. **Line 412**: Display actual risk in comparison table
   ```python
   <td>{result["actual_risk"]:.3f}</td>
   ```

6. **Line 450**: Show actual risk in detailed card
   ```python
   <div class="metric-label">Actual Risk Value</div>
   <div class="metric-value">{result["actual_risk"]:.3f}</div>
   ```

## Risk Formula

```
Actual Risk = (1 - λ) × actual_score - λ × actual_cost
```

**Highest actual risk = Best method** (best quality-cost tradeoff in reality)

## Before vs After

| Aspect | Before (Wrong) | After (Correct) |
|--------|----------------|-----------------|
| Risk shown | Predicted risk | **Actual risk** ✅ |
| Trophy basis | Predicted performance | **Actual performance** ✅ |
| Winner | Best routing decision | **Best real-world result** ✅ |
| Reflects | Model predictions | **Ground truth from judge** ✅ |

## Why This Matters

- **Before**: CoRE could win with predicted_risk=0.8 but actual_risk=0.3 (bad execution)
- **After**: Winner is the method that actually performed best according to the judge

---
*Now the demo truthfully shows which routing method achieved the best quality-cost tradeoff in reality!*
