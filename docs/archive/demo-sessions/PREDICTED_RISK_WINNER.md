# Predicted Risk Winner Selection

## Summary

Changed the demo to select the winner based on **predicted risk value** (routing decision) rather than actual risk value. This shows which routing method made the best decision based on its predictions.

## Changes Made

### 1. Replaced Score Error with Predicted Risk Value

**File: `demo/app.py`** (lines 454-467)

**Before:**
```python
<div class="metric-box">
    <div class="metric-label">Score Error</div>
    <div class="metric-value">
        {score_error:.3f}
    </div>
</div>
```

**After:**
```python
<div class="metric-box">
    <div class="metric-label">Predicted Risk Value</div>
    <div class="metric-value">{result["predicted_risk"]:.3f}</div>
</div>
```

### 2. Changed Winner Selection Logic

**File: `demo/app.py`** (lines 393-416)

**Before:**
```python
# Find best actual risk value for highlighting (highest actual risk = best)
best_risk = max(r["actual_risk"] for r in valid_results)
is_winner = result["actual_risk"] == best_risk
```

**After:**
```python
# Find best predicted risk value for highlighting (highest predicted risk = best)
# Winner is determined by predicted risk (routing decision)
best_predicted_risk = max(r["predicted_risk"] for r in valid_results)
is_winner = result["predicted_risk"] == best_predicted_risk
```

### 3. Updated Comparison Table Display

**File: `demo/app.py`** (line 414)

**Before:**
```python
<td><span class="good">{result["actual_risk"]:.3f}</span></td>
```

**After:**
```python
<td><span class="good">{result["predicted_risk"]:.3f}</span></td>
```

## What This Means

### Winner Selection
- **Trophy 🏆** now goes to the method with **highest predicted risk**
- This identifies which routing method made the **best routing decision**
- Predicted risk = `(1 - λ) × predicted_score - λ × predicted_cost`

### Display Structure

**Comparison Table:**
| Method | Selected LLM | Token Limit | Actual Score | Actual Cost | Risk Value |
|--------|-------------|-------------|--------------|-------------|------------|
| CoRE 🏆 | Qwen3-235B | 150 | 0.85 | 95.2 | **0.542** ← Predicted Risk |
| CARROT-KNN | Llama3-70B | unlimited | 0.90 | 150.0 | **0.480** ← Predicted Risk |

**Detailed Card Metrics:**
```
First Row (Header):
- Selected LLM: Qwen3-235B
- Token Limit: 150
- Actual Risk Value: 0.512  ← Shows actual performance

Second Row (Predictions):
- Predicted Score: 0.82
- Actual Score: 0.85
- Predicted Risk Value: 0.542  ← Used for winner selection

Third Row (Costs):
- Predicted Cost: 98.0
- Actual Cost: 95.2
- Actual Tokens: 952
```

## Interpretation

### Highest Predicted Risk = Best Router

The method with highest predicted risk made the **best routing decision** because:
1. It predicted high quality (score)
2. It predicted low cost
3. It optimally balanced quality vs cost according to λ

### Why Use Predicted Risk for Winner?

**Evaluates routing quality, not LLM quality:**
- Predicted risk shows which router made the smartest choice
- Actual risk would conflate router quality with LLM/judge variance
- This is more fair for comparing routing methods

**Example:**
```
CoRE predicts: score=0.9, cost=100 → predicted_risk = 0.60
Actual result: score=0.8, cost=95  → actual_risk = 0.52

CARROT predicts: score=0.7, cost=80 → predicted_risk = 0.46
Actual result: score=0.9, cost=75  → actual_risk = 0.67
```

With predicted risk winner:
- **CoRE wins 🏆** - Made better routing decision (0.60 > 0.46)
- Even though CARROT got lucky with actual performance

With actual risk winner:
- **CARROT wins** - But only due to LLM variance, not routing quality
- Doesn't reflect routing method quality

## Displayed Information

Users can see both:
1. **Predicted Risk** (in comparison table & detailed metrics)
   - Shows routing decision quality
   - Used for winner selection

2. **Actual Risk** (in detailed card header)
   - Shows real-world performance
   - For reference/debugging

## Benefits

✅ **Fair Comparison**: Evaluates routing methods, not LLM luck
✅ **Routing Quality**: Measures prediction accuracy, not outcome variance
✅ **Consistent**: Winner based on what router controlled (predictions)
✅ **Transparent**: Still shows actual risk for full picture

---

*The winner is now the routing method that made the best decision based on its predictions, not the one that happened to get the best outcome.*
