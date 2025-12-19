# Predicted Risk Winner - Quick Summary

## What Changed

Winner selection now based on **predicted risk** (routing decision quality) instead of actual risk (outcome quality).

## Changes Made

### File: `demo/app.py`

1. **Lines 464-465**: Replaced "Score Error" with "Predicted Risk Value"
   ```python
   <div class="metric-label">Predicted Risk Value</div>
   <div class="metric-value">{result["predicted_risk"]:.3f}</div>
   ```

2. **Line 398**: Find best predicted risk for winner
   ```python
   best_predicted_risk = max(r["predicted_risk"] for r in valid_results)
   ```

3. **Line 404**: Award trophy based on predicted risk
   ```python
   is_winner = result["predicted_risk"] == best_predicted_risk
   ```

4. **Line 414**: Display predicted risk in comparison table
   ```python
   <td>{result["predicted_risk"]:.3f}</td>
   ```

## Winner Logic

```
Predicted Risk = (1 - λ) × predicted_score - λ × predicted_cost

Highest Predicted Risk = Winner 🏆
```

## What's Displayed

| Location | Metric | Purpose |
|----------|--------|---------|
| Comparison Table | **Predicted Risk** | Winner selection |
| Detailed Card Header | Actual Risk | Real performance |
| Detailed Card Metrics | **Predicted Risk** | Routing decision |

## Why Predicted Risk?

✅ **Evaluates router quality** (not LLM/judge luck)
✅ **Fair comparison** (based on predictions, not variance)
✅ **Measures routing decisions** (what the method controlled)

## Example

```
CoRE:    predicted_risk = 0.60 🏆 (WINS - best routing decision)
         actual_risk = 0.52

CARROT:  predicted_risk = 0.46
         actual_risk = 0.67 (got lucky with outcome)
```

CoRE wins because it made the **better routing decision**, even though CARROT happened to get better actual results.

---

*Winner = Best routing decision (predicted risk), not best outcome (actual risk)*
