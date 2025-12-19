# Click Selection - Quick Summary

## What Was Added

Interactive click handlers for plot selection. Users can click on any point in CoRE curves or CARROT points to run inference with that option.

## Changes Made

### File: `demo/app.py`

1. **Lines 97-226**: New function `run_single_inference(query, llm_name, token_limit)`
   - Runs inference for single selected option
   - Gets predictions for that specific LLM + token limit
   - Calls LLM, evaluates, returns results HTML

2. **Lines 765-790**: Click handler `handle_plot_click(query, evt)`
   - Extracts LLM name and token limit from clicked point
   - Calls `run_single_inference()`
   - Returns results to display

3. **Lines 780-790**: Event connections
   - Connected both plots to click handler
   - `core_plot.select()` and `carrot_knn_plot.select()`

## How It Works

### User Flow
```
1. Enter query
2. Click "Show Visualizations"
3. See plots (CoRE curves + CARROT points)
4. Click on any data point
5. System runs inference with that option
6. Results appear below
```

### Data Flow
```
User clicks point
    ↓
Gradio captures SelectData
    ↓
Extract customdata: [llm_name, token_limit]
    ↓
run_single_inference()
    ↓
- Get predictions
- Call LLM
- Evaluate response
    ↓
Display results (HTML)
```

## Example

**Click on**: Qwen3-235B @ 150 tokens (on CoRE curve)

**System does:**
```
👆 User clicked: Qwen3-235B @ 150

🔄 Selected: Qwen3-235B @ 150
   → Predicted: score=0.850, cost=95.2

   Calling Qwen3-235B...
   Evaluating response...
   ✓ Actual: score=0.900, cost=92.5
```

**Shows:**
- Method: CoRE
- LLM: Qwen3-235B @ 150
- Predictions vs actuals
- Input prompt (highlighted)
- Response
- Judge evaluation

## Benefits

✅ **Interactive**: Click to explore different options
✅ **Visual**: Pick based on cost-quality curves/points
✅ **No lambda**: Selection is manual, not algorithmic
✅ **Immediate**: Results appear right away
✅ **Detailed**: Same metrics as Route Query mode

## Technical Details

- Uses Gradio's `.select()` event for Plot components
- Customdata stores `[llm_name, token_limit]` for each point
- Lambda=0 (quality-only) used in visualization mode
- Single result displayed (no comparison)

---

*Click any point on the plots to try it out!*
