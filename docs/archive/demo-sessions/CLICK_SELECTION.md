# Click Selection Feature for Visualization Mode

## Summary

Implemented click handlers for interactive plot selection. Users can now click on any point in the R2-Router curves or CARROT points to run inference with that specific LLM and token limit.

## Implementation

### 1. Single Inference Function

**File: `demo/app.py`** (lines 97-226)

**Function: `run_single_inference(query, llm_name, token_limit)`**

**Purpose**: Run inference for a single user-selected (LLM, token_limit) combination

**Workflow:**
1. Generate embedding for query
2. Determine if selection is from R2-Router or CARROT
3. Get predictions for the specific option:
   - **R2-Router**: Extract prediction from specific token limit
   - **CARROT**: Extract prediction for the specific LLM (unlimited)
4. Call LLM with selected option
5. Evaluate response with judge
6. Calculate metrics (predicted/actual score, cost, risk)
7. Generate HTML results (single result, no comparison)

**Key Logic:**

```python
# Determine router
if llm_name in r2_router.predictors:
    method_name = "R2-Router"
    # Get prediction for specific token limit
    if token_limit == "unlimited":
        predicted_score = quality_unlimited[0]
    else:
        token_idx = limited_token_limits.index(int(token_limit))
        predicted_score = quality_limited[0, token_idx]
else:
    method_name = "CARROT-KNN"
    # Get prediction for specific LLM
    llm_idx = llm_names.index(llm_name)
    predicted_score = Y_hat_score[0, llm_idx]
```

**Risk Calculation:**
- Uses `lambda=0` (quality-only) for consistency in visualization mode
- `predicted_risk = predicted_score`
- `actual_risk = actual_score`

---

### 2. Click Handler

**File: `demo/app.py`** (lines 765-790)

**Function: `handle_plot_click(query, evt: gr.SelectData)`**

**Purpose**: Extract clicked point data and trigger inference

**Workflow:**
1. Gradio `.select()` event provides `SelectData` object
2. Extract `customdata` from clicked point
3. Parse `llm_name` and `token_limit` from customdata
4. Call `run_single_inference()` with extracted data
5. Return HTML results to display

**Implementation:**
```python
def handle_plot_click(query, evt: gr.SelectData):
    """Handle click on plot - extract LLM name and token limit from customdata"""
    if evt.value and 'customdata' in evt.value:
        customdata = evt.value['customdata']
        llm_name = customdata[0]
        token_limit = customdata[1]

        print(f"\n👆 User clicked: {llm_name} @ {token_limit}")

        # Run inference for selected option
        return run_single_inference(query, llm_name, token_limit)
    else:
        return "<p style='color: orange;'>⚠️ Please click directly on a data point.</p>"
```

---

### 3. Event Connections

**File: `demo/app.py`** (lines 780-790)

**Connected both plots to the handler:**

```python
core_plot.select(
    fn=handle_plot_click,
    inputs=[query_input],
    outputs=results_output
)

carrot_knn_plot.select(
    fn=handle_plot_click,
    inputs=[query_input],
    outputs=results_output
)
```

---

### 4. Customdata in Visualizations

**Already implemented in `demo/visualizer.py`:**

**R2-Router visualization** (line 102):
```python
customdata=[[llm_name, str(limit)] for limit in token_limits_sorted]
```

**CARROT visualization** (line 219):
```python
customdata=[[llm_name, "unlimited"]]
```

Each data point stores `[llm_name, token_limit]` in customdata, which is accessed when clicked.

---

## User Workflow

### Visualization Mode

1. **Enter query**
2. **Click "Show Visualizations"**
   - R2-Router curves appear (left)
   - CARROT points appear (right)
3. **Click on any point**:
   - For R2-Router: Click on any point on any curve
   - For CARROT: Click on any scatter point
4. **System runs inference**:
   - Calls selected LLM with selected token limit
   - Evaluates response
   - Shows results below
5. **Results displayed**:
   - Method used (R2-Router or CARROT-KNN)
   - Selected LLM and token limit
   - Predictions vs actuals
   - Input prompt (with highlighting)
   - Response
   - Judge reasoning

---

## Example Flow

**User action:**
```
1. Query: "What is the capital of France?"
2. Click "Show Visualizations"
3. See R2-Router curves showing different cost-quality points
4. Click on Qwen3-235B curve at token limit 150
```

**System response:**
```
👆 User clicked: Qwen3-235B @ 150

🔄 Selected: Qwen3-235B @ 150
   → Predicted: score=0.850, cost=95.2

   Calling Qwen3-235B...
   Evaluating response...
   ✓ Actual: score=0.900, cost=92.5, risk=0.900

Results displayed:
┌─────────────────────────────────────────┐
│ R2-Router                                    │
│ Selected: Qwen3-235B @ 150 tokens       │
│ Predicted Risk: 0.850                   │
│ Actual Risk: 0.900                      │
│                                         │
│ [Detailed metrics...]                   │
│ [Input prompt with highlighting...]     │
│ [Response...]                          │
│ [Judge reasoning...]                    │
└─────────────────────────────────────────┘
```

---

## Benefits

✅ **Interactive exploration**: Users can try different options by clicking
✅ **Visual selection**: Choose based on cost-quality tradeoff visualization
✅ **No lambda needed**: Visual selection is intuitive
✅ **Immediate feedback**: See results right after clicking
✅ **Detailed metrics**: Same detailed display as Route Query mode

---

## Technical Details

### SelectData Object

Gradio provides `gr.SelectData` when a plot is clicked:

```python
evt.value = {
    'x': <x_coordinate>,
    'y': <y_coordinate>,
    'customdata': [llm_name, token_limit],  # Set by us
    'curveNumber': <trace_index>,
    'pointNumber': <point_index>
}
```

We use `customdata` to pass LLM name and token limit.

### Error Handling

- If click misses data point: Shows warning message
- If query is empty: Returns error message
- If inference fails: Shows error with traceback

### Lambda Value

- Visualization mode uses `lambda=0` (quality-only)
- Risk = Score (cost doesn't affect risk)
- Consistent with "explore predictions" concept
- User picks visually, not algorithmically

---

## Comparison: Route Query vs Visualize

| Aspect | Route Query | Visualize & Click |
|--------|-------------|-------------------|
| **Input** | Lambda value | None (visual) |
| **Selection** | Algorithmic (risk max) | Manual (click point) |
| **Prediction** | All options compared | Single option |
| **Output** | Multiple results | Single result |
| **Use case** | Compare routing methods | Explore specific options |

---

*Click on any point in the visualizations to run inference with that specific LLM and token limit!*
