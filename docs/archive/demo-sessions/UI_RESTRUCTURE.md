# UI Restructuring - Two Equal Modes

## Summary

Restructured the demo UI to have two equal workflow modes side-by-side:
1. **Route Query**: Direct routing with lambda (removed budget)
2. **Visualize Predictions**: Show curves/points for user to click and select

## Major Changes

### 1. Removed Budget Input

**Motivation**: Budget constraint not useful for the demo

**Changes:**
- Removed `budget_input` textbox from UI
- Removed `budget_str` parameter from `process_query()`
- Removed `budget_str` parameter from `generate_visualizations()`
- Removed budget display from results HTML
- Always use `budget = None` (no constraint)

**Files Modified:**
- `demo/app.py` lines 107-132, 66-90, 390-393, 621-630

---

### 2. Restructured UI Layout

**New Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  Query Input [Large]         Example Queries [Small]    │
└─────────────────────────────────────────────────────────┘

┌──────────────────────┬──────────────────────────────────┐
│  🚀 Route Query      │  📊 Visualize Predictions        │
│  ├─ Lambda slider    │  ├─ Info text                    │
│  ├─ Method checkboxes│  └─ [Show Visualizations] button │
│  └─ [Route] button   │                                  │
└──────────────────────┴──────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  R2-Router Plot [50%]               CARROT Plot [50%]        │
│  (visible after visualization) (visible after viz)      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Results (HTML display)                                 │
└─────────────────────────────────────────────────────────┘
```

**Key Changes:**
- Two modes are now side-by-side columns (equal level)
- Removed tabs structure (no more "Visualizations" tab vs "Results" tab)
- Plots shown horizontally below the mode selection
- Both plots start hidden, appear after clicking "Show Visualizations"
- Results always shown below

**Files Modified:**
- `demo/app.py` lines 551-645

---

### 3. Mode 1: Route Query

**Purpose**: Traditional routing with lambda parameter

**Interface:**
```
### 🚀 Route Query
Select routing methods and lambda, then route directly.

Lambda (λ): Quality ← → Cost
[Slider: 0.0 to 1.0]

[✓] R2-Router    [✓] CARROT-KNN

[🚀 Route Query]
```

**Workflow:**
1. User sets lambda (cost-quality tradeoff)
2. User selects methods (R2-Router, CARROT-KNN, or both)
3. Click "Route Query"
4. System runs routing, calls LLM, evaluates, shows results

**No visualization involved** - direct routing based on risk maximization

---

### 4. Mode 2: Visualize Predictions

**Purpose**: Visual selection of LLM and token limit

**Interface:**
```
### 📊 Visualize Predictions
View cost-quality curves/points, click to select LLM and token limit.

[📊 Show Visualizations]
```

**Workflow:**
1. User clicks "Show Visualizations"
2. Plots appear showing predicted cost-quality for all options:
   - R2-Router: Curves with points for each token limit
   - CARROT: Points for each LLM (unlimited only)
3. **[TODO]** User clicks on a point to select
4. **[TODO]** System runs inference with selected option
5. **[TODO]** Shows results (same format as Route Query)

**Key Difference**: No lambda needed - user visually picks based on curves/points

---

### 5. Horizontal Plot Layout

**Before:**
- Plots stacked vertically (each full width)

**After:**
- Plots side-by-side horizontally (each 50% width)
- Uses `gr.Row()` with both plots inside

```python
with gr.Row():
    core_plot = gr.Plot(label="R2-Router: Cost-Quality Curves", visible=False)
    carrot_knn_plot = gr.Plot(label="CARROT-KNN: Cost-Quality Points", visible=False)
```

**Benefits:**
- Easy visual comparison
- Space-efficient
- Both methods visible at once

---

### 6. Dynamic Visibility

**Implementation:**
- Plots start with `visible=False`
- Clicking "Show Visualizations" triggers `show_visualizations()` function
- Returns `gr.update(value=fig, visible=True)` to show plots

```python
def show_visualizations(query):
    r2_fig, carrot_fig = generate_visualizations(query)
    return {
        core_plot: gr.update(value=r2_fig, visible=True),
        carrot_knn_plot: gr.update(value=carrot_fig, visible=True)
    }
```

---

## TODO: Click Selection Handler

**Still needs implementation:**

1. **Detect plot clicks** in Gradio (Plotly click events)
2. **Extract clicked data**: LLM name, token limit
3. **Run inference** with selected option
4. **Display results** in same format as Route Query mode

**Proposed flow:**
```python
@core_plot.select
def handle_core_click(evt: gr.SelectData):
    llm_name = evt.value["customdata"][0]
    token_limit = evt.value["customdata"][1]
    # Run inference and return results
    return run_single_inference(query_input.value, llm_name, token_limit)
```

---

## Comparison: Old vs New

### Old UI
- Budget input (unused, confusing)
- Tabs: "Visualizations" vs "Results" (separate)
- Two buttons at same location
- Plots stacked vertically

### New UI
- No budget (simpler)
- Two modes side-by-side (equal level)
- Each mode has its own button
- Plots horizontal (easier comparison)

### Workflows

**Old:**
1. Enter query + budget
2. Choose tab (Viz or Route)
3. Do action
4. See results in Results tab

**New - Route Query:**
1. Enter query
2. Set lambda
3. Click "Route Query"
4. See results immediately below

**New - Visualize:**
1. Enter query
2. Click "Show Visualizations"
3. See plots appear
4. **[TODO]** Click point to select
5. **[TODO]** See results immediately below

---

## Benefits

✅ **Clearer workflows**: Two distinct modes, not mixed
✅ **Simpler**: Removed unused budget parameter
✅ **Better layout**: Side-by-side comparison of modes
✅ **Space efficient**: Horizontal plots for easy comparison
✅ **More intuitive**: Visualization mode doesn't need lambda

---

## Files Modified

**`demo/app.py`:**
- Lines 66-90: Updated `generate_visualizations()` - removed budget
- Lines 107-132: Updated `process_query()` - removed budget
- Lines 390-393: Removed budget from results HTML
- Lines 551-645: Completely restructured UI layout
  - Two-column mode selection
  - Horizontal plots
  - Dynamic visibility

---

*UI now has two clear, equal-level modes: Route Query (with lambda) and Visualize Predictions (click to select).*
