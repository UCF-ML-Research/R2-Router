# Removed CARROT-Linear from Demo

## Summary

Removed CARROT-Linear baseline from the demo, keeping only CoRE and CARROT-KNN for simplicity.

## Changes Made

### 1. Removed CARROT-Linear Router Initialization

**File: `demo/app.py`** (lines 49-53)

**Removed:**
```python
carrot_linear_router = get_carrot_router("linear")
carrot_linear_router.load_model()
```

**Result:** Only CARROT-KNN router is initialized

---

### 2. Updated Visualization Function

**File: `demo/app.py`** (lines 66-100)

**Changed:**
- Function signature: Returns `Tuple[core_figure, carrot_knn_figure]` (was 3-tuple)
- Docstring: Updated to reflect CARROT-KNN only
- Removed: `carrot_linear_fig = generate_carrot_visualization(embedding, carrot_linear_router)`
- Return: `return core_fig, carrot_knn_fig` (was 3-tuple)

---

### 3. Updated Process Query Function

**File: `demo/app.py`** (lines 107-155)

**Changed:**
- Removed parameter: `use_carrot_linear: bool`
- Updated validation: `if not (use_core or use_carrot_knn)` (removed `use_carrot_linear`)
- Removed: `if use_carrot_linear: methods.append(("CARROT-Linear", carrot_linear_router))`

---

### 4. Removed CARROT-Linear Checkbox from UI

**File: `demo/app.py`** (lines 584-593)

**Removed:**
```python
linear_check = gr.Checkbox(
    label="CARROT-Linear",
    value=False
)
```

**Result:** Only shows CoRE and CARROT-KNN checkboxes

---

### 5. Updated Visualization Tab

**File: `demo/app.py`** (lines 616-620)

**Changed:**
- Removed side-by-side column layout for CARROT plots
- Now shows single CARROT-KNN plot (full width)

**Before:**
```python
with gr.Row():
    with gr.Column():
        carrot_knn_plot = gr.Plot(label="CARROT-KNN: Cost-Quality Points")
    with gr.Column():
        carrot_linear_plot = gr.Plot(label="CARROT-Linear: Cost-Quality Points")
```

**After:**
```python
with gr.Row():
    carrot_knn_plot = gr.Plot(label="CARROT-KNN: Cost-Quality Points")
```

---

### 6. Updated Button Connections

**File: `demo/app.py`** (lines 626-645)

**Changed:**

**submit_btn inputs:**
- Removed `linear_check` from inputs list
- Now: `[query_input, lambda_input, budget_input, core_check, knn_check]`

**visualize_btn outputs:**
- Changed from 3-tuple to 2-tuple
- Now: `[core_plot, carrot_knn_plot]`

---

## UI Changes

### Before
```
Checkboxes:
[✓] CoRE (Our Method)
[✓] CARROT-KNN
[ ] CARROT-Linear

Visualization Tab:
┌─────────────────────────────────────────┐
│  CoRE: Cost-Quality Curves              │
│  [Full width plot]                      │
│                                          │
│  CARROT-KNN          CARROT-Linear      │
│  [Plot]              [Plot]             │
└─────────────────────────────────────────┘
```

### After
```
Checkboxes:
[✓] CoRE (Our Method)
[✓] CARROT-KNN

Visualization Tab:
┌─────────────────────────────────────────┐
│  CoRE: Cost-Quality Curves              │
│  [Full width plot]                      │
│                                          │
│  CARROT-KNN                             │
│  [Full width plot]                      │
└─────────────────────────────────────────┘
```

---

## Benefits

✅ **Simpler UI**: Fewer options, less confusion
✅ **Cleaner comparison**: CoRE vs CARROT-KNN only
✅ **Faster loading**: One less model to initialize
✅ **Better layout**: CARROT-KNN plot now full width

---

## What's Still Available

- ✅ CoRE routing (with token limit curves)
- ✅ CARROT-KNN routing (with single points)
- ✅ Interactive visualizations for both
- ✅ All metrics and comparisons
- ✅ Input prompt display
- ✅ Predicted risk winner selection

---

## Testing

To verify the changes work correctly:

1. **Startup**: Demo should load without errors (no CARROT-Linear initialization)
2. **UI**: Only CoRE and CARROT-KNN checkboxes visible
3. **Routing**: Can route with CoRE only, CARROT-KNN only, or both
4. **Visualization**: Shows 2 plots (CoRE curves + CARROT-KNN points)
5. **Results**: Comparison table and detailed cards work for both methods

---

*CARROT-Linear removed. Demo now focuses on CoRE vs CARROT-KNN comparison.*
