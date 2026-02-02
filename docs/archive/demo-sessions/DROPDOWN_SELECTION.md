# Dropdown Selection for Visualization Mode

## Summary

Implemented dropdown selection for choosing LLM and token limit after viewing visualizations. This replaces the problematic plot click approach.

## Issue with Plot Clicks

**Problem**: Gradio's `Plot` component doesn't support `.select()` event handler
```python
# This doesn't work:
core_plot.select(fn=handler, ...)
# AttributeError: 'Plot' object has no attribute 'select'
```

**Solution**: Use dropdown selectors and a button instead

## Implementation

### 1. Selection UI

**File: `demo/app.py`** (lines 733-748)

**Added selection controls row:**
```python
with gr.Row(visible=False) as selection_row:
    with gr.Column():
        gr.Markdown("### Select Option from Visualization")
        gr.Markdown("After viewing the plots above, select an LLM and token limit to run inference.")

        llm_dropdown = gr.Dropdown(
            label="Select LLM",
            choices=list(config.LLM_POOL.keys()),
            value=None
        )

        token_dropdown = gr.Dropdown(
            label="Select Token Limit",
            choices=["10", "20", "30", "40", "50", "80", "100",
                     "150", "200", "300", "500", "800", "1200",
                     "2000", "4000", "unlimited"],
            value="unlimited"
        )

        run_selected_btn = gr.Button("▶️ Run Selected Option", variant="primary", size="lg")
```

**Key Features:**
- `visible=False` initially (appears after visualization)
- LLM dropdown with all available LLMs
- Token limit dropdown with all 16 options
- Run button to trigger inference

---

### 2. Show Visualizations with Selection

**File: `demo/app.py`** (lines 765-778)

**Updated function to show selection controls:**
```python
def show_visualizations(query):
    r2_fig, carrot_fig = generate_visualizations(query)
    return {
        core_plot: gr.update(value=r2_fig, visible=True),
        carrot_knn_plot: gr.update(value=carrot_fig, visible=True),
        selection_row: gr.update(visible=True)  # <-- Show selection UI
    }

visualize_btn.click(
    fn=show_visualizations,
    inputs=[query_input],
    outputs=[core_plot, carrot_knn_plot, selection_row]  # <-- 3 outputs
)
```

**Flow:**
1. User clicks "Show Visualizations"
2. Plots appear (R2-Router curves + CARROT points)
3. Selection dropdowns appear below plots
4. User can now view plots and make selection

---

### 3. Run Selected Option

**File: `demo/app.py`** (lines 780-785)

**Connect button to inference:**
```python
run_selected_btn.click(
    fn=run_single_inference,
    inputs=[query_input, llm_dropdown, token_dropdown],
    outputs=results_output
)
```

**Direct connection:**
- Takes query, selected LLM, selected token limit
- Calls `run_single_inference()` (already implemented)
- Displays results

---

## User Workflow

### Complete Flow

```
1. Enter query
   ↓
2. Click "📊 Show Visualizations"
   ↓
3. Plots appear:
   - R2-Router: Curves showing cost-quality across token limits
   - CARROT: Points showing unlimited options
   ↓
4. Selection controls appear below plots
   ↓
5. User views plots and decides on option
   ↓
6. Select from dropdowns:
   - LLM: Choose from available LLMs
   - Token Limit: Choose from 10 to unlimited
   ↓
7. Click "▶️ Run Selected Option"
   ↓
8. Inference runs → Results appear below
```

### Visual Layout

```
┌─────────────────────────────────────────────┐
│  🚀 Route Query  │  📊 Visualize Predictions│
│  ...             │  [Show Visualizations]    │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  R2-Router Plot [50%]    CARROT Plot [50%]       │
│  (Horizontal, appear after clicking button) │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Select Option from Visualization           │
│  LLM: [Dropdown ▼]                          │
│  Token Limit: [Dropdown ▼]                  │
│  [▶️ Run Selected Option]                   │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Results (HTML display)                     │
└─────────────────────────────────────────────┘
```

---

## Example Usage

**Step-by-step:**

1. **Query**: "What is the capital of France?"
2. **Click**: "Show Visualizations"
3. **Observe plots**:
   - R2-Router shows Qwen3-235B has high quality at 150 tokens
   - Cost is around 95.2
4. **Select**:
   - LLM: Qwen3-235B
   - Token Limit: 150
5. **Click**: "Run Selected Option"
6. **Results**:
   ```
   Method: R2-Router
   LLM: Qwen3-235B @ 150 tokens
   Predicted Score: 0.850, Cost: 95.2
   Actual Score: 0.900, Cost: 92.5
   [Full details...]
   ```

---

## Advantages Over Plot Clicks

### Why Dropdown Selection is Better

✅ **Works reliably** - No browser/version compatibility issues
✅ **More accessible** - Easier for users with disabilities
✅ **Clearer intent** - Explicit selection vs ambiguous clicking
✅ **Mobile friendly** - Dropdowns work well on touch screens
✅ **Error-proof** - Can't misclick or miss a point
✅ **Shows all options** - Don't need to hunt for points on plot

### Comparison

| Aspect | Plot Clicks | Dropdown Selection |
|--------|-------------|-------------------|
| **Compatibility** | Limited Gradio support | ✅ Full support |
| **Accessibility** | Difficult | ✅ Easy |
| **Mobile** | Small touch targets | ✅ Touch-friendly |
| **Clarity** | Ambiguous | ✅ Explicit |
| **Errors** | Can misclick | ✅ Clear options |
| **Discovery** | Need to find points | ✅ All visible |

---

## Technical Details

### Dynamic Visibility

- Selection row starts hidden: `visible=False`
- Becomes visible when plots are shown
- Uses `gr.update(visible=True)` to toggle

### Available Choices

**LLMs** (from `config.LLM_POOL`):
- All configured LLMs in the demo
- Typically: GLM-4.5-Air, gemma-3-4b, Llama-3.1-70B, Qwen3-235B, etc.

**Token Limits**:
- 15 limited: 10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000
- 1 unlimited

### Validation

- LLM dropdown starts with `value=None` (user must select)
- Token limit defaults to "unlimited" (safe default)
- `run_single_inference()` handles validation

---

## Benefits

### For Users
- **Easy to use**: Select from clear dropdowns
- **Visual guidance**: Plots help inform selection
- **No confusion**: Know exactly what you're selecting
- **Reliable**: Works every time

### For Development
- **No complex event handling**: Simple button click
- **Gradio compatible**: Uses standard components
- **Maintainable**: Clear, simple code
- **Testable**: Easy to verify selections

---

*View plots, select from dropdowns, run inference!*
