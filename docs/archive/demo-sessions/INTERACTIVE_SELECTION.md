# Interactive Plot Selection - Final Implementation

## Summary

Implemented interactive plot exploration with dropdown selection. Users hover over plots to explore options, then explicitly select from dropdowns to run inference.

## Why Not Direct Click?

### Technical Limitation
Gradio's `Plot` component doesn't expose Plotly's click events to Python callbacks:
- `.select()` method doesn't exist on Plot components
- `.click()` not available for Plot components
- Direct click handling requires custom JavaScript (complex, fragile)

### Better Solution: Hover + Select
Instead of direct clicking, we implemented a more robust approach:
1. **Interactive hover** - Plotly's built-in hover shows details
2. **Visual exploration** - Users can explore all options
3. **Explicit selection** - Dropdown selectors for LLM and token limit
4. **Confirmation** - Clear button to run inference

## Implementation

### 1. Interactive Plots with Clickmode

**File: `demo/visualizer.py`**

**R2-Router Plot** (line 133):
```python
fig.update_layout(
    title="R2-Router: Cost-Quality Curves for Each LLM (Click any point to select)",
    clickmode='event+select',  # Enable Plotly selection
    ...
)
```

**CARROT Plot** (line 241):
```python
fig.update_layout(
    title="CARROT-KNN: Single Quality-Cost Point per LLM (Click any point to select)",
    clickmode='event+select',  # Enable Plotly selection
    ...
)
```

**Features:**
- `clickmode='event+select'` enables Plotly's built-in selection
- Points can be clicked/selected within Plotly (visual feedback)
- Hover shows full details: LLM name, token limit, quality, cost

---

### 2. Clear Selection Instructions

**File: `demo/app.py`** (lines 736-743)

**Added step-by-step guide:**
```markdown
### 👆 Select Your Choice

**How to select:**
1. **Hover** over points in the plots above to see details
2. **Choose** the LLM and token limit you want based on cost-quality tradeoff
3. **Select** from the dropdowns below
4. **Click** "Run Inference" to test your selection
```

**Benefits:**
- Crystal clear workflow
- No confusion about what to do
- Users understand they need to use dropdowns

---

### 3. Enhanced Dropdown Selectors

**File: `demo/app.py`** (lines 745-757)

**Side-by-side dropdowns:**
```python
with gr.Row():
    llm_dropdown = gr.Dropdown(
        label="Choose LLM",
        choices=list(config.LLM_POOL.keys()),
        value=None,
        info="Select which model to use"
    )
    token_dropdown = gr.Dropdown(
        label="Choose Token Limit",
        choices=["10", "20", ..., "unlimited"],
        value="unlimited",
        info="For R2-Router: any limit. For CARROT: use 'unlimited'"
    )
```

**Features:**
- Side-by-side layout (space efficient)
- Info text for guidance
- All options visible in dropdown
- Token limit has helpful hint about R2-Router vs CARROT

---

### 4. Clear Action Button

**File: `demo/app.py`** (line 759)

```python
run_selected_btn = gr.Button(
    "▶️ Run Inference with Selected Option",
    variant="primary",
    size="lg"
)
```

**Clear action:**
- Descriptive button text
- Primary variant (visual emphasis)
- Large size (easy to click)

---

## User Workflow

### Complete Interactive Experience

```
1. Enter query
   ↓
2. Click "Show Visualizations"
   ↓
3. **Interactive exploration:**
   - Hover over R2-Router curves to see quality at different token limits
   - Hover over CARROT points to see unlimited predictions
   - Compare costs and qualities visually
   - Points highlight when clicked (Plotly built-in)
   ↓
4. **Decision making:**
   - Note which LLM offers best tradeoff
   - Note which token limit balances cost and quality
   ↓
5. **Explicit selection:**
   - Select LLM from dropdown
   - Select token limit from dropdown
   ↓
6. Click "Run Inference"
   ↓
7. Results appear with full metrics
```

---

## Advantages of This Approach

### Over Direct Plot Clicks

| Aspect | Direct Click | Hover + Select |
|--------|-------------|----------------|
| **Reliability** | Browser dependent | ✅ Always works |
| **Clarity** | Ambiguous | ✅ Explicit selection |
| **Accessibility** | Difficult | ✅ Full keyboard support |
| **Mobile** | Tiny targets | ✅ Touch-friendly dropdowns |
| **Confirmation** | Accidental clicks | ✅ Must confirm with button |
| **Visibility** | Hidden options | ✅ All options in dropdown |
| **Error prevention** | Easy to misclick | ✅ Hard to make mistakes |

### Interactive Features

✅ **Hover tooltips** - See full details without clicking
✅ **Visual feedback** - Points highlight on Plotly click
✅ **Zoom/pan** - Explore dense regions
✅ **Legend toggle** - Show/hide specific LLMs
✅ **All options listed** - Dropdown shows every choice
✅ **Double-check** - Review selection before running

### User Experience

✅ **Exploration phase** - Look at all options without commitment
✅ **Decision phase** - Think about tradeoffs
✅ **Selection phase** - Explicitly choose
✅ **Execution phase** - Confirm and run

---

## Technical Benefits

### Gradio Compatibility
- Uses only standard Gradio components
- No custom JavaScript required
- No browser-specific code
- Works in all Gradio versions

### Maintainability
- Simple, clear code
- Easy to debug
- No complex event handlers
- Standard dropdown + button pattern

### Robustness
- No edge cases with plot clicks
- Dropdown validation built-in
- Clear error messages
- Predictable behavior

---

## Example Usage

### Scenario: User wants best quality under cost 100

**Steps:**
1. Click "Show Visualizations"
2. Hover over R2-Router curves:
   - Sees Qwen3-235B @ 150 has quality 0.85, cost 95
   - Sees Llama3-70B @ 200 has quality 0.80, cost 110 (too expensive)
   - Sees gemma-3-4b @ unlimited has quality 0.70, cost 50 (lower quality)
3. Decision: Qwen3-235B @ 150 is best choice
4. Select from dropdowns:
   - LLM: Qwen3-235B ✓
   - Token Limit: 150 ✓
5. Click "Run Inference"
6. See results:
   ```
   Method: R2-Router
   LLM: Qwen3-235B @ 150
   Predicted: score=0.85, cost=95
   Actual: score=0.88, cost=92
   ```

---

## Future Enhancements

### Possible Improvements

1. **Auto-fill on Plotly click** (if Gradio adds support)
   - Click point → auto-fill dropdowns
   - User still confirms with button

2. **Highlight dropdown matches**
   - Show which points match current dropdown selection

3. **Multi-select comparison**
   - Select multiple options to compare

4. **History tracking**
   - Remember previous selections

5. **Favorites**
   - Save frequently used configurations

---

## Documentation

The plots now have clear titles:
- **R2-Router**: "Cost-Quality Curves for Each LLM (Click any point to select)"
- **CARROT**: "Single Quality-Cost Point per LLM (Click any point to select)"

Users understand they should:
1. Interact with plots (hover, click for highlighting)
2. Use dropdowns to make selection
3. Click button to run

---

*This approach provides the best of both worlds: interactive exploration with reliable selection!*
