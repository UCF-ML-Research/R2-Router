# Separate Tabs for Direct and Visualized Routes

## Summary

Restructured the UI into two separate tabs with distinct workflows:
1. **Direct Route**: Algorithmic routing with lambda parameter
2. **Visualized Route**: Manual selection from cost-quality plots with separate controls for CoRE and CARROT-KNN

## Key Changes

### 1. Separate Tabs

**File: `demo/app.py`** (lines 694-771)

**Two distinct tabs:**
```python
with gr.Tabs():
    # Tab 1: Direct Route
    with gr.Tab("🚀 Direct Route"):
        # Lambda slider + method checkboxes + Route button
        # Results shown in this tab

    # Tab 2: Visualized Route
    with gr.Tab("📊 Visualized Route"):
        # Visualizations + separate selections + Run buttons
        # Results shown in this tab
```

**Benefits:**
- Clear separation of workflows
- No confusion between modes
- Each tab is self-contained
- Results stay in the same tab

---

### 2. Separate Selection for CoRE and CARROT

**Problem Solved:** Previously had one LLM dropdown and one token limit dropdown, which was confusing because:
- CoRE needs: LLM + Token Limit
- CARROT needs: LLM only (always unlimited)
- User might pick different LLMs for each method

**Solution:** Two independent selection columns

#### CoRE Selection (Left Column)

**Lines 742-756:**
```python
with gr.Column():
    gr.Markdown("### 🎯 CoRE Selection")
    gr.Markdown("Select LLM and token limit from the curves above")

    core_llm_dropdown = gr.Dropdown(
        label="CoRE: Choose LLM",
        choices=list(config.LLM_POOL.keys()),
        value=None
    )

    core_token_dropdown = gr.Dropdown(
        label="CoRE: Choose Token Limit",
        choices=["10", "20", ..., "unlimited"],
        value="unlimited"
    )

    run_core_btn = gr.Button("▶️ Run CoRE with Selection")
```

**Features:**
- Independent LLM selection for CoRE
- Full token limit selection (16 options)
- Dedicated run button
- Clear labeling

#### CARROT-KNN Selection (Right Column)

**Lines 759-769:**
```python
with gr.Column():
    gr.Markdown("### 🥕 CARROT-KNN Selection")
    gr.Markdown("Select LLM from the points above (always unlimited)")

    carrot_llm_dropdown = gr.Dropdown(
        label="CARROT: Choose LLM",
        choices=list(config.LLM_POOL.keys()),
        value=None
    )

    gr.Markdown("*Token limit: unlimited (fixed)*")

    run_carrot_btn = gr.Button("▶️ Run CARROT with Selection")
```

**Features:**
- Independent LLM selection for CARROT
- No token limit dropdown (always unlimited)
- Explicit note about fixed unlimited
- Dedicated run button

---

### 3. Separate Run Functions

**CoRE Run Button** (lines 801-805):
```python
run_core_btn.click(
    fn=run_single_inference,
    inputs=[query_input, core_llm_dropdown, core_token_dropdown],
    outputs=visualized_results
)
```

**CARROT Run Button** (lines 808-815):
```python
def run_carrot_selection(query, llm_name):
    return run_single_inference(query, llm_name, "unlimited")

run_carrot_btn.click(
    fn=run_carrot_selection,
    inputs=[query_input, carrot_llm_dropdown],
    outputs=visualized_results
)
```

**Key difference:** CARROT wrapper function always passes `"unlimited"` as token limit

---

## Complete UI Structure

```
┌─────────────────────────────────────────────────┐
│  Query Input              Example Queries       │
└─────────────────────────────────────────────────┘

┌─ 🚀 Direct Route ───────────────────────────────┐
│                                                  │
│  ### Algorithmic Routing with Lambda            │
│                                                  │
│  Lambda (λ): [Slider 0.0 to 1.0]                │
│                                                  │
│  [✓] CoRE    [✓] CARROT-KNN                     │
│                                                  │
│  [🚀 Route Query]                                │
│                                                  │
│  Direct Route Results:                          │
│  [HTML display area]                            │
└──────────────────────────────────────────────────┘

┌─ 📊 Visualized Route ───────────────────────────┐
│                                                  │
│  ### Visual Selection from Cost-Quality Plots   │
│                                                  │
│  [📊 Show Visualizations]                        │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │ CoRE Plot [50%]  │  CARROT Plot [50%]   │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌───────────────────┬────────────────────────┐ │
│  │ 🎯 CoRE Selection │ 🥕 CARROT-KNN Selection│ │
│  │ LLM: [Dropdown▼] │ LLM: [Dropdown▼]      │ │
│  │ Token: [Dropdown▼]│ Token: unlimited (fixed)│ │
│  │ [▶️ Run CoRE]    │ [▶️ Run CARROT]       │ │
│  └───────────────────┴────────────────────────┘ │
│                                                  │
│  Visualized Route Results:                      │
│  [HTML display area]                            │
└──────────────────────────────────────────────────┘
```

---

## Workflows

### Workflow 1: Direct Route

```
1. Enter query
2. Go to "Direct Route" tab
3. Set lambda value
4. Select methods (CoRE, CARROT, or both)
5. Click "Route Query"
6. System automatically selects best LLM and token limit
7. Results shown with comparison table
```

### Workflow 2: Visualized Route - CoRE

```
1. Enter query
2. Go to "Visualized Route" tab
3. Click "Show Visualizations"
4. View CoRE curves (left plot)
5. Hover to see details
6. Choose based on cost-quality tradeoff
7. Select LLM from CoRE dropdown
8. Select token limit from CoRE dropdown
9. Click "Run CoRE with Selection"
10. Results shown with CoRE metrics
```

### Workflow 3: Visualized Route - CARROT

```
1. Enter query
2. Go to "Visualized Route" tab
3. Click "Show Visualizations"
4. View CARROT points (right plot)
5. Hover to see details
6. Choose based on cost-quality point
7. Select LLM from CARROT dropdown
8. Click "Run CARROT with Selection"
9. Results shown with CARROT metrics
   (Token limit is automatically "unlimited")
```

---

## Advantages

### Separate Tabs

✅ **Clear workflows**: Two distinct modes, no confusion
✅ **Focused experience**: One mode at a time
✅ **Results stay in tab**: No jumping between views
✅ **Self-contained**: Each tab has everything needed

### Separate Selections

✅ **Independent choices**: Can pick different LLMs for CoRE vs CARROT
✅ **Clear labeling**: Explicit "CoRE" vs "CARROT" prefixes
✅ **Correct options**: Token limit only for CoRE
✅ **No ambiguity**: Each selection is clearly for one method

### User Experience

✅ **Intuitive**: Each tab explains what it does
✅ **No mistakes**: Can't accidentally mix up selections
✅ **Flexible**: Can try different LLMs for each method
✅ **Educational**: Understand difference between methods

---

## Comparison: Before vs After

### Before (Single Page, Shared Selection)

```
❌ Two modes side-by-side (cramped)
❌ One LLM dropdown (shared between CoRE and CARROT)
❌ One token limit dropdown (confusing for CARROT users)
❌ Unclear which selection is for which method
❌ Results mixed together
```

### After (Separate Tabs, Separate Selections)

```
✅ Two tabs (clear separation)
✅ CoRE has its own LLM dropdown
✅ CARROT has its own LLM dropdown
✅ Token limit only for CoRE (no confusion)
✅ Results shown in respective tabs
```

---

## Example Usage

### Direct Route Example

**User wants**: Best option with λ=0.3 (balanced)

1. Tab: "Direct Route"
2. Lambda: 0.3
3. Methods: Both CoRE and CARROT
4. Click "Route Query"
5. See comparison: CoRE picks Qwen3-235B @ 150, CARROT picks Llama3-70B @ unlimited

### Visualized Route Example

**User wants**: Try specific options they see in plots

1. Tab: "Visualized Route"
2. Click "Show Visualizations"
3. **For CoRE**: Sees Qwen3-235B @ 200 has quality 0.87, cost 110
   - Select: Qwen3-235B, Token: 200
   - Click "Run CoRE"
4. **For CARROT**: Sees gemma-3-4b has quality 0.75, cost 45 (cheap!)
   - Select: gemma-3-4b
   - Click "Run CARROT"
5. Compare results to decide

---

## Technical Implementation

### Tab Structure

Uses Gradio's `gr.Tabs()` and `gr.Tab()` components:
- Automatic tab switching
- Separate state for each tab
- Results isolated per tab

### Independent State

- `direct_results`: HTML output for Direct Route tab
- `visualized_results`: HTML output for Visualized Route tab
- No interference between tabs

### Wrapper Function

CARROT selection uses wrapper to fix token limit:
```python
def run_carrot_selection(query, llm_name):
    return run_single_inference(query, llm_name, "unlimited")
```

This ensures CARROT always uses unlimited, matching its architectural constraint.

---

*Two tabs, two workflows, clear separation!*
