# Interactive Cost-Quality Visualizations

## Summary

Added interactive Plotly visualizations showing cost-quality tradeoffs for CoRE and CARROT routing methods. Users can visualize how different LLMs and token limits perform on their query.

## New Files

### `demo/visualizer.py`
Complete visualization module with three main functions:

1. **`generate_core_visualization()`**: Creates cost-quality curves for CoRE
   - One curve per LLM showing quality across all token limits
   - X-axis: Predicted Cost (tokens × model size)
   - Y-axis: Predicted Quality Score (0-1)
   - Optional budget line (vertical red dashed line)
   - Interactive hover shows: LLM name, token limit, predicted quality, predicted cost
   - Click on points to see details

2. **`generate_carrot_visualization()`**: Creates cost-quality points for CARROT
   - One point per LLM (unlimited only)
   - X-axis: Predicted Cost
   - Y-axis: Predicted Quality Score
   - Points labeled with LLM names
   - Interactive hover shows same info as CoRE

3. **`generate_combined_visualization()`**: Helper to generate both

## Integration into Demo

### Modified Files

**`demo/app.py`** - Added visualization support:

1. **Import statement** (line 26):
   ```python
   from visualizer import generate_core_visualization, generate_carrot_visualization
   ```

2. **New function `generate_visualizations()`** (lines 69-104):
   - Takes query and budget as input
   - Generates embedding
   - Creates visualizations for CoRE and both CARROT variants
   - Returns tuple of (core_fig, carrot_knn_fig, carrot_linear_fig)

3. **Updated Gradio UI** (lines 606-661):
   - Added "📊 Visualize Predictions" button
   - Added tabs: "📊 Visualizations" and "🎯 Routing Results"
   - Visualization tab shows:
     - One large CoRE plot (full width)
     - Two CARROT plots side-by-side (KNN and Linear)
   - Wired button to call `generate_visualizations()`

## User Interface

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  Query Input                                            │
│  Lambda Slider | Budget Input                          │
│  [✓] CoRE  [✓] CARROT-KNN  [ ] CARROT-Linear          │
│  [🚀 Route Query]  [📊 Visualize Predictions]          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ [📊 Visualizations] [🎯 Routing Results]                │
│                                                          │
│  CoRE: Cost-Quality Curves                              │
│  ┌────────────────────────────────────────────┐         │
│  │    [Interactive Plotly chart]              │         │
│  │    Multiple curves, one per LLM            │         │
│  │    Budget line if specified                │         │
│  └────────────────────────────────────────────┘         │
│                                                          │
│  CARROT-KNN               CARROT-Linear                 │
│  ┌──────────────────┐    ┌──────────────────┐          │
│  │  [Scatter plot]  │    │  [Scatter plot]  │          │
│  │  Points for LLMs │    │  Points for LLMs │          │
│  └──────────────────┘    └──────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

### CoRE Visualization Features

**Curves for Each LLM:**
- Each LLM has a different colored curve
- Curve shows (cost, quality) across 16 token limits:
  - 15 limited: 10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000
  - 1 unlimited
- Points on curve = specific token limit settings

**Budget Line:**
- Vertical red dashed line at specified budget
- Shows constraint: only consider points left of this line
- Helps visualize which options are within budget

**Interactivity:**
- Hover over any point to see details
- Zoom in/out with mouse wheel
- Pan by dragging
- Legend shows/hides curves by clicking LLM names
- Click on curve points to see token limit and predictions

**Example:**
```
Quality
  1.0 ┤
      │     ╱──────╲ Qwen3-235B (expensive, high quality)
  0.8 ┤    ╱        ╲
      │   ╱          ╲
  0.6 ┤  ╱  ╱─────╲  ╲
      │ ╱  ╱       ╲  ╲
  0.4 ┤╱  ╱ GLM-4.5 ╲  ╲
      │  ╱           ╲  ╲
  0.2 ┤ ╱  gemma-3   ╲  ╲
      │╱              ╲  ╲
  0.0 └────────────────────────> Cost
      0   50  100  150 |200  250
                      Budget
```

### CARROT Visualization Features

**Points for Each LLM:**
- Each LLM = one point (unlimited only)
- Point size: 15px with white border
- Color-coded by LLM
- Text label above each point

**Interactivity:**
- Hover to see: LLM name, token limit (unlimited), predicted quality, predicted cost
- Zoom and pan enabled
- Click legend to show/hide specific LLMs

**Example:**
```
Quality
  1.0 ┤
      │      ● Qwen3-235B
  0.8 ┤   ● Llama3-70B
      │
  0.6 ┤ ● GLM-4.5
      │
  0.4 ┤● gemma-3
      │
  0.2 ┤
      │
  0.0 └────────────────────> Cost
      0   50  100  150  200
```

## Technical Details

### Data Flow

1. User enters query and optional budget
2. Click "📊 Visualize Predictions"
3. `generate_visualizations()` called:
   - Embeds query using embedder
   - For CoRE: Gets predictions from all predictors for all token limits
   - For CARROT: Gets predictions for unlimited setting only
   - Creates Plotly figures with interactive features
4. Figures displayed in Gradio Plot components

### Prediction Data

**CoRE predictions for each LLM:**
```python
quality_limited: (1, 15)  # 15 limited token limits
quality_unlimited: (1,)   # unlimited
token_count_unlimited: (1,)  # predicted tokens for unlimited
```

**CARROT predictions:**
```python
Y_hat_score: (1, N_llms)  # quality for each LLM (unlimited)
Y_hat_count: (1, N_llms)  # token count for each LLM
```

### Cost Calculation

```python
predicted_cost = predicted_tokens × llm_size

where:
- predicted_tokens: from predictor
- llm_size: model parameter count (from config.LLM_POOL)
```

### Plotly Configuration

**Common settings:**
- Template: `plotly_white` (clean background)
- Size: 900px width × 600px height
- Grid: Enabled with light gray lines
- Y-axis range: [0, 1] (quality is always 0-1)
- Hover mode: closest point

**CoRE-specific:**
- Mode: `lines+markers` (connected curves)
- Line width: 2px
- Marker size: 8px
- Sorted by cost for smooth curves

**CARROT-specific:**
- Mode: `markers+text` (scatter points with labels)
- Marker size: 15px with white border
- Text position: top center
- One trace per LLM for legend control

## Benefits

✅ **Visual Understanding**: See cost-quality tradeoffs at a glance
✅ **CoRE vs CARROT Comparison**: Curves vs points clearly show architectural difference
✅ **Budget Constraints**: Vertical line shows feasible region
✅ **Interactive Exploration**: Hover, zoom, pan to explore predictions
✅ **Per-Query Predictions**: Visualizes how predictions change for different queries
✅ **Token Limit Impact**: See how quality changes with token limits (CoRE only)
✅ **LLM Comparison**: Easily compare different LLMs' cost-quality profiles

## Usage Example

1. Enter query: "What is the capital of France?"
2. Set budget: 100 (optional)
3. Click "📊 Visualize Predictions"
4. Observe:
   - **CoRE**: Multiple curves showing tradeoffs
     - Cheap LLMs: Low cost, low quality
     - Expensive LLMs: High cost, high quality
     - Token limits: More tokens = higher cost, potentially better quality
   - **CARROT**: Single points per LLM
     - Only unlimited setting available
     - No token limit granularity
5. Budget line shows which options are affordable
6. Hover over points to see exact predictions

## Research Insights

This visualization helps understand:

1. **CoRE's advantage**: Can operate across the cost-quality spectrum via token limits
2. **CARROT's limitation**: Fixed unlimited setting per LLM, no flexibility
3. **LLM characteristics**: Some LLMs offer better quality/cost ratio
4. **Query dependence**: Predictions change based on query complexity
5. **Budget constraints**: How many options remain within budget

---

*Interactive visualizations make the cost-quality tradeoff tangible and help users understand routing decisions!*
