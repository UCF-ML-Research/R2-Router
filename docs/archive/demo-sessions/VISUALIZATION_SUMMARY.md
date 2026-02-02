# Interactive Visualizations - Quick Summary

## What Was Added

Interactive Plotly visualizations showing cost-quality tradeoffs for R2-Router and CARROT.

## New Files

**`demo/visualizer.py`** - Complete visualization module
- `generate_r2_visualization()`: Curves for each LLM across token limits
- `generate_carrot_visualization()`: Points for each LLM (unlimited only)

## Modified Files

**`demo/app.py`**:
1. Line 26: Import visualizer functions
2. Lines 69-104: `generate_visualizations()` function
3. Lines 606-661: Added visualization tab and button

## Features

### R2-Router Visualization
- **Curves** showing cost-quality tradeoff for each LLM
- Each curve = one LLM across 16 token limits (10 to unlimited)
- X-axis: Predicted Cost
- Y-axis: Predicted Quality (0-1)
- **Budget line**: Vertical red dashed line showing constraint
- **Interactive**: Hover for details, zoom, pan, click legend

### CARROT Visualization
- **Points** showing one cost-quality pair per LLM
- Only unlimited setting (CARROT doesn't use token limits)
- Same axes as R2-Router
- Labels on each point
- **Interactive**: Same features as R2-Router

## UI Layout

```
Buttons:
[🚀 Route Query]  [📊 Visualize Predictions]

Tabs:
┌─ 📊 Visualizations ────────────────────┐
│  R2-Router: Cost-Quality Curves             │
│  [Large interactive plot]              │
│                                         │
│  CARROT-KNN          CARROT-Linear     │
│  [Plot]              [Plot]            │
└─────────────────────────────────────────┘

┌─ 🎯 Routing Results ────────────────────┐
│  [HTML results from routing]            │
└─────────────────────────────────────────┘
```

## Key Differences Visualized

| R2-Router | CARROT |
|------|--------|
| Curves (multiple token limits) | Points (unlimited only) |
| 16 options per LLM | 1 option per LLM |
| Flexible cost-quality tradeoff | Fixed unlimited setting |
| Can stay within budget constraints | Limited budget control |

## Example Interpretation

**R2-Router Curve:**
```
Quality ↑
    │    ╱────╲ LLM curve
    │   ╱      ╲
    │  ╱        ╲
    │ ╱          ╲
    └──────────────→ Cost

Lower token limits = left side (cheap, lower quality)
Higher token limits = right side (expensive, better quality)
```

**CARROT Points:**
```
Quality ↑
    │  ● ● ●  Different LLMs
    │ ●   ●
    │●
    └──────────────→ Cost

Each LLM = one point (unlimited only)
```

## Benefits

✅ **Visual comparison**: See R2-Router vs CARROT architectures
✅ **Budget constraints**: Vertical line shows feasible options
✅ **Query-specific**: Predictions update per query
✅ **Interactive**: Hover, zoom, explore predictions
✅ **Research insights**: Understand cost-quality tradeoffs

---

*Click "📊 Visualize Predictions" to see interactive cost-quality plots!*
