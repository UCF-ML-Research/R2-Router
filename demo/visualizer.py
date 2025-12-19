"""
Interactive Visualization for CoRE and CARROT
Generates plotly figures showing cost-quality tradeoffs
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Tuple
import config


def generate_core_visualization(
    embedding: np.ndarray,
    core_router,
    budget: float = None
) -> go.Figure:
    """
    Generate interactive CoRE visualization with curves for each LLM

    Args:
        embedding: Query embedding vector
        core_router: CoRERouter instance with loaded models
        budget: Optional budget to show as vertical line

    Returns:
        Plotly figure with cost-quality curves
    """
    # Ensure embedding is 2D
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    fig = go.Figure()

    # Color palette for different LLMs
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # For each LLM, generate a curve across token limits
    for idx, (llm_name, predictor) in enumerate(core_router.predictors.items()):
        llm_size = core_router.llm_pool[llm_name]["size"]

        # Get predictions
        quality_limited, quality_unlimited, token_count_unlimited = predictor.predict(embedding)

        # Clip to valid ranges
        quality_limited = np.clip(quality_limited, 0, 1)
        quality_unlimited = np.clip(quality_unlimited, 0, 1)
        token_count_unlimited = np.maximum(token_count_unlimited, 0)

        # Collect (cost, quality) points for this LLM
        costs = []
        qualities = []
        token_limits_list = []

        # Add limited token limits
        limited_token_limits = [t for t in core_router.token_limits if t != "unlimited"]
        for token_idx, token_limit in enumerate(limited_token_limits):
            predicted_score = quality_limited[0, token_idx]
            predicted_count = min(token_limit, token_count_unlimited[0])
            predicted_cost = predicted_count * llm_size

            costs.append(predicted_cost)
            qualities.append(predicted_score)
            token_limits_list.append(f"{token_limit}")

        # Add unlimited point
        predicted_score_unlimited = quality_unlimited[0]
        predicted_cost_unlimited = token_count_unlimited[0] * llm_size

        costs.append(predicted_cost_unlimited)
        qualities.append(predicted_score_unlimited)
        token_limits_list.append("unlimited")

        # Sort by cost for smooth curve
        sorted_indices = np.argsort(costs)
        costs = np.array(costs)[sorted_indices]
        qualities = np.array(qualities)[sorted_indices]
        token_limits_sorted = np.array(token_limits_list)[sorted_indices]

        # Create hover text
        hover_text = [
            f"<b>{llm_name}</b><br>" +
            f"Token Limit: {limit}<br>" +
            f"Predicted Quality: {q:.3f}<br>" +
            f"Predicted Cost: {c:.1f}"
            for limit, q, c in zip(token_limits_sorted, qualities, costs)
        ]

        # Add curve for this LLM
        fig.add_trace(go.Scatter(
            x=costs,
            y=qualities,
            mode='lines+markers',
            name=llm_name,
            line=dict(color=colors[idx % len(colors)], width=2),
            marker=dict(size=8),
            hovertext=hover_text,
            hoverinfo='text',
            customdata=[[llm_name, str(limit)] for limit in token_limits_sorted]
        ))

    # Add budget line if specified
    if budget is not None:
        fig.add_vline(
            x=budget,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Budget: {budget:.0f}",
            annotation_position="top"
        )

    # Update layout - enable click events
    fig.update_layout(
        title="CoRE: Cost-Quality Curves for Each LLM (Click any point to select)",
        xaxis_title="Predicted Cost (tokens × size)",
        yaxis_title="Predicted Quality Score",
        hovermode='closest',
        template='plotly_white',
        width=900,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        font=dict(size=12),
        clickmode='event+select'  # Enable click events
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[0, 1])

    return fig


def generate_carrot_visualization(
    embedding: np.ndarray,
    carrot_router
) -> go.Figure:
    """
    Generate interactive CARROT visualization with points for each LLM

    Args:
        embedding: Query embedding vector
        carrot_router: CARROTRouter instance with loaded model

    Returns:
        Plotly figure with cost-quality points
    """
    # Ensure embedding is 2D
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    fig = go.Figure()

    # Color palette
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # Get predictions for all LLMs (unlimited only)
    Y_hat_score, Y_hat_count = carrot_router.model.predict(embedding)

    # Clip predictions
    Y_hat_score = np.clip(Y_hat_score, 0, 1)
    Y_hat_count = np.maximum(Y_hat_count, 0)

    llm_names = list(carrot_router.llm_pool.keys())

    costs = []
    qualities = []
    hover_texts = []
    llm_name_list = []

    for idx, llm_name in enumerate(llm_names):
        llm_size = carrot_router.llm_pool[llm_name]["size"]

        predicted_score = Y_hat_score[0, idx]
        predicted_count = Y_hat_count[0, idx]
        predicted_cost = predicted_count * llm_size

        costs.append(predicted_cost)
        qualities.append(predicted_score)
        llm_name_list.append(llm_name)

        hover_text = (
            f"<b>{llm_name}</b><br>"
            f"Token Limit: unlimited<br>"
            f"Predicted Quality: {predicted_score:.3f}<br>"
            f"Predicted Cost: {predicted_cost:.1f}"
        )
        hover_texts.append(hover_text)

    # Add scatter plot with one trace per LLM for clickability
    for idx, (llm_name, cost, quality, hover_text) in enumerate(
        zip(llm_name_list, costs, qualities, hover_texts)
    ):
        fig.add_trace(go.Scatter(
            x=[cost],
            y=[quality],
            mode='markers+text',
            name=llm_name,
            marker=dict(
                size=15,
                color=colors[idx % len(colors)],
                line=dict(width=2, color='white')
            ),
            text=[llm_name],
            textposition="top center",
            textfont=dict(size=10),
            hovertext=[hover_text],
            hoverinfo='text',
            customdata=[[llm_name, "unlimited"]]
        ))

    # Update layout - enable click events
    fig.update_layout(
        title="CARROT-KNN: Single Quality-Cost Point per LLM (Click any point to select)",
        xaxis_title="Predicted Cost (tokens × size)",
        yaxis_title="Predicted Quality Score",
        hovermode='closest',
        template='plotly_white',
        width=900,
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        font=dict(size=12),
        clickmode='event+select'  # Enable click events
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[0, 1])

    return fig


def generate_combined_visualization(
    embedding: np.ndarray,
    core_router,
    carrot_router,
    budget: float = None
) -> Tuple[go.Figure, go.Figure]:
    """
    Generate both CoRE and CARROT visualizations

    Args:
        embedding: Query embedding vector
        core_router: CoRERouter instance
        carrot_router: CARROTRouter instance
        budget: Optional budget line

    Returns:
        Tuple of (core_figure, carrot_figure)
    """
    core_fig = generate_core_visualization(embedding, core_router, budget)
    carrot_fig = generate_carrot_visualization(embedding, carrot_router)

    return core_fig, carrot_fig
