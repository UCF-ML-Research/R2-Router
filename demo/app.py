"""
CoRE Router Demo - Gradio Web Interface
Interactive demo for testing CoRE routing vs CARROT baselines
"""

import sys
from pathlib import Path

# Ensure demo directory is first in path to use local modules
demo_dir = Path(__file__).parent.absolute()
if str(demo_dir) not in sys.path:
    sys.path.insert(0, str(demo_dir))

import gradio as gr
import numpy as np
from typing import List, Dict
import time
import plotly.graph_objects as go

# Import demo modules (now guaranteed to be from demo directory)
import config
from embedder import get_embedder
from router import get_core_router
from baselines import get_carrot_router
from llm_client import get_llm_client
from judge import get_judge
from visualizer import generate_core_visualization, generate_carrot_visualization


# ============================================================================
# Global State (loaded once at startup)
# ============================================================================

print("="*60)
print("CoRE Router Demo - Initializing...")
print("="*60)
print()

# Validate configuration
config.validate_config()
print()

# Initialize components
embedder = get_embedder()
embedder.load_model()

core_router = get_core_router()
core_router.load_models()

carrot_knn_router = get_carrot_router("knn")
carrot_knn_router.load_model()

llm_client = get_llm_client()
judge = get_judge()

print()
print("="*60)
print("✅ Initialization complete!")
print("="*60)
print()


# ============================================================================
# Visualization Function
# ============================================================================

def generate_result_analysis(core_result, carrot_result):
    """
    Generate Result Analysis section with Plotly grouped bar chart comparing metrics
    Each metric uses its own local Y-scale for better visual comparison

    Args:
        core_result: Dictionary with CoRE results (or None)
        carrot_result: Dictionary with CARROT results (or None)

    Returns:
        Plotly figure object (for gr.Plot) or None
    """
    if not core_result or not carrot_result:
        return None

    # Extract metrics
    metrics_data = [
        {
            'name': 'Tokens',
            'core': core_result.get('actual_tokens', 0),
            'carrot': carrot_result.get('actual_tokens', 0),
            'lower_better': True,
            'use_log': False
        },
        {
            'name': 'Cost',
            'core': core_result.get('actual_cost', 0),
            'carrot': carrot_result.get('actual_cost', 0),
            'lower_better': True,
            'use_log': False
        },
        {
            'name': 'Quality',
            'core': core_result.get('actual_score', 0),
            'carrot': carrot_result.get('actual_score', 0),
            'lower_better': False,
            'use_log': False
        },
        {
            'name': 'Score',
            'core': core_result.get('actual_risk', 0),
            'carrot': carrot_result.get('actual_risk', 0),
            'lower_better': False,
            'use_log': True  # Use log scale because values can be negative
        }
    ]

    # Create Plotly figure with subplots for each metric
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.08
    )

    # Add bars for each metric with independent Y-scale
    for idx, metric in enumerate(metrics_data, start=1):
        core_val = metric['core']
        carrot_val = metric['carrot']

        # Determine winner (check for ties)
        if metric['lower_better']:
            core_better = core_val < carrot_val
            is_tie = core_val == carrot_val
        else:
            core_better = core_val > carrot_val
            is_tie = core_val == carrot_val

        # Winner indicators
        core_text = f"🏆 {core_val:.3f}" if core_better else f"{core_val:.3f}"
        carrot_text = f"🏆 {carrot_val:.3f}" if not core_better and not is_tie else f"{carrot_val:.3f}"

        # Format display text based on metric
        if metric['name'] == 'Tokens':
            core_text = f"🏆 {int(core_val)}" if core_better else f"{int(core_val)}"
            carrot_text = f"🏆 {int(carrot_val)}" if not core_better and not is_tie else f"{int(carrot_val)}"
        elif metric['name'] == 'Cost':
            core_text = f"🏆 {core_val:.2f}" if core_better else f"{core_val:.2f}"
            carrot_text = f"🏆 {carrot_val:.2f}" if not core_better and not is_tie else f"{carrot_val:.2f}"

        # Fixed colors: CoRE = blue, CARROT = orange
        core_color = '#667eea'   # Blue for CoRE
        carrot_color = '#f39c12'  # Orange for CARROT

        # Add CoRE bar
        fig.add_trace(
            go.Bar(
                name='🎯 CoRE',
                x=['CoRE'],
                y=[core_val],
                text=[core_text],
                textposition='outside',
                marker_color=core_color,
                showlegend=(idx == 1),  # Only show legend for first subplot
                hovertemplate=f'CoRE: {core_text}<extra></extra>'
            ),
            row=1, col=idx
        )

        # Add CARROT bar
        fig.add_trace(
            go.Bar(
                name='🥕 CARROT-KNN',
                x=['CARROT'],
                y=[carrot_val],
                text=[carrot_text],
                textposition='outside',
                marker_color=carrot_color,
                showlegend=(idx == 1),  # Only show legend for first subplot
                hovertemplate=f'CARROT-KNN: {carrot_text}<extra></extra>'
            ),
            row=1, col=idx
        )

        # Update Y-axis for this subplot (use default auto-scaling for all metrics)
        fig.update_yaxes(
            title_text="",
            showticklabels=True,
            row=1, col=idx
        )

        # Update X-axis for this subplot
        fig.update_xaxes(
            showticklabels=True,
            row=1, col=idx
        )

    # Update overall layout
    fig.update_layout(
        title_text="📈 Result Analysis: CoRE vs CARROT-KNN",
        title_font=dict(size=24, color='#9b59b6', family='Arial Black'),
        title_x=0.5,
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=14)
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        paper_bgcolor='white',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        margin=dict(t=120, b=100, l=50, r=50),
        font=dict(size=12)
    )

    # Add subplot titles below each chart using annotations (centered)
    subplot_positions = [0.115, 0.365, 0.615, 0.865]  # X positions for each subplot
    for pos, metric in zip(subplot_positions, metrics_data):
        fig.add_annotation(
            text=f"<b>{metric['name']}</b><br>({'↓' if metric['lower_better'] else '↑'} better)",
            xref="paper", yref="paper",
            x=pos, y=-0.12,
            showarrow=False,
            font=dict(size=14, color='#2c3e50'),
            xanchor='center',
            yanchor='top',
            align='center'
        )

    # Return the figure object directly for gr.Plot
    return fig


def generate_visualizations(
    query: str
):
    """
    Generate interactive visualizations for CoRE and CARROT-KNN

    Args:
        query: User query text

    Returns:
        Tuple of (core_figure, carrot_knn_figure)
    """
    if not query.strip():
        return None, None

    # Generate embedding
    embedding = embedder.embed(query)

    # Generate CoRE visualization (no budget line)
    core_fig = generate_core_visualization(embedding, core_router, budget=None)

    # Generate CARROT-KNN visualization
    carrot_knn_fig = generate_carrot_visualization(embedding, carrot_knn_router)

    return core_fig, carrot_knn_fig


# ============================================================================
# Single Selection Inference (from visualization)
# ============================================================================

def run_single_inference_internal(
    query: str,
    llm_name: str,
    token_limit: str,
    embedding: np.ndarray = None,
    method_name: str = "CoRE"
) -> dict:
    """
    Internal function: Run inference and return results as dictionary

    Args:
        query: User query text
        llm_name: Selected LLM name
        embedding: Pre-computed embedding (optional, will compute if None)
        method_name: "CoRE" or "CARROT-KNN"

    Returns:
        Dictionary with all results
    """
    # Get embedding if not provided
    if embedding is None:
        embedding = embedder.embed(query)

    # Initialize routing result
    routing_result = {
        "llm_name": llm_name,
        "token_limit": token_limit,
        "method": method_name
    }

    # Recalculate predictions for this specific option
    if method_name == "CoRE":
        predictor = core_router.predictors[llm_name]
        llm_size = core_router.llm_pool[llm_name]["size"]

        quality_limited, quality_unlimited, token_count_unlimited = predictor.predict(embedding)
        quality_limited = np.clip(quality_limited, 0, 1)
        quality_unlimited = np.clip(quality_unlimited, 0, 1)
        token_count_unlimited = np.maximum(token_count_unlimited, 0)

        if token_limit == "unlimited":
            predicted_score = quality_unlimited[0]
            predicted_count = token_count_unlimited[0]
        else:
            token_limit_int = int(token_limit)
            limited_token_limits = [t for t in core_router.token_limits if t != "unlimited"]
            token_idx = limited_token_limits.index(token_limit_int)
            predicted_score = quality_limited[0, token_idx]
            predicted_count = min(token_limit_int, token_count_unlimited[0])

        predicted_cost = predicted_count * llm_size
    else:
        # CARROT-KNN
        Y_hat_score, Y_hat_count = carrot_knn_router.model.predict(embedding)
        Y_hat_score = np.clip(Y_hat_score, 0, 1)
        Y_hat_count = np.maximum(Y_hat_count, 0)

        # DEBUG: Print CARROT prediction details
        print(f"\n{'='*60}")
        print(f"DEBUG: CARROT Extraction (get_llm_response_with_selection)")
        print(f"{'='*60}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Y_hat_score shape: {Y_hat_score.shape}")
        print(f"Y_hat_count shape: {Y_hat_count.shape}")
        print(f"Y_hat_score: {Y_hat_score[0]}")
        print(f"Y_hat_count: {Y_hat_count[0]}")

        llm_names = list(carrot_knn_router.llm_pool.keys())
        print(f"LLM pool: {llm_names} (count={len(llm_names)})")
        print(f"Requested LLM: {llm_name}")

        llm_idx = llm_names.index(llm_name)
        print(f"LLM index: {llm_idx}")

        llm_size = carrot_knn_router.llm_pool[llm_name]["size"]

        predicted_score = Y_hat_score[0, llm_idx]
        predicted_count = Y_hat_count[0, llm_idx]
        predicted_cost = predicted_count * llm_size

        print(f"Extracted: score={predicted_score:.3f}, count={predicted_count:.1f}, cost={predicted_cost:.1f}")
        print(f"{'='*60}\n")

    # Call LLM
    print(f"\n🔄 Selected: {llm_name} @ {token_limit}")
    print(f"   → Predicted: score={predicted_score:.3f}, cost={predicted_cost:.1f}")

    response, actual_tokens, actual_prompt = llm_client.call_llm_by_name(
        llm_name=llm_name,
        query=query,
        token_limit=token_limit
    )

    # Judge response
    actual_score, judge_feedback = judge.evaluate(query, response)

    # Calculate actual cost
    llm_size = core_router.llm_pool[llm_name]["size"]
    actual_cost = actual_tokens * llm_size

    # Calculate metrics
    score_error = abs(predicted_score - actual_score)
    cost_effectiveness = actual_score / actual_cost if actual_cost > 0 else 0

    return {
        "llm_name": llm_name,
        "token_limit": token_limit,
        "method": method_name,
        "predicted_score": float(predicted_score),
        "predicted_cost": float(predicted_cost),
        "actual_score": float(actual_score),
        "actual_cost": float(actual_cost),
        "score_error": float(score_error),
        "cost_effectiveness": float(cost_effectiveness),
        "response": response,
        "judge_feedback": judge_feedback,
        "actual_tokens": actual_tokens
    }


def run_single_inference(
    query: str,
    llm_name: str,
    token_limit: str
) -> str:
    """
    Run inference for a single selected (LLM, token_limit) option
    (Wrapper that converts internal dict to HTML)

    Args:
        query: User query text
        llm_name: Selected LLM name
        token_limit: Selected token limit (string)

    Returns:
        HTML string with results
    """
    if not query.strip():
        return "<p style='color: red;'>Please enter a query!</p>"

    try:
        start_time = time.time()

        # Determine which method this LLM belongs to
        if llm_name in core_router.predictors:
            method_name = "CoRE"
        else:
            method_name = "CARROT-KNN"

        # Call internal function
        result_dict = run_single_inference_internal(query, llm_name, token_limit, None, method_name)

        elapsed_time = time.time() - start_time

        # Convert to format expected by generate_results_html
        result = {
            "method": result_dict["method"],
            "llm_name": result_dict["llm_name"],
            "token_limit": result_dict["token_limit"],
            "response": result_dict["response"],
            "actual_prompt": "",  # Not tracked in internal version
            "original_query": query,
            "predicted_score": result_dict["predicted_score"],
            "predicted_cost": result_dict["predicted_cost"],
            "predicted_risk": result_dict["predicted_score"],  # lambda=0
            "actual_score": result_dict["actual_score"],
            "actual_cost": result_dict["actual_cost"],
            "actual_risk": result_dict["actual_score"],  # lambda=0
            "actual_tokens": result_dict["actual_tokens"],
            "reasoning": result_dict["judge_feedback"]
        }

        # Generate HTML (single result, no comparison)
        html = generate_results_html(query, 0.0, None, [result], elapsed_time)
        return html

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return f"<p style='color: red;'>Error: {e}</p><pre>{error_msg}</pre>"


# ============================================================================
# Main Processing Function
# ============================================================================

def process_query(
    query: str,
    lambda_val: float,
    use_core: bool,
    use_carrot_knn: bool,
    progress=gr.Progress()
):
    """
    Process a query through selected routing methods with step-by-step progress

    Args:
        query: User query text
        lambda_val: Lambda value (0-1)
        use_core: Whether to use CoRE
        use_carrot_knn: Whether to use CARROT-KNN
        progress: Gradio progress tracker

    Returns:
        Tuple of (HTML string with results, Plotly figure for result analysis)
    """
    if not query.strip():
        return "<p style='color: red;'>Please enter a query!</p>", None

    if not (use_core or use_carrot_knn):
        return "<p style='color: red;'>Please select at least one routing method!</p>", None

    # No budget constraint
    budget = None

    # Determine which methods to use
    methods = []
    if use_core:
        methods.append(("CoRE", core_router))
    if use_carrot_knn:
        methods.append(("CARROT-KNN", carrot_knn_router))

    methods_text = " and ".join([m[0] for m in methods])

    # Step 1: Generate embedding
    progress(0.1, desc="Generating query embedding...")
    print(f"\n📝 Query: {query}")
    print(f"   Lambda: {lambda_val}, Budget: {budget}")

    embedding = embedder.embed(query)

    # Step 2: Routing decisions
    progress(0.3, desc=f"Computing routing decisions for {methods_text}...")
    results = []
    routing_decisions = []

    # Compute all routing decisions first
    for method_name, router in methods:
        print(f"\n🔄 Routing with {method_name}...")
        routing_result = router.route(embedding, lambda_val=lambda_val, budget=budget)
        routing_decisions.append((method_name, router, routing_result))

    # Show routing decisions in card-style format (side-by-side)
    # Separate CoRE and CARROT results
    core_result = None
    carrot_result = None

    for method_name, _, routing_result in routing_decisions:
        if method_name == "CoRE":
            core_result = routing_result
        elif "CARROT" in method_name:
            carrot_result = routing_result

    # Create side-by-side layout
    routing_info_html = '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 15px 0;">'

    # CoRE column
    if core_result:
        routing_info_html += f"""
        <div style="background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 20px; font-weight: bold; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #667eea;">
                🎯 CoRE
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin: 15px 0;">
                <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #667eea;">
                    <div style="font-size: 12px; color: #666; text-transform: uppercase;">Selected LLM</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{core_result['llm_name']}</div>
                </div>
                <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #667eea;">
                    <div style="font-size: 12px; color: #666; text-transform: uppercase;">Token Limit</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{core_result['token_limit']}</div>
                </div>
                <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #667eea;">
                    <div style="font-size: 12px; color: #666; text-transform: uppercase;">Predicted Score</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{core_result['predicted_score']:.3f}</div>
                </div>
                <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #667eea;">
                    <div style="font-size: 12px; color: #666; text-transform: uppercase;">Predicted Cost</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{core_result['predicted_cost']:.1f}</div>
                </div>
            </div>
        </div>
        """

    # CARROT column
    if carrot_result:
        routing_info_html += f"""
        <div style="background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 20px; font-weight: bold; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #f39c12;">
                🥕 CARROT-KNN
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin: 15px 0;">
                <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #f39c12;">
                    <div style="font-size: 12px; color: #666; text-transform: uppercase;">Selected LLM</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{carrot_result['llm_name']}</div>
                </div>
                <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #f39c12;">
                    <div style="font-size: 12px; color: #666; text-transform: uppercase;">Token Limit</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{carrot_result['token_limit']}</div>
                </div>
                <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #f39c12;">
                    <div style="font-size: 12px; color: #666; text-transform: uppercase;">Predicted Score</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{carrot_result['predicted_score']:.3f}</div>
                </div>
                <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #f39c12;">
                    <div style="font-size: 12px; color: #666; text-transform: uppercase;">Predicted Cost</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{carrot_result['predicted_cost']:.1f}</div>
                </div>
            </div>
        </div>
        """

    routing_info_html += '</div>'

    # Step 3: Call LLMs and evaluate
    num_methods = len(routing_decisions)
    for idx, (method_name, router, routing_result) in enumerate(routing_decisions):
        llm_name = routing_result["llm_name"]
        token_limit = routing_result["token_limit"]

        # Update progress for LLM calling
        base_progress = 0.4
        step_size = 0.3 / num_methods
        progress(base_progress + idx * step_size, desc=f"Querying {method_name} LLM ({llm_name} @ {token_limit})...")

        try:
            print(f"   → Selected: {llm_name} @ {token_limit} tokens")
            print(f"   → Predicted: score={routing_result['predicted_score']:.3f}, "
                  f"cost={routing_result['predicted_cost']:.1f}")

            # Call LLM
            print(f"   Calling {llm_name}...")
            response, actual_tokens, actual_prompt = llm_client.call_llm_by_name(
                llm_name=llm_name,
                query=query,
                token_limit=token_limit
            )

            # Calculate actual cost
            llm_size = config.LLM_POOL[llm_name]["size"]
            actual_cost = actual_tokens * llm_size

            # Update progress for evaluation
            progress(base_progress + (idx + 0.5) * step_size, desc=f"Evaluating {method_name} response with judge model...")

            # Evaluate with judge
            print(f"   Evaluating response...")
            actual_score, reasoning = judge.evaluate(query, response)

            # Calculate actual risk value using actual score and cost
            actual_risk = (1 - lambda_val) * actual_score - lambda_val * actual_cost

            print(f"   ✓ Actual: score={actual_score:.3f}, cost={actual_cost:.1f}, risk={actual_risk:.3f}")

            # Store result
            results.append({
                "method": method_name,
                "llm_name": llm_name,
                "token_limit": token_limit,
                "response": response,
                "actual_prompt": actual_prompt,
                "original_query": query,
                "predicted_score": routing_result["predicted_score"],
                "predicted_cost": routing_result["predicted_cost"],
                "predicted_risk": routing_result["risk"],
                "actual_score": actual_score,
                "actual_cost": actual_cost,
                "actual_risk": actual_risk,
                "actual_tokens": actual_tokens,
                "reasoning": reasoning
            })

        except Exception as e:
            print(f"   ✗ Error in {method_name}: {e}")
            results.append({
                "method": method_name,
                "error": str(e)
            })

    # Build LLM responses section in card-style format (side-by-side)
    # Separate CoRE and CARROT results
    core_response_result = None
    carrot_response_result = None

    for result in results:
        if "error" in result:
            continue
        method_name = result["method"]
        if method_name == "CoRE":
            core_response_result = result
        elif "CARROT" in method_name:
            carrot_response_result = result

    # Create side-by-side layout
    llm_responses_html = '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 15px 0;">'

    # CoRE column
    if core_response_result:
        # Format the CoRE input with highlighted instructional prompt
        core_input_html = format_prompt_with_highlight(
            core_response_result['original_query'],
            core_response_result.get('actual_prompt', core_response_result['original_query']),
            str(core_response_result['token_limit'])
        )

        llm_responses_html += f"""
        <div style="background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 20px; font-weight: bold; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #667eea;">
                🎯 CoRE
            </div>
            <div style="margin-bottom: 15px;">
                <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #667eea; height: 120px; display: flex; flex-direction: column;">
                    <div style="font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 8px;">Input</div>
                    <div style="font-size: 14px; color: #333; font-family: monospace; white-space: pre-wrap; overflow-y: auto; flex: 1;">{core_input_html}</div>
                </div>
            </div>
            <div style="margin-bottom: 15px;">
                <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #667eea; height: 120px; display: flex; flex-direction: column;">
                    <div style="font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 8px;">
                        <span style="text-decoration: underline; text-decoration-color: red; text-decoration-thickness: 2px;">Output</span>
                    </div>
                    <div style="font-size: 14px; color: #333; font-family: monospace; white-space: pre-wrap; overflow-y: auto; flex: 1;">{core_response_result['response']}</div>
                </div>
            </div>
            <div style="margin-bottom: 15px;">
                <div style="background: #e8f5e9; padding: 12px; border-radius: 5px; border-left: 4px solid #4caf50; height: 100px; display: flex; flex-direction: column;">
                    <div style="font-size: 12px; color: #2e7d32; font-weight: bold; margin-bottom: 8px;">Judge Feedback</div>
                    <div style="font-size: 14px; color: #333; overflow-y: auto; flex: 1;">{core_response_result['reasoning']}</div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #667eea;">
                    <div style="font-size: 11px; color: #666; text-transform: uppercase;">Actual Quality</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 3px;">{core_response_result['actual_score']:.3f}</div>
                </div>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #667eea;">
                    <div style="font-size: 11px; color: #666; text-transform: uppercase;">Actual Tokens</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 3px;">{core_response_result['actual_tokens']}</div>
                </div>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #667eea;">
                    <div style="font-size: 11px; color: #666; text-transform: uppercase;">Actual Cost</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 3px;">{core_response_result['actual_cost']:.2f}</div>
                </div>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #667eea;">
                    <div style="font-size: 11px; color: #666; text-transform: uppercase;">Actual Score</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 3px;">{core_response_result['actual_risk']:.3f}</div>
                    <div style="font-size: 10px; color: #666; margin-top: 5px;">= (1-{lambda_val}) × {core_response_result['actual_score']:.3f} - {lambda_val} × {core_response_result['actual_cost']:.3f}</div>
                </div>
            </div>
        </div>
        """

    # CARROT column
    if carrot_response_result:
        llm_responses_html += f"""
        <div style="background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 20px; font-weight: bold; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #f39c12;">
                🥕 CARROT-KNN
            </div>
            <div style="margin-bottom: 15px;">
                <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #f39c12; height: 120px; display: flex; flex-direction: column;">
                    <div style="font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 8px;">Input</div>
                    <div style="font-size: 14px; color: #333; font-family: monospace; white-space: pre-wrap; overflow-y: auto; flex: 1;">{carrot_response_result.get('actual_prompt', carrot_response_result['original_query'])}</div>
                </div>
            </div>
            <div style="margin-bottom: 15px;">
                <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #f39c12; height: 120px; display: flex; flex-direction: column;">
                    <div style="font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 8px;">
                        <span style="text-decoration: underline; text-decoration-color: red; text-decoration-thickness: 2px;">Output</span>
                    </div>
                    <div style="font-size: 14px; color: #333; font-family: monospace; white-space: pre-wrap; overflow-y: auto; flex: 1;">{carrot_response_result['response']}</div>
                </div>
            </div>
            <div style="margin-bottom: 15px;">
                <div style="background: #e8f5e9; padding: 12px; border-radius: 5px; border-left: 4px solid #4caf50; height: 100px; display: flex; flex-direction: column;">
                    <div style="font-size: 12px; color: #2e7d32; font-weight: bold; margin-bottom: 8px;">Judge Feedback</div>
                    <div style="font-size: 14px; color: #333; overflow-y: auto; flex: 1;">{carrot_response_result['reasoning']}</div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #f39c12;">
                    <div style="font-size: 11px; color: #666; text-transform: uppercase;">Actual Quality</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 3px;">{carrot_response_result['actual_score']:.3f}</div>
                </div>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #f39c12;">
                    <div style="font-size: 11px; color: #666; text-transform: uppercase;">Actual Tokens</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 3px;">{carrot_response_result['actual_tokens']}</div>
                </div>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #f39c12;">
                    <div style="font-size: 11px; color: #666; text-transform: uppercase;">Actual Cost</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 3px;">{carrot_response_result['actual_cost']:.2f}</div>
                </div>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #f39c12;">
                    <div style="font-size: 11px; color: #666; text-transform: uppercase;">Actual Score</div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 3px;">{carrot_response_result['actual_risk']:.3f}</div>
                    <div style="font-size: 10px; color: #666; margin-top: 5px;">= (1-{lambda_val}) × {carrot_response_result['actual_score']:.3f} - {lambda_val} × {carrot_response_result['actual_cost']:.3f}</div>
                </div>
            </div>
        </div>
        """

    llm_responses_html += '</div>'

    # Generate Result Analysis with bar plots
    progress(0.9, desc="Generating results visualization...")
    result_analysis_fig = generate_result_analysis(core_response_result, carrot_response_result)

    # Combine: routing decisions + LLM responses (no result analysis HTML anymore)
    final_html = f"""
    <div style="margin-top: 20px;">
        <h3 style="font-size: 28px; font-weight: bold; color: #667eea; text-align: center; margin-bottom: 20px; padding: 15px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 10px;">📊 Routing Selections</h3>
        {routing_info_html}
    </div>

    <div style="margin-top: 40px;">
        <h3 style="font-size: 28px; font-weight: bold; color: #f39c12; text-align: center; margin-bottom: 20px; padding: 15px; background: linear-gradient(135deg, rgba(243, 156, 18, 0.1) 0%, rgba(230, 126, 34, 0.1) 100%); border-radius: 10px;">💬 LLM Responses</h3>
        {llm_responses_html}
    </div>
    """

    progress(1.0, desc="Done!")
    return final_html, result_analysis_fig


def format_prompt_with_highlight(original_query: str, actual_prompt: str, token_limit: str) -> str:
    """
    Format the prompt with highlighted instructional portion

    Args:
        original_query: The original user query
        actual_prompt: The actual prompt sent to LLM (may include instructional prompt)
        token_limit: The token limit used ("unlimited" or integer)

    Returns:
        HTML string with highlighted instructional prompt
    """
    import html

    # Escape HTML characters
    original_query_escaped = html.escape(original_query)
    actual_prompt_escaped = html.escape(actual_prompt)

    # Check if instructional prompt was added (for limited settings)
    if token_limit != "unlimited" and token_limit != "N/A":
        # The instructional prompt is the part after the original query
        if actual_prompt.startswith(original_query):
            # Split at the original query
            instructional_part = actual_prompt[len(original_query):]

            # Highlight the instructional part with inline styles
            return (
                f"{original_query_escaped}"
                f'<span style="background: #fff3cd; padding: 5px; border-radius: 3px; color: #856404; font-weight: bold;">{html.escape(instructional_part)}</span>'
            )

    # For unlimited or CARROT (no instructional prompt)
    return actual_prompt_escaped


def generate_results_html(
    query: str,
    lambda_val: float,
    budget: float,
    results: List[Dict],
    elapsed_time: float
) -> str:
    """Generate HTML for displaying results"""

    html = f"""
    <style>
        .results-container {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 750px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .query-box {{
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #667eea;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .method-card {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .method-header {{
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .metric-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .metric-box {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        .response-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }}
        .good {{
            color: #28a745;
        }}
        .bad {{
            color: #dc3545;
        }}
        .neutral {{
            color: #ffc107;
        }}
        .error-box {{
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #dc3545;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .comparison-table th, .comparison-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        .comparison-table th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        .comparison-table tr:hover {{
            background: #f8f9fa;
        }}
        .winner {{
            background: #d4edda !important;
        }}
        .input-frame {{
            background: #f0f8ff;
            padding: 15px;
            border-left: 4px solid #667eea;
            margin: 15px 0;
            border-radius: 5px;
            font-family: monospace;
            white-space: pre-wrap;
        }}
        .input-label {{
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        .instructional-prompt {{
            background: #fff3cd;
            padding: 5px;
            border-radius: 3px;
            color: #856404;
            font-weight: bold;
        }}
    </style>

    <div class="results-container">
        <div class="header">
            <h2>🤖 CoRE Router Results</h2>
            <p>Processed in {elapsed_time:.2f} seconds</p>
        </div>

        <div class="query-box">
            <strong>Query:</strong> {query}<br>
            <strong>Lambda (λ):</strong> {lambda_val}
        </div>
    """

    # Now add the side-by-side comparison table (no detailed results, go directly to comparison)
    valid_results = [r for r in results if "error" not in r]
    if len(valid_results) >= 2:
        # Assume we have CoRE and CARROT results
        core_result = None
        carrot_result = None

        for r in valid_results:
            if r["method"] == "CoRE":
                core_result = r
            elif "CARROT" in r["method"]:
                carrot_result = r

        if core_result and carrot_result:
            html += f"""
                <h3 style="text-align: center; color: #2c3e50; margin-top: 40px;">🔬 Method Comparison</h3>
                <table style="width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <thead>
                        <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                            <th style="padding: 15px; text-align: left; border: 1px solid #ddd;">Metric</th>
                            <th style="padding: 15px; text-align: center; border: 1px solid #ddd;">🎯 CoRE</th>
                            <th style="padding: 15px; text-align: center; border: 1px solid #ddd;">🥕 CARROT-KNN</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Selected LLM</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{core_result['llm_name']}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{carrot_result['llm_name']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Token Limit</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{core_result['token_limit']}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{carrot_result['token_limit']}</td>
                        </tr>
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Predicted Quality</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{core_result['predicted_score']:.3f}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{carrot_result['predicted_score']:.3f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Actual Quality</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center; color: {'green' if core_result['actual_score'] >= 0.7 else 'orange' if core_result['actual_score'] >= 0.4 else 'red'}; font-weight: bold;">{core_result['actual_score']:.3f}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center; color: {'green' if carrot_result['actual_score'] >= 0.7 else 'orange' if carrot_result['actual_score'] >= 0.4 else 'red'}; font-weight: bold;">{carrot_result['actual_score']:.3f}</td>
                        </tr>
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Quality Error</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{abs(core_result['actual_score'] - core_result['predicted_score']):.3f}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{abs(carrot_result['actual_score'] - carrot_result['predicted_score']):.3f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Predicted Cost</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{core_result['predicted_cost']:.2f}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{carrot_result['predicted_cost']:.2f}</td>
                        </tr>
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Actual Cost</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{core_result['actual_cost']:.2f}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{carrot_result['actual_cost']:.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Cost-Effectiveness</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center; font-weight: bold;">{core_result['actual_score'] / core_result['actual_cost'] if core_result['actual_cost'] > 0 else 0:.4f}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center; font-weight: bold;">{carrot_result['actual_score'] / carrot_result['actual_cost'] if carrot_result['actual_cost'] > 0 else 0:.4f}</td>
                        </tr>
                    </tbody>
                </table>
            """

    html += "</div>"

    return html


# ============================================================================
# Gradio Interface
# ============================================================================

def create_demo():
    """Create Gradio interface"""

    with gr.Blocks(
        title="CoRE Router Demo",
        theme=gr.themes.Soft(),
        css="""
        html, body {
            width: 100%;
            display: flex;
            justify-content: center;
            background-color: #fafafa;
        }
        .gradio-container {
            max-width: 1200px !important;
            width: 100%;
            margin: 0 auto !important;
            transition: none !important;
        }
        """
    ) as demo:
        gr.Markdown("""
        # 🤖 CoRE Router Demo

        Compare **CoRE** (Constrained Response Evaluator) against **CARROT** baselines for intelligent LLM routing.
        """)

        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="Enter your question here...",
                    lines=8,
                    value=config.EXAMPLE_QUERIES[0]
                )

            with gr.Column(scale=1):
                gr.Markdown('<h3 style="margin-bottom: 8px;">📚 Example Queries</h3>')
                for example in config.EXAMPLE_QUERIES:
                    gr.Button(example, size="sm").click(
                        fn=lambda x=example: x,
                        inputs=None,
                        outputs=query_input
                    )

        # Two separate tabs for different routing modes
        with gr.Tabs():
            # Tab 1: Direct Route with Lambda
            with gr.Tab("🚀 Direct Route"):
                gr.Markdown("""
                ### Algorithmic Routing with Lambda
                Select routing methods and lambda parameter. The system will automatically select the best LLM and token limit.
                """)

                lambda_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=config.DEFAULT_LAMBDA,
                    step=0.1,
                    label="Lambda (λ): Quality ← → Cost",
                    info="0 = prioritize quality, 1 = prioritize cost"
                )

                with gr.Row():
                    core_check = gr.Checkbox(
                        label="CoRE",
                        value=True
                    )
                    knn_check = gr.Checkbox(
                        label="CARROT-KNN",
                        value=True
                    )

                submit_btn = gr.Button("🚀 Route Query", variant="primary", size="lg")

                direct_results = gr.HTML(label="Direct Route Results")
                result_analysis_plot = gr.Plot(label="📈 Result Analysis")

            # Tab 2: Visualized Route with Manual Selection
            with gr.Tab("📊 Visualized Route"):
                gr.Markdown("""
                ### Visual Selection from Cost-Quality Plots
                View predictions, then manually select LLM and token limit based on the visualizations.
                """)

                visualize_btn = gr.Button("📊 Show Visualizations", variant="secondary", size="lg")

                # Visualization plots
                with gr.Row():
                    core_plot = gr.Plot(label="CoRE: Cost-Quality Curves", visible=False)
                    carrot_knn_plot = gr.Plot(label="CARROT-KNN: Cost-Quality Points", visible=False)

                # Selection UI (both methods in one compact layout)
                with gr.Column(visible=False) as selection_row:
                    gr.Markdown("### 🎯 Make Your Selections")

                    # CoRE Selection - inline (LLM + Token Limit in one row)
                    gr.Markdown("**CoRE Selection:** Choose LLM and token limit from the curves above")
                    with gr.Row():
                        core_llm_dropdown = gr.Dropdown(
                            label="CoRE: LLM",
                            choices=list(config.LLM_POOL.keys()),
                            value=None,
                            scale=2
                        )
                        core_token_dropdown = gr.Dropdown(
                            label="CoRE: Token Limit",
                            choices=["10", "20", "30", "40", "50", "80", "100", "150", "200", "300", "500", "800", "1200", "2000", "4000", "unlimited"],
                            value="unlimited",
                            scale=1
                        )

                    # CARROT Selection - single LLM dropdown
                    gr.Markdown("**CARROT-KNN Selection:** Choose LLM from the points above (always unlimited)")
                    carrot_llm_dropdown = gr.Dropdown(
                        label="CARROT: LLM",
                        choices=list(config.LLM_POOL.keys()),
                        value=None
                    )

                    # Single button to run both
                    run_both_btn = gr.Button("▶️ Run Both Methods & Compare", variant="primary", size="lg")

                visualized_results = gr.HTML(label="Comparison Results")
                visualized_analysis_plot = gr.Plot(label="📈 Result Analysis")

        # Connect Direct Route button
        submit_btn.click(
            fn=process_query,
            inputs=[
                query_input,
                lambda_input,
                core_check,
                knn_check
            ],
            outputs=[direct_results, result_analysis_plot]
        )

        # Connect Visualize button
        def show_visualizations(query):
            core_fig, carrot_fig = generate_visualizations(query)
            return {
                core_plot: gr.update(value=core_fig, visible=True),
                carrot_knn_plot: gr.update(value=carrot_fig, visible=True),
                selection_row: gr.update(visible=True)
            }

        visualize_btn.click(
            fn=show_visualizations,
            inputs=[query_input],
            outputs=[core_plot, carrot_knn_plot, selection_row]
        )

        # Connect single button to run both methods
        def run_both_methods_and_compare(query, core_llm, core_token, carrot_llm, progress=gr.Progress()):
            """Run both CoRE and CARROT selections with step-by-step progress, then display comparison table"""
            if not query.strip():
                return "<p style='color: red;'>Please enter a query!</p>", None

            if not core_llm or not carrot_llm:
                return "<p style='color: red;'>Please select LLMs for both CoRE and CARROT-KNN!</p>", None

            # Step 1: Generate embedding
            progress(0.1, desc="Generating query embedding...")
            embedding = embedder.embed(query)

            # Step 2: Computing predictions
            progress(0.2, desc=f"Computing predictions for CoRE ({core_llm}) and CARROT-KNN ({carrot_llm})...")

            # Get predictions (without running inference yet)
            # CoRE predictions
            predictor = core_router.predictors[core_llm]
            llm_size = core_router.llm_pool[core_llm]["size"]
            quality_limited, quality_unlimited, token_count_unlimited = predictor.predict(embedding)
            quality_limited = np.clip(quality_limited, 0, 1)
            quality_unlimited = np.clip(quality_unlimited, 0, 1)
            token_count_unlimited = np.maximum(token_count_unlimited, 0)

            if core_token == "unlimited":
                core_pred_score = quality_unlimited[0]
                core_pred_count = token_count_unlimited[0]
            else:
                token_limit_int = int(core_token)
                limited_token_limits = [t for t in core_router.token_limits if t != "unlimited"]
                token_idx = limited_token_limits.index(token_limit_int)
                core_pred_score = quality_limited[0, token_idx]
                core_pred_count = min(token_limit_int, token_count_unlimited[0])
            core_pred_cost = core_pred_count * llm_size

            # CARROT predictions
            Y_hat_score, Y_hat_count = carrot_knn_router.model.predict(embedding)
            # Y_hat_score = np.clip(Y_hat_score, 0, 1)
            Y_hat_count = np.maximum(Y_hat_count, 0)

            # DEBUG: Print CARROT prediction details
            print(f"\n{'='*60}")
            print(f"DEBUG: CARROT Extraction (run_both_methods_and_compare)")
            print(f"{'='*60}")
            print(f"Embedding shape: {embedding.shape}")
            print(f"Y_hat_score shape: {Y_hat_score.shape}")
            print(f"Y_hat_count shape: {Y_hat_count.shape}")
            print(f"Y_hat_score: {Y_hat_score[0]}")
            print(f"Y_hat_count: {Y_hat_count[0]}")

            llm_names = list(carrot_knn_router.llm_pool.keys())
            print(f"LLM pool: {llm_names} (count={len(llm_names)})")
            print(f"Requested LLM: {carrot_llm}")

            llm_idx = llm_names.index(carrot_llm)
            print(f"LLM index: {llm_idx}")

            carrot_llm_size = carrot_knn_router.llm_pool[carrot_llm]["size"]
            carrot_pred_score = Y_hat_score[0, llm_idx]
            carrot_pred_count = Y_hat_count[0, llm_idx]
            carrot_pred_cost = carrot_pred_count * carrot_llm_size

            print(f"Extracted: score={carrot_pred_score:.3f}, count={carrot_pred_count:.1f}, cost={carrot_pred_cost:.1f}")
            print(f"{'='*60}\n")

            # Step 3: Calling CoRE LLM
            progress(0.4, desc=f"Querying CoRE LLM ({core_llm} @ {core_token})...")
            core_response, core_tokens, core_actual_prompt = llm_client.call_llm_by_name(core_llm, query, core_token)
            core_actual_cost = core_tokens * llm_size

            # Step 4: Calling CARROT LLM
            progress(0.6, desc=f"Querying CARROT-KNN LLM ({carrot_llm} @ unlimited)...")
            carrot_response, carrot_tokens, carrot_actual_prompt = llm_client.call_llm_by_name(carrot_llm, query, "unlimited")
            carrot_actual_cost = carrot_tokens * carrot_llm_size

            # Step 5: Evaluating with judge
            progress(0.8, desc="Evaluating responses with judge model...")

            # Evaluate both responses
            core_score, core_feedback = judge.evaluate(query, core_response)
            carrot_score, carrot_feedback = judge.evaluate(query, carrot_response)

            # Calculate metrics
            core_score_error = abs(core_pred_score - core_score)
            carrot_score_error = abs(carrot_pred_score - carrot_score)
            core_cost_effectiveness = core_score / core_actual_cost if core_actual_cost > 0 else 0
            carrot_cost_effectiveness = carrot_score / carrot_actual_cost if carrot_actual_cost > 0 else 0

            # Package results
            core_result = {
                'llm_name': core_llm,
                'token_limit': core_token,
                'predicted_score': float(core_pred_score),
                'predicted_cost': float(core_pred_cost),
                'actual_score': float(core_score),
                'actual_cost': float(core_actual_cost),
                'score_error': float(core_score_error),
                'cost_effectiveness': float(core_cost_effectiveness),
                'response': core_response,
                'judge_feedback': core_feedback
            }

            carrot_result = {
                'llm_name': carrot_llm,
                'token_limit': 'unlimited',
                'predicted_score': float(carrot_pred_score),
                'predicted_cost': float(carrot_pred_cost),
                'actual_score': float(carrot_score),
                'actual_cost': float(carrot_actual_cost),
                'score_error': float(carrot_score_error),
                'cost_effectiveness': float(carrot_cost_effectiveness),
                'response': carrot_response,
                'judge_feedback': carrot_feedback
            }

            # Format the CoRE input with highlighted instructional prompt
            core_input_html_viz = format_prompt_with_highlight(
                query,
                core_actual_prompt,
                str(core_token)
            )

            # Final: Build comparison table with card-style selections (side-by-side)
            selections_html = f"""
            <div style="margin-top: 20px;">
                <h3 style="font-size: 28px; font-weight: bold; color: #667eea; text-align: center; margin-bottom: 20px; padding: 15px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 10px;">📊 Routing Selections</h3>

                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 15px 0;">
                    <div style="background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <div style="font-size: 20px; font-weight: bold; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #667eea;">
                            🎯 CoRE
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin: 15px 0;">
                            <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #667eea;">
                                <div style="font-size: 12px; color: #666; text-transform: uppercase;">Selected LLM</div>
                                <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{core_result['llm_name']}</div>
                            </div>
                            <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #667eea;">
                                <div style="font-size: 12px; color: #666; text-transform: uppercase;">Token Limit</div>
                                <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{core_result['token_limit']}</div>
                            </div>
                            <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #667eea;">
                                <div style="font-size: 12px; color: #666; text-transform: uppercase;">Predicted Score</div>
                                <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{core_result['predicted_score']:.3f}</div>
                            </div>
                            <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #667eea;">
                                <div style="font-size: 12px; color: #666; text-transform: uppercase;">Predicted Cost</div>
                                <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{core_result['predicted_cost']:.1f}</div>
                            </div>
                        </div>
                    </div>

                    <div style="background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <div style="font-size: 20px; font-weight: bold; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #f39c12;">
                            🥕 CARROT-KNN
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin: 15px 0;">
                            <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #f39c12;">
                                <div style="font-size: 12px; color: #666; text-transform: uppercase;">Selected LLM</div>
                                <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{carrot_result['llm_name']}</div>
                            </div>
                            <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #f39c12;">
                                <div style="font-size: 12px; color: #666; text-transform: uppercase;">Token Limit</div>
                                <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">unlimited</div>
                            </div>
                            <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #f39c12;">
                                <div style="font-size: 12px; color: #666; text-transform: uppercase;">Predicted Score</div>
                                <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{carrot_result['predicted_score']:.3f}</div>
                            </div>
                            <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #f39c12;">
                                <div style="font-size: 12px; color: #666; text-transform: uppercase;">Predicted Cost</div>
                                <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">{carrot_result['predicted_cost']:.1f}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """

            html = selections_html + f"""
                <h3 style="text-align: center; color: #2c3e50; margin-top: 30px;">🔬 Method Comparison</h3>
                <table style="width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <thead>
                        <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                            <th style="padding: 15px; text-align: left; border: 1px solid #ddd;">Metric</th>
                            <th style="padding: 15px; text-align: center; border: 1px solid #ddd;">🎯 CoRE</th>
                            <th style="padding: 15px; text-align: center; border: 1px solid #ddd;">🥕 CARROT-KNN</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Selected LLM</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{core_result['llm_name']}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{carrot_result['llm_name']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Token Limit</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{core_result['token_limit']}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{carrot_result['token_limit']}</td>
                        </tr>
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Predicted Quality</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{core_result['predicted_score']:.3f}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{carrot_result['predicted_score']:.3f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Actual Quality</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center; color: {'green' if core_result['actual_score'] >= 0.7 else 'orange' if core_result['actual_score'] >= 0.4 else 'red'}; font-weight: bold;">{core_result['actual_score']:.3f}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center; color: {'green' if carrot_result['actual_score'] >= 0.7 else 'orange' if carrot_result['actual_score'] >= 0.4 else 'red'}; font-weight: bold;">{carrot_result['actual_score']:.3f}</td>
                        </tr>
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Quality Error</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{core_result['score_error']:.3f}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{carrot_result['score_error']:.3f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Predicted Cost</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{core_result['predicted_cost']:.2f}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{carrot_result['predicted_cost']:.2f}</td>
                        </tr>
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Actual Cost</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{core_result['actual_cost']:.2f}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center;">{carrot_result['actual_cost']:.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Cost-Effectiveness</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center; font-weight: bold;">{core_result['cost_effectiveness']:.4f}</td>
                            <td style="padding: 12px; border: 1px solid #ddd; text-align: center; font-weight: bold;">{carrot_result['cost_effectiveness']:.4f}</td>
                        </tr>
                    </tbody>
                </table>

                <h3 style="font-size: 28px; font-weight: bold; color: #f39c12; text-align: center; margin-top: 40px; margin-bottom: 20px; padding: 15px; background: linear-gradient(135deg, rgba(243, 156, 18, 0.1) 0%, rgba(230, 126, 34, 0.1) 100%); border-radius: 10px;">💬 LLM Responses</h3>

                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 15px 0;">
                    <div style="background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <div style="font-size: 20px; font-weight: bold; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #667eea;">
                            🎯 CoRE
                        </div>
                        <div style="margin-bottom: 15px;">
                            <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #667eea; height: 120px; display: flex; flex-direction: column;">
                                <div style="font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 8px;">Input</div>
                                <div style="font-size: 14px; color: #333; font-family: monospace; white-space: pre-wrap; overflow-y: auto; flex: 1;">{core_input_html_viz}</div>
                            </div>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #667eea; height: 120px; display: flex; flex-direction: column;">
                                <div style="font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 8px;">
                                    <span style="text-decoration: underline; text-decoration-color: red; text-decoration-thickness: 2px;">Output</span>
                                </div>
                                <div style="font-size: 14px; color: #333; font-family: monospace; white-space: pre-wrap; overflow-y: auto; flex: 1;">{core_result['response']}</div>
                            </div>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <div style="background: #e8f5e9; padding: 12px; border-radius: 5px; border-left: 4px solid #4caf50; height: 100px; display: flex; flex-direction: column;">
                                <div style="font-size: 12px; color: #2e7d32; font-weight: bold; margin-bottom: 8px;">Judge Feedback</div>
                                <div style="font-size: 14px; color: #333; overflow-y: auto; flex: 1;">{core_result['judge_feedback']}</div>
                            </div>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #667eea;">
                                <div style="font-size: 11px; color: #666; text-transform: uppercase;">Actual Quality</div>
                                <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 3px;">{core_result['actual_score']:.3f}</div>
                            </div>
                            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #667eea;">
                                <div style="font-size: 11px; color: #666; text-transform: uppercase;">Actual Tokens</div>
                                <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 3px;">{core_tokens}</div>
                            </div>
                            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #667eea;">
                                <div style="font-size: 11px; color: #666; text-transform: uppercase;">Actual Cost</div>
                                <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 3px;">{core_result['actual_cost']:.2f}</div>
                            </div>
                        </div>
                    </div>

                    <div style="background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <div style="font-size: 20px; font-weight: bold; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #f39c12;">
                            🥕 CARROT-KNN
                        </div>
                        <div style="margin-bottom: 15px;">
                            <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #f39c12; height: 120px; display: flex; flex-direction: column;">
                                <div style="font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 8px;">Input</div>
                                <div style="font-size: 14px; color: #333; font-family: monospace; white-space: pre-wrap; overflow-y: auto; flex: 1;">{carrot_actual_prompt}</div>
                            </div>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #f39c12; height: 120px; display: flex; flex-direction: column;">
                                <div style="font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 8px;">
                                    <span style="text-decoration: underline; text-decoration-color: red; text-decoration-thickness: 2px;">Output</span>
                                </div>
                                <div style="font-size: 14px; color: #333; font-family: monospace; white-space: pre-wrap; overflow-y: auto; flex: 1;">{carrot_result['response']}</div>
                            </div>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <div style="background: #e8f5e9; padding: 12px; border-radius: 5px; border-left: 4px solid #4caf50; height: 100px; display: flex; flex-direction: column;">
                                <div style="font-size: 12px; color: #2e7d32; font-weight: bold; margin-bottom: 8px;">Judge Feedback</div>
                                <div style="font-size: 14px; color: #333; overflow-y: auto; flex: 1;">{carrot_result['judge_feedback']}</div>
                            </div>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #f39c12;">
                                <div style="font-size: 11px; color: #666; text-transform: uppercase;">Actual Quality</div>
                                <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 3px;">{carrot_result['actual_score']:.3f}</div>
                            </div>
                            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #f39c12;">
                                <div style="font-size: 11px; color: #666; text-transform: uppercase;">Actual Tokens</div>
                                <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 3px;">{carrot_tokens}</div>
                            </div>
                            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 3px solid #f39c12;">
                                <div style="font-size: 11px; color: #666; text-transform: uppercase;">Actual Cost</div>
                                <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 3px;">{carrot_result['actual_cost']:.2f}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """

            # Add Result Analysis
            progress(0.95, desc="Generating results visualization...")
            result_analysis_fig = generate_result_analysis(core_result, carrot_result)

            progress(1.0, desc="Done!")
            return html, result_analysis_fig

        run_both_btn.click(
            fn=run_both_methods_and_compare,
            inputs=[query_input, core_llm_dropdown, core_token_dropdown, carrot_llm_dropdown],
            outputs=[visualized_results, visualized_analysis_plot]
        )

        gr.Markdown("""
        ---
        ### How it works:
        1. **Query Embedding**: Your query is converted to a vector representation
        2. **Routing**: Each method predicts scores and costs for all (LLM, token_limit) combinations
        3. **Selection**: The best option is chosen based on: `trade-off score = (1-λ) × score - λ × cost`
        4. **Inference**: The selected LLM generates a response
        5. **Evaluation**: A judge LLM evaluates the response quality

        ### Metrics:
        - **Actual Quality**: Quality rating from judge (0-1, higher is better)
        - **Actual Trade-off Score**: Combined utility score = (1-λ) × quality - λ × cost
        - **Actual Cost**: Token count × price per token
        """)

    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    demo = create_demo()

    # Launch options
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Default Gradio port
        share=True,              # Create public link via Gradio share
        show_error=True
    )
