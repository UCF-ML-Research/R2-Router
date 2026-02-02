"""Compare R2-Router vs CARROT baselines."""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..shared.dataset_manager import DatasetManager
from ..shared.llm_loader import load_llm
from ..r2.predictor_sklearn import TokenPerformancePredictor
from ..r2.predictor import route_scores
from ..baselines.carrot.baselines_carrot import CarrotKNNBaseline, CarrotLinearBaseline, route_baseline
from ..baselines.irt.baselines_irt import IRTBaseline, NIRTBaseline


def parse_args():
    parser = argparse.ArgumentParser(description='Compare R2-Router vs CARROT')
    parser.add_argument('--model', action='append', nargs=4,
                        metavar=('NAME', 'SIZE', 'CSV', 'CHECKPOINT'),
                        help='Model configuration: name size csv_path checkpoint_path')
    parser.add_argument('--lambda-dist', type=str, default='0,0.2,20;0.2,1.0,50',
                        help='Lambda distribution: "min,max,num;min,max,num;..." (default: 0,0.2,20;0.2,1.0,50)')
    parser.add_argument('--target-accuracy-rate', type=float, default=1.0,
                        help='Target accuracy rate for QNC (default: 1.0 for 100%%). E.g., 0.9 means 90%% of best LLM')
    return parser.parse_args()


def parse_lambda_distribution(lambda_dist_str):
    """
    Parse lambda distribution string into numpy array.

    Format: "min,max,num;min,max,num;..."
    Example: "0,0.2,20;0.2,1.0,50" -> concatenate linspace(0,0.2,20) and linspace(0.2,1.0,50)
    """
    segments = []
    for segment in lambda_dist_str.split(';'):
        parts = segment.split(',')
        if len(parts) != 3:
            raise ValueError(f"Invalid lambda distribution segment: {segment}. Expected 'min,max,num'")
        min_val, max_val, num_points = float(parts[0]), float(parts[1]), int(parts[2])
        segments.append(np.linspace(min_val, max_val, num_points))
    return np.unique(np.concatenate(segments))


def calculate_audc(cost_curve, perf_curve, normalize=True, global_cost_range=None):
    """
    Area Under Deferral Curve.

    Args:
        cost_curve: Array of cost values
        perf_curve: Array of performance values
        normalize: If True, normalize cost to [0,1] before integration
        global_cost_range: Tuple of (min_cost, max_cost) for global normalization.
                          If None, normalize using this method's cost range.

    Returns:
        AUDC value
    """
    sorted_indices = np.argsort(cost_curve)
    sorted_cost = cost_curve[sorted_indices]
    sorted_perf = perf_curve[sorted_indices]

    if normalize:
        # Use global normalization if provided, otherwise per-method normalization
        if global_cost_range is not None:
            min_cost, max_cost = global_cost_range
        else:
            min_cost, max_cost = sorted_cost.min(), sorted_cost.max()

        if max_cost > min_cost:
            x_values = (sorted_cost - min_cost) / (max_cost - min_cost)
        else:
            x_values = np.zeros_like(sorted_cost)
    else:
        x_values = sorted_cost

    return np.trapezoid(sorted_perf, x_values)


def calculate_qnc(cost_curve, perf_curve, normalize=True, target_perf=None, global_cost_range=None, target_accuracy_rate=None):
    """
    Calculate Query-Normalized Cost (QNC) or Actual Cost to reach target performance.

    QNC is the relative cost to achieve the same performance as the most accurate single LLM.
    If a method cannot reach the target, QNC = 1.0 (100%).

    Args:
        cost_curve: Array of cost values
        perf_curve: Array of performance values
        normalize: If True, return normalized cost [0,1]; if False, return actual cost
        target_perf: Target performance to reach. If None, use 90% of this method's peak.
        global_cost_range: Tuple of (min_cost, max_cost) for global normalization.
                          If None, normalize using this method's cost range.
        target_accuracy_rate: If provided with target_perf, multiply target_perf by this rate.
                             E.g., 0.9 means target is 90% of target_perf.

    Returns:
        Cost value at which performance reaches target performance
        Lower is better (achieving high performance with lower cost)
    """
    # If target_perf not specified, use 90% of this method's peak (backward compatible)
    if target_perf is None:
        peak_perf = np.max(perf_curve)
        target_perf = 0.9 * peak_perf
    elif target_accuracy_rate is not None:
        # Apply target_accuracy_rate to target_perf
        target_perf = target_perf * target_accuracy_rate

    # Find the cost at which performance reaches target
    sorted_indices = np.argsort(cost_curve)
    sorted_cost = cost_curve[sorted_indices]
    sorted_perf = perf_curve[sorted_indices]

    # Find where performance crosses target threshold
    idx = np.where(sorted_perf >= target_perf)[0]
    if len(idx) == 0:
        # Never reaches target - return max cost (QNC = 1.0 if normalized)
        if normalize:
            return 1.0
        else:
            return sorted_cost[-1]

    # Get the cost at first crossing
    cost_at_target = sorted_cost[idx[0]]

    if normalize:
        # Normalize cost to [0, 1]
        if global_cost_range is not None:
            min_cost, max_cost = global_cost_range
        else:
            min_cost, max_cost = sorted_cost.min(), sorted_cost.max()

        if max_cost > min_cost:
            normalized_cost = (cost_at_target - min_cost) / (max_cost - min_cost)
        else:
            normalized_cost = 0.0
        return normalized_cost
    else:
        return cost_at_target


def calculate_oracle_unlimited(llms, lamb_range):
    """Oracle with unlimited token setting only."""
    llm_names = list(llms.keys())
    sizes_vec = np.array([llms[name]["size"] for name in llm_names])[None, :]
    Y_score_true = np.stack([llms[name]["true_test_unlimited_score"] for name in llm_names], axis=1)
    Y_count_true = np.stack([llms[name]["true_test_unlimited_count"] for name in llm_names], axis=1)
    true_cost_mat = Y_count_true * sizes_vec
    oracle_cost, oracle_perf = [], []
    n_queries = Y_score_true.shape[0]
    for lam in lamb_range:
        oracle_risk = (1 - lam) * Y_score_true - lam * true_cost_mat
        chosen = oracle_risk.argmax(axis=1)
        oracle_perf.append(Y_score_true[np.arange(n_queries), chosen].mean())
        oracle_cost.append(true_cost_mat[np.arange(n_queries), chosen].mean())
    return np.array(oracle_cost), np.array(oracle_perf)


def calculate_oracle_all_limits(llms, lamb_range):
    """Oracle with all token limits."""
    llm_names = list(llms.keys())
    scores_blocks, costs_blocks = [], []
    for name in llm_names:
        s_limited = llms[name]["true_test_score"]
        c_limited = llms[name]["true_test_count"] * llms[name]["size"]
        s_unlim = llms[name]["true_test_unlimited_score"][:, None]
        c_unlim = (llms[name]["true_test_unlimited_count"] * llms[name]["size"])[:, None]
        scores_blocks.append(np.concatenate([s_limited, s_unlim], axis=1))
        costs_blocks.append(np.concatenate([c_limited, c_unlim], axis=1))
    Y_score_all = np.concatenate(scores_blocks, axis=1)
    Y_cost_all = np.concatenate(costs_blocks, axis=1)
    oracle_cost, oracle_perf = [], []
    n_queries = Y_score_all.shape[0]
    for lam in lamb_range:
        oracle_risk = (1 - lam) * Y_score_all - lam * Y_cost_all
        chosen = oracle_risk.argmax(axis=1)
        oracle_perf.append(Y_score_all[np.arange(n_queries), chosen].mean())
        oracle_cost.append(Y_cost_all[np.arange(n_queries), chosen].mean())
    return np.array(oracle_cost), np.array(oracle_perf)


def main():
    args = parse_args()

    TOKEN_LIMITS_SCORE = [
        '10_score', '20_score', '30_score', '40_score', '50_score',
        '80_score', '100_score', '150_score', '200_score', '300_score',
        '500_score', '800_score', '1200_score', '2000_score', '4000_score',
        'unlimited_score'
    ]
    TOKEN_LIMITS_COUNT = [
        '10_count', '20_count', '30_count', '40_count', '50_count',
        '80_count', '100_count', '150_count', '200_count', '300_count',
        '500_count', '800_count', '1200_count', '2000_count', '4000_count',
        'unlimited_count'
    ]
    # Lambda range for cost-performance tradeoff
    # Formula: score = (1-λ)*quality - λ*cost
    # λ ∈ [0,1]: λ=0 pure quality, λ=1 pure cost minimization
    LAMBDA_RANGE = parse_lambda_distribution(args.lambda_dist)
    print(f"\nLambda distribution: {args.lambda_dist}")
    print(f"Lambda points: {len(LAMBDA_RANGE)} (range: [{LAMBDA_RANGE.min():.4f}, {LAMBDA_RANGE.max():.4f}])")

    # Load dataset
    dataset_manager = DatasetManager(
        embeddings_path="data/prompt_embeddings.pkl",
        train_ratio=0.8,
        seed=42
    )
    embeddings = dataset_manager.get_embeddings()
    train_idx, test_idx = dataset_manager.get_split_indices()

    # Load R2-Router models
    print("\nLoading R2-Router models...")
    print(f"Attempting to load {len(args.model)} models...")
    llms = {}
    for name, size, csv_path, checkpoint in args.model:
        print(f"\n--- Checking {name} ---")
        print(f"    CSV: {csv_path}")
        print(f"    Checkpoint: {checkpoint}")
        # Check CSV first
        if not os.path.exists(csv_path):
            print(f"[SKIP] {name}: CSV not found at {csv_path}")
            continue

        # Check checkpoint directory
        if not os.path.exists(checkpoint):
            print(f"[SKIP] {name}: Checkpoint not found at {checkpoint}")
            continue

        # Check if checkpoint contains required files
        required_files = [
            os.path.join(checkpoint, "limited_score_predictors.joblib"),
            os.path.join(checkpoint, "unlimited_score_predictor.joblib"),
            os.path.join(checkpoint, "unlimited_token_predictor.joblib")
        ]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"[SKIP] {name}: Checkpoint incomplete. Missing files:")
            for f in missing_files:
                print(f"       - {os.path.basename(f)}")
            continue

        print(f"  Loading {name}...")
        try:
            llms[name] = load_llm(
                name=name,
                size=float(size),
                score_df_path=csv_path,
                load_dir=checkpoint,
                embeddings=embeddings,
                train_idx=train_idx,
                test_idx=test_idx,
                token_limits_score=TOKEN_LIMITS_SCORE,
                token_limits_count=TOKEN_LIMITS_COUNT,
                predictor_class=TokenPerformancePredictor
            )
        except Exception as e:
            print(f"[ERROR] Failed to load {name}: {e}")
            continue

    print(f"\n[OK] Loaded {len(llms)} R2-Router models")

    if len(llms) == 0:
        print("\n[ERROR] No R2-Router models were loaded!")
        print("Please check that:")
        print("  1. R2-Router models have been trained (Step 1 completed)")
        print("  2. Checkpoint paths are correct")
        print("  3. CSV files exist")
        exit(1)

    # Route with R2-Router
    print("\nRouting with R2-Router...")
    r2_cost, core_perf = route_scores(llms, LAMBDA_RANGE)

    # Load and route with CARROT
    print("\nLoading CARROT models...")
    llm_names = list(llms.keys())
    embedding_test = embeddings[test_idx]
    quality_test = np.stack([llms[name]["true_test_unlimited_score"] for name in llm_names], axis=1)
    token_count_test = np.stack([llms[name]["true_test_unlimited_count"] for name in llm_names], axis=1)
    sizes_vec = np.array([llms[name]["size"] for name in llm_names])[None, :]

    # CARROT-KNN
    carrot_knn = CarrotKNNBaseline(load_dir="./checkpoints/main/carrot_knn")
    Y_hat_score_knn, Y_hat_count_knn = carrot_knn.predict(embedding_test)
    carrot_knn_cost, carrot_knn_perf = route_baseline(
        Y_hat_score_knn, Y_hat_count_knn, quality_test, token_count_test, LAMBDA_RANGE, sizes_vec
    )
    print("[OK] CARROT-KNN routing done")

    # CARROT-Linear
    carrot_linear = CarrotLinearBaseline(load_dir="./checkpoints/main/carrot_linear")
    Y_hat_score_linear, Y_hat_count_linear = carrot_linear.predict(embedding_test)
    carrot_linear_cost, carrot_linear_perf = route_baseline(
        Y_hat_score_linear, Y_hat_count_linear, quality_test, token_count_test, LAMBDA_RANGE, sizes_vec
    )
    print("[OK] CARROT-Linear routing done")

    # Try to load IRT baselines
    try:
        # MIRT
        print("\nLoading MIRT baseline...")
        mirt = IRTBaseline(load_dir="./checkpoints/main/irt_mirt")
        Y_hat_score_mirt = mirt.predict(embedding_test)

        # IRT only routes among LLMs (no token budget optimization)
        # Cost is based only on model size (price per token), not token usage
        # Use constant token count for all models so cost ∝ model_size only
        mean_token_count = token_count_test.mean()  # Average unlimited token count across all queries/models
        constant_token_count = np.full_like(quality_test, mean_token_count)  # Same value for all LLMs
        mirt_cost, mirt_perf = route_baseline(
            Y_hat_score_mirt, constant_token_count, quality_test, token_count_test, LAMBDA_RANGE, sizes_vec
        )
        print("[OK] MIRT routing done")
        has_mirt = True
    except Exception as e:
        print(f"[SKIP] MIRT not available: {e}")
        has_mirt = False

    try:
        # NIRT
        print("\nLoading NIRT baseline...")
        nirt = NIRTBaseline(load_dir="./checkpoints/main/irt_nirt")
        # NIRT generates relevance vectors automatically when embedding is passed
        Y_hat_score_nirt = nirt.predict(embedding_test)

        # IRT only routes among LLMs (no token budget optimization)
        # Cost is based only on model size (price per token), not token usage
        # Use constant token count for all models so cost ∝ model_size only
        mean_token_count = token_count_test.mean()  # Average unlimited token count across all queries/models
        constant_token_count = np.full_like(quality_test, mean_token_count)  # Same value for all LLMs
        nirt_cost, nirt_perf = route_baseline(
            Y_hat_score_nirt, constant_token_count, quality_test, token_count_test, LAMBDA_RANGE, sizes_vec
        )
        print("[OK] NIRT routing done")
        has_nirt = True
    except Exception as e:
        print(f"[SKIP] NIRT not available: {e}")
        has_nirt = False

    # Calculate Oracle curves
    print("\nCalculating Oracle curves...")
    oracle_unlimited_cost, oracle_unlimited_perf = calculate_oracle_unlimited(llms, LAMBDA_RANGE)
    oracle_all_cost, oracle_all_perf = calculate_oracle_all_limits(llms, LAMBDA_RANGE)

    # Save CSV
    print("\nSaving results...")
    os.makedirs("./comparison_results/main", exist_ok=True)
    data = []
    methods_to_save = [
        ('R2-Router', r2_cost, core_perf),
        ('CARROT-KNN', carrot_knn_cost, carrot_knn_perf),
        ('CARROT-Linear', carrot_linear_cost, carrot_linear_perf),
        ('Oracle (Unlimited)', oracle_unlimited_cost, oracle_unlimited_perf),
        ('Oracle (All Limits)', oracle_all_cost, oracle_all_perf),
    ]
    if has_mirt:
        methods_to_save.append(('MIRT', mirt_cost, mirt_perf))
    if has_nirt:
        methods_to_save.append(('NIRT', nirt_cost, nirt_perf))

    # Add both actual and normalized cost to curves CSV
    for method, cost, perf in methods_to_save:
        # Normalize cost for this method
        cost_min, cost_max = cost.min(), cost.max()
        if cost_max > cost_min:
            normalized_cost = (cost - cost_min) / (cost_max - cost_min)
        else:
            normalized_cost = np.zeros_like(cost)

        for lam, c, nc, p in zip(LAMBDA_RANGE, cost, normalized_cost, perf):
            data.append({
                'method': method,
                'lambda': lam,
                'cost': c,
                'normalized_cost': nc,
                'performance': p
            })
    df = pd.DataFrame(data)
    df.to_csv("./comparison_results/main/r2_vs_baselines_curves.csv", index=False)
    print("[OK] Saved CSV: ./comparison_results/main/r2_vs_baselines_curves.csv")

    # Plot
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10.colors
    ax.plot(oracle_unlimited_cost, oracle_unlimited_perf, '-', label="Oracle (Unlimited)", linewidth=2.5, color="red", alpha=0.8)
    ax.plot(oracle_all_cost, oracle_all_perf, '-', label="Oracle (All Limits)", linewidth=2.5, color="brown", alpha=0.8)
    ax.plot(r2_cost, core_perf, '-', label="R2-Router (Our Method)", linewidth=2.5, color="black")
    ax.plot(carrot_knn_cost, carrot_knn_perf, '--', label="CARROT-KNN", linewidth=2, color="orange")
    ax.plot(carrot_linear_cost, carrot_linear_perf, '--', label="CARROT-Linear", linewidth=2, color="green")
    if has_mirt:
        ax.plot(mirt_cost, mirt_perf, '-.', label="MIRT", linewidth=2, color="purple")
    if has_nirt:
        ax.plot(nirt_cost, nirt_perf, '-.', label="NIRT", linewidth=2, color="magenta")
    for i, name in enumerate(llm_names):
        score = llms[name]["true_test_unlimited_score"].mean()
        cost = (llms[name]["true_test_unlimited_count"] * llms[name]["size"]).mean()
        ax.scatter(cost, score, marker="*", s=200, color=colors[i % len(colors)],
                  edgecolor="black", linewidth=1.5, label=f"{name}", zorder=5)
    ax.set_xlabel("Cost (Token Count × Model Size)", fontsize=12)
    ax.set_ylabel("Average Quality Score", fontsize=12)
    title_methods = "R2-Router vs Baselines"
    if has_mirt or has_nirt:
        title_methods += " (including IRT)"
    ax.set_title(f"Quality-Cost Tradeoff: {title_methods} ({len(llm_names)} LLMs)", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./comparison_results/main/r2_vs_baselines_curves_actual.png", dpi=150, bbox_inches='tight')
    print("[OK] Saved plot: ./comparison_results/main/r2_vs_baselines_curves_actual.png")
    plt.close()

    # Generate normalized cost plot
    print("\nGenerating normalized cost plot...")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Normalize costs for each method
    def normalize_cost(cost_array):
        """Normalize cost to [0,1] range."""
        cost_min, cost_max = cost_array.min(), cost_array.max()
        if cost_max > cost_min:
            return (cost_array - cost_min) / (cost_max - cost_min)
        else:
            return np.zeros_like(cost_array)

    oracle_unlimited_norm = normalize_cost(oracle_unlimited_cost)
    oracle_all_norm = normalize_cost(oracle_all_cost)
    core_norm = normalize_cost(r2_cost)
    carrot_knn_norm = normalize_cost(carrot_knn_cost)
    carrot_linear_norm = normalize_cost(carrot_linear_cost)
    if has_mirt:
        mirt_norm = normalize_cost(mirt_cost)
    if has_nirt:
        nirt_norm = normalize_cost(nirt_cost)

    # Plot with normalized costs
    ax.plot(oracle_unlimited_norm, oracle_unlimited_perf, '-', label="Oracle (Unlimited)", linewidth=2.5, color="red", alpha=0.8)
    ax.plot(oracle_all_norm, oracle_all_perf, '-', label="Oracle (All Limits)", linewidth=2.5, color="brown", alpha=0.8)
    ax.plot(core_norm, core_perf, '-', label="R2-Router (Our Method)", linewidth=2.5, color="black")
    ax.plot(carrot_knn_norm, carrot_knn_perf, '--', label="CARROT-KNN", linewidth=2, color="orange")
    ax.plot(carrot_linear_norm, carrot_linear_perf, '--', label="CARROT-Linear", linewidth=2, color="green")
    if has_mirt:
        ax.plot(mirt_norm, mirt_perf, '-.', label="MIRT", linewidth=2, color="purple")
    if has_nirt:
        ax.plot(nirt_norm, nirt_perf, '-.', label="NIRT", linewidth=2, color="magenta")

    # Normalize and plot individual LLM points
    for i, name in enumerate(llm_names):
        score = llms[name]["true_test_unlimited_score"].mean()
        cost = (llms[name]["true_test_unlimited_count"] * llms[name]["size"]).mean()
        # Normalize this point relative to the overall cost range
        all_costs = np.concatenate([oracle_unlimited_cost, oracle_all_cost, r2_cost])
        cost_min, cost_max = all_costs.min(), all_costs.max()
        if cost_max > cost_min:
            norm_cost = (cost - cost_min) / (cost_max - cost_min)
        else:
            norm_cost = 0.0
        ax.scatter(norm_cost, score, marker="*", s=200, color=colors[i % len(colors)],
                  edgecolor="black", linewidth=1.5, label=f"{name}", zorder=5)

    ax.set_xlabel("Normalized Cost [0,1]", fontsize=12)
    ax.set_ylabel("Average Quality Score", fontsize=12)
    ax.set_title(f"Quality-Cost Tradeoff (Normalized): {title_methods} ({len(llm_names)} LLMs)", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./comparison_results/main/r2_vs_baselines_curves_normalized.png", dpi=150, bbox_inches='tight')
    print("[OK] Saved plot: ./comparison_results/main/r2_vs_baselines_curves_normalized.png")
    plt.close()

    # Find best single LLM's average performance (for correct QNC computation)
    print("\n" + "=" * 80)
    print("FINDING BEST SINGLE LLM FOR QNC TARGET")
    print("=" * 80)

    best_llm_name = None
    best_llm_accuracy = 0.0

    for model_key, model_data in llms.items():
        # Get unlimited token accuracy
        unlimited_accuracy = model_data['true_test_unlimited_score'].mean()
        print(f"{model_data['name']}: {unlimited_accuracy:.4f}")

        if unlimited_accuracy > best_llm_accuracy:
            best_llm_accuracy = unlimited_accuracy
            best_llm_name = model_data['name']

    # Apply target accuracy rate
    target_accuracy = best_llm_accuracy * args.target_accuracy_rate

    print(f"\n✓ Best Single LLM: {best_llm_name} with accuracy {best_llm_accuracy:.4f}")
    print(f"  Target Accuracy Rate: {args.target_accuracy_rate:.2f} ({args.target_accuracy_rate*100:.0f}%)")
    print(f"  QNC Target Accuracy: {target_accuracy:.4f}")

    # Compute global cost range for QNC normalization
    all_costs = np.concatenate([r2_cost, carrot_knn_cost, carrot_linear_cost,
                                oracle_unlimited_cost, oracle_all_cost])
    global_cost_range = (all_costs.min(), all_costs.max())
    print(f"\nGlobal cost range: [{global_cost_range[0]:.2f}, {global_cost_range[1]:.2f}]")

    # Calculate metrics
    core_audc_norm = calculate_audc(r2_cost, core_perf, normalize=True, global_cost_range=global_cost_range)
    core_qnc = calculate_qnc(r2_cost, core_perf, normalize=True,
                             target_perf=best_llm_accuracy, global_cost_range=global_cost_range,
                             target_accuracy_rate=args.target_accuracy_rate)
    core_audc_actual = calculate_audc(r2_cost, core_perf, normalize=False)
    core_peak = core_perf.max()

    carrot_knn_audc_norm = calculate_audc(carrot_knn_cost, carrot_knn_perf, normalize=True, global_cost_range=global_cost_range)
    carrot_knn_qnc = calculate_qnc(carrot_knn_cost, carrot_knn_perf, normalize=True,
                                   target_perf=best_llm_accuracy, global_cost_range=global_cost_range,
                                   target_accuracy_rate=args.target_accuracy_rate)
    carrot_knn_audc_actual = calculate_audc(carrot_knn_cost, carrot_knn_perf, normalize=False)
    carrot_knn_peak = carrot_knn_perf.max()

    carrot_linear_audc_norm = calculate_audc(carrot_linear_cost, carrot_linear_perf, normalize=True, global_cost_range=global_cost_range)
    carrot_linear_qnc = calculate_qnc(carrot_linear_cost, carrot_linear_perf, normalize=True,
                                      target_perf=best_llm_accuracy, global_cost_range=global_cost_range,
                                      target_accuracy_rate=args.target_accuracy_rate)
    carrot_linear_audc_actual = calculate_audc(carrot_linear_cost, carrot_linear_perf, normalize=False)
    carrot_linear_peak = carrot_linear_perf.max()

    # Calculate Oracle metrics
    oracle_unlimited_audc_norm = calculate_audc(oracle_unlimited_cost, oracle_unlimited_perf, normalize=True, global_cost_range=global_cost_range)
    oracle_unlimited_qnc = calculate_qnc(oracle_unlimited_cost, oracle_unlimited_perf, normalize=True,
                                         target_perf=best_llm_accuracy, global_cost_range=global_cost_range,
                                         target_accuracy_rate=args.target_accuracy_rate)
    oracle_unlimited_audc_actual = calculate_audc(oracle_unlimited_cost, oracle_unlimited_perf, normalize=False)
    oracle_unlimited_peak = oracle_unlimited_perf.max()

    oracle_all_audc_norm = calculate_audc(oracle_all_cost, oracle_all_perf, normalize=True, global_cost_range=global_cost_range)
    oracle_all_qnc = calculate_qnc(oracle_all_cost, oracle_all_perf, normalize=True,
                                   target_perf=best_llm_accuracy, global_cost_range=global_cost_range,
                                   target_accuracy_rate=args.target_accuracy_rate)
    oracle_all_audc_actual = calculate_audc(oracle_all_cost, oracle_all_perf, normalize=False)
    oracle_all_peak = oracle_all_perf.max()

    # Calculate IRT metrics if available
    if has_mirt:
        mirt_audc_norm = calculate_audc(mirt_cost, mirt_perf, normalize=True, global_cost_range=global_cost_range)
        mirt_qnc = calculate_qnc(mirt_cost, mirt_perf, normalize=True,
                                target_perf=best_llm_accuracy, global_cost_range=global_cost_range,
                                target_accuracy_rate=args.target_accuracy_rate)
        mirt_audc_actual = calculate_audc(mirt_cost, mirt_perf, normalize=False)
        mirt_peak = mirt_perf.max()

    if has_nirt:
        nirt_audc_norm = calculate_audc(nirt_cost, nirt_perf, normalize=True, global_cost_range=global_cost_range)
        nirt_qnc = calculate_qnc(nirt_cost, nirt_perf, normalize=True,
                                target_perf=best_llm_accuracy, global_cost_range=global_cost_range,
                                target_accuracy_rate=args.target_accuracy_rate)
        nirt_audc_actual = calculate_audc(nirt_cost, nirt_perf, normalize=False)
        nirt_peak = nirt_perf.max()

    # Save metrics to CSV
    metrics_data = [
        {'method': 'R2-Router', 'peak_accuracy': core_peak,
         'AUDC_normalized': core_audc_norm, 'QNC': core_qnc,
         'AUDC_actual': core_audc_actual},
        {'method': 'CARROT-KNN', 'peak_accuracy': carrot_knn_peak,
         'AUDC_normalized': carrot_knn_audc_norm, 'QNC': carrot_knn_qnc,
         'AUDC_actual': carrot_knn_audc_actual},
        {'method': 'CARROT-Linear', 'peak_accuracy': carrot_linear_peak,
         'AUDC_normalized': carrot_linear_audc_norm, 'QNC': carrot_linear_qnc,
         'AUDC_actual': carrot_linear_audc_actual},
        {'method': 'Oracle (Unlimited)', 'peak_accuracy': oracle_unlimited_peak,
         'AUDC_normalized': oracle_unlimited_audc_norm, 'QNC': oracle_unlimited_qnc,
         'AUDC_actual': oracle_unlimited_audc_actual},
        {'method': 'Oracle (All Limits)', 'peak_accuracy': oracle_all_peak,
         'AUDC_normalized': oracle_all_audc_norm, 'QNC': oracle_all_qnc,
         'AUDC_actual': oracle_all_audc_actual},
    ]
    if has_mirt:
        metrics_data.append({'method': 'MIRT', 'peak_accuracy': mirt_peak,
                            'AUDC_normalized': mirt_audc_norm, 'QNC': mirt_qnc,
                            'AUDC_actual': mirt_audc_actual})
    if has_nirt:
        metrics_data.append({'method': 'NIRT', 'peak_accuracy': nirt_peak,
                            'AUDC_normalized': nirt_audc_norm, 'QNC': nirt_qnc,
                            'AUDC_actual': nirt_audc_actual})

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv("./comparison_results/main/r2_vs_baselines_metrics.csv", index=False)
    print("[OK] Saved metrics: ./comparison_results/main/r2_vs_baselines_metrics.csv")

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nR2-Router (Our Method):")
    print(f"  Peak Accuracy: {core_peak:.4f}")
    print(f"  AUDC (normalized): {core_audc_norm:.4f}")
    print(f"  QNC [0,1]: {core_qnc:.4f}")
    print(f"\nCARROT-KNN:")
    print(f"  Peak Accuracy: {carrot_knn_peak:.4f}")
    print(f"  AUDC (normalized): {carrot_knn_audc_norm:.4f}")
    print(f"  QNC [0,1]: {carrot_knn_qnc:.4f}")
    print(f"\nCARROT-Linear:")
    print(f"  Peak Accuracy: {carrot_linear_peak:.4f}")
    print(f"  AUDC (normalized): {carrot_linear_audc_norm:.4f}")
    print(f"  QNC [0,1]: {carrot_linear_qnc:.4f}")

    if has_mirt:
        print(f"\nMIRT:")
        print(f"  Peak Accuracy: {mirt_peak:.4f}")
        print(f"  AUDC (normalized): {mirt_audc_norm:.4f}")
        print(f"  QNC [0,1]: {mirt_qnc:.4f}")

    if has_nirt:
        print(f"\nNIRT:")
        print(f"  Peak Accuracy: {nirt_peak:.4f}")
        print(f"  AUDC (normalized): {nirt_audc_norm:.4f}")
        print(f"  QNC [0,1]: {nirt_qnc:.4f}")

    print(f"\nOracle (Unlimited):")
    print(f"  Peak Accuracy: {oracle_unlimited_peak:.4f}")
    print(f"  AUDC (normalized): {oracle_unlimited_audc_norm:.4f}")
    print(f"  QNC [0,1]: {oracle_unlimited_qnc:.4f}")

    print(f"\nOracle (All Limits):")
    print(f"  Peak Accuracy: {oracle_all_peak:.4f}")
    print(f"  AUDC (normalized): {oracle_all_audc_norm:.4f}")
    print(f"  QNC [0,1]: {oracle_all_qnc:.4f}")

    print("\nMetric Definitions:")
    print("  - Peak Accuracy: Maximum performance achieved [0,1]")
    print("  - AUDC_normalized: Area under cost-performance curve (normalized cost) [higher is better]")
    print("  - QNC: Query-Normalized Cost to match best single LLM [0,1] (lower is better)")

    # Find best baseline
    baseline_audcs_norm = [carrot_knn_audc_norm, carrot_linear_audc_norm]
    baseline_peaks = [carrot_knn_peak, carrot_linear_peak]
    baseline_qncs = [carrot_knn_qnc, carrot_linear_qnc]

    if has_mirt:
        baseline_audcs_norm.append(mirt_audc_norm)
        baseline_peaks.append(mirt_peak)
        baseline_qncs.append(mirt_qnc)
    if has_nirt:
        baseline_audcs_norm.append(nirt_audc_norm)
        baseline_peaks.append(nirt_peak)
        baseline_qncs.append(nirt_qnc)

    best_baseline_audc = max(baseline_audcs_norm)
    best_baseline_peak = max(baseline_peaks)
    best_baseline_qnc = min(baseline_qncs)  # Lower QNC is better

    audc_improvement = ((core_audc_norm - best_baseline_audc) / best_baseline_audc) * 100
    peak_improvement = ((core_peak - best_baseline_peak) / best_baseline_peak) * 100
    qnc_improvement = ((best_baseline_qnc - core_qnc) / best_baseline_qnc) * 100  # Lower is better, so reversed

    print(f"\n" + "=" * 80)
    print(f"R2-Router vs Best Baseline:")
    print(f"  AUDC improvement: {audc_improvement:+.2f}%")
    print(f"  Peak accuracy improvement: {peak_improvement:+.2f}%")
    print(f"  QNC improvement (lower is better): {qnc_improvement:+.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
