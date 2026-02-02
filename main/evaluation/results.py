# %%
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..shared.dataset_manager import DatasetManager
from ..shared.llm_loader import load_llm
from ..r2.predictor import route_scores
from ..baselines.carrot.baselines_carrot import CarrotKNNBaseline as CarrotBaseline, CarrotLinearBaseline as LinearCarrotBaseline, route_baseline
from ..baselines.irt.baselines_irt import IRTBaseline, NIRTBaseline

colors = plt.cm.tab10.colors

# ============================================================================
# PREDICTOR CONFIGURATION
# ============================================================================
# Choose which predictor to use by uncommenting ONE of the following:

# Option 1: PyTorch-based neural network predictor (default)
# from predictor import TokenPerformancePredictor as PredictorClass

# Option 2: Sklearn-based linear regression predictor
from ..r2.predictor_sklearn import TokenPerformancePredictor as PredictorClass

# ============================================================================

# %% Initialize centralized dataset manager
print("=" * 60)
print("INITIALIZING DATASET MANAGER")
print("=" * 60)
dataset_manager = DatasetManager(
    embeddings_path="data/prompt_embeddings.pkl",
    train_ratio=0.8,
    seed=42
)

# Get shared embeddings and split indices
embeddings = dataset_manager.get_embeddings()
train_idx, test_idx = dataset_manager.get_split_indices()

# Token limit configuration (shared across all LLMs)
token_limits_score = [
    '10_score', '20_score', '30_score', '40_score', '50_score', '80_score', '100_score', '150_score',
    '200_score', '300_score', '500_score', '800_score', '1200_score', '2000_score', '4000_score', 'unlimited_score'
]
token_limits_count = [
    '10_count', '20_count', '30_count', '40_count', '50_count', '80_count', '100_count', '150_count',
    '200_count', '300_count', '500_count', '800_count', '1200_count', '2000_count', '4000_count', 'unlimited_count'
]

# %% Load LLMs with centralized split
print("\n" + "=" * 60)
print("LOADING LLMs")
print("=" * 60)

llms = {
    # "GLM_4_6": load_llm(
    #     name="GLM-4.6",
    #     size=1.75,
    #     score_df_path="data/GLM-4.6.csv",
    #     load_dir="./checkpoints/GLM-4.6_multi",
    #     embeddings=embeddings,
    #     train_idx=train_idx,
    #     test_idx=test_idx,
    #     token_limits_score=token_limits_score,
    #     token_limits_count=token_limits_count,
    #     predictor_class=PredictorClass
    # ),
    "GLM_4_5_Air": load_llm(
        name="GLM-4.5-Air",
        size=0.85,
        score_df_path="data/GLM-4.5-Air.csv",
        load_dir="./checkpoints/GLM-4.5-Air_multi",
        embeddings=embeddings,
        train_idx=train_idx,
        test_idx=test_idx,
        token_limits_score=token_limits_score,
        token_limits_count=token_limits_count,
        predictor_class=PredictorClass
    ),
    "gemma3_4B": load_llm(
        name="gemma3-4B",
        size=0.068,
        score_df_path="data/gemma-3-4b-it.csv",
        load_dir="./checkpoints/gemma3-4B-Instruct_multi",
        embeddings=embeddings,
        train_idx=train_idx,
        test_idx=test_idx,
        token_limits_score=token_limits_score,
        token_limits_count=token_limits_count,
        predictor_class=PredictorClass
    ),
    "Llama3_3B": load_llm(
        name="Llama3-3B",
        size=0.06,
        score_df_path="data/Llama-3.2-3B-Instruct.csv",
        load_dir="./checkpoints/Llama-3.2-3B-Instruct_multi",
        embeddings=embeddings,
        train_idx=train_idx,
        test_idx=test_idx,
        token_limits_score=token_limits_score,
        token_limits_count=token_limits_count,
        predictor_class=PredictorClass
    ),
    "Llama3_1_70B": load_llm(
        name="Llama3-1-70B",
        size=0.40,
        score_df_path="data/Llama-3.1-70B-Instruct.csv",
        load_dir="./checkpoints/Llama-3.1-70B-Instruct_multi",
        embeddings=embeddings,
        train_idx=train_idx,
        test_idx=test_idx,
        token_limits_score=token_limits_score,
        token_limits_count=token_limits_count,
        predictor_class=PredictorClass
    ),
    # "Qwen3_80B": load_llm(
    #     name="Qwen3_80B",
    #     size=0.287,
    #     score_df_path="data/Qwen3-Next-80B-A3B-Instruct.csv",
    #     load_dir="./checkpoints/Qwen3-Next-80B-A3B-Instruct_multi",
    #     embeddings=embeddings,
    #     train_idx=train_idx,
    #     test_idx=test_idx,
    #     token_limits_score=token_limits_score,
    #     token_limits_count=token_limits_count
    # ),
    "Qwen3_0_6B": load_llm(
        name="Qwen3-0.6B",
        size=0.0173,
        score_df_path="data/Qwen3-0.6B.csv",
        load_dir="./checkpoints/Qwen3-0.6B_multi",
        embeddings=embeddings,
        train_idx=train_idx,
        test_idx=test_idx,
        token_limits_score=token_limits_score,
        token_limits_count=token_limits_count,
        predictor_class=PredictorClass
    ),
    # "Qwen2_5_Math_1_5B": load_llm(
    #     name="Qwen2.5-Math-1.5B",
    #     size=0.05,
    #     score_df_path="data/Qwen2.5-Math-1.5B-Instruct.csv",
    #     load_dir="./checkpoints/Qwen2.5-Math-1.5B-Instruct_multi",
    #     embeddings=embeddings,
    #     train_idx=train_idx,
    #     test_idx=test_idx,
    #     token_limits_score=token_limits_score,
    #     token_limits_count=token_limits_count,
    #     predictor_class=PredictorClass
    # ),
    "Qwen2_5_Math_7B": load_llm(
        name="Qwen2.5-Math-7B",
        size=0.35,
        score_df_path="data/Qwen2.5-Math-7B-Instruct.csv",
        load_dir="./checkpoints/Qwen2.5-Math-7B-Instruct_multi",
        embeddings=embeddings,
        train_idx=train_idx,
        test_idx=test_idx,
        token_limits_score=token_limits_score,
        token_limits_count=token_limits_count,
        predictor_class=PredictorClass
    ),
    # "Qwen3_235B": load_llm(
    #     name="Qwen3-235B",
    #     size=5.25,
    #     score_df_path="data/Qwen3-235B-A22B-Instruct-2507.csv",
    #     load_dir="./checkpoints/Qwen3-235B-A22B-Instruct-2507_multi",
    #     embeddings=embeddings,
    #     train_idx=train_idx,
    #     test_idx=test_idx,
    #     token_limits_score=token_limits_score,
    #     token_limits_count=token_limits_count,
    #     predictor_class=PredictorClass
    # )
}

print(f"\nLoaded {len(llms)} LLMs successfully!")
# %%
# Lambda range for cost-performance tradeoff
# Formula: score = (1-λ)*quality - λ*cost
# λ ∈ [0,1]: λ=0 pure quality, λ=1 pure cost minimization
lamb_range = np.unique(np.concatenate([
    np.linspace(0, 0.2, 20),
    np.linspace(0.2, 1.0, 50)
]))

routing_cost, routing_perf = route_scores(
    llms,
    lamb_range
)

# %% Baseline (Carrot)
carrot_baseline = CarrotBaseline(llms).fit()
Y_hat_score_test, Y_hat_count_test = carrot_baseline.predict()

llm_names = list(llms.keys())
sizes_vec = np.array([llms[name]["size"] for name in llm_names])[None, :]

# Use the ground-truth matrices built by CarrotBaseline
routing_cost_carrot, routing_perf_carrot = route_baseline(
    Y_hat_score_test,
    Y_hat_count_test,
    carrot_baseline.Y_score_test_true,
    carrot_baseline.Y_count_test_true,
    lamb_range,
    sizes_vec
)

# %% Linear Carrot Baseline
linear_carrot_baseline = LinearCarrotBaseline(llms).fit()
Y_hat_score_test_linear, Y_hat_count_test_linear = linear_carrot_baseline.predict()

# Calculate routing performance for Linear Carrot
routing_cost_linear, routing_perf_linear = route_baseline(
    Y_hat_score_test_linear,
    Y_hat_count_test_linear,
    linear_carrot_baseline.Y_score_test_true,
    linear_carrot_baseline.Y_count_test_true,
    lamb_range,
    sizes_vec
)

# %% Oracle Curve
def calculate_oracle_curve(llms, lamb_range):
    """
    Calculate Oracle curve - always selects the most accurate model at the lowest possible cost.
    This represents the theoretical performance ceiling.
    """
    llm_names = list(llms.keys())
    sizes_vec = np.array([llms[name]["size"] for name in llm_names])[None, :]
    
    # Get ground-truth performance and cost matrices
    Y_score_true = np.stack([llms[name]["true_test_unlimited_score"] for name in llm_names], axis=1)
    Y_count_true = np.stack([llms[name]["true_test_unlimited_count"] for name in llm_names], axis=1)
    
    # Calculate true cost matrix (count * size)
    true_cost_mat = Y_count_true * sizes_vec
    
    oracle_cost, oracle_perf = [], []
    n_queries = Y_score_true.shape[0]
    
    for lam in lamb_range:
        # Oracle uses perfect knowledge of true performance and cost
        oracle_risk = (1 - lam) * Y_score_true - lam * true_cost_mat
        chosen = oracle_risk.argmax(axis=1)
        
        chosen_perf = Y_score_true[np.arange(n_queries), chosen]
        chosen_cost = true_cost_mat[np.arange(n_queries), chosen]
        
        oracle_perf.append(chosen_perf.mean())
        oracle_cost.append(chosen_cost.mean())
    
    return np.array(oracle_cost), np.array(oracle_perf)

routing_cost_oracle, routing_perf_oracle = calculate_oracle_curve(llms, lamb_range)

# Oracle considering all LLMs and all output token limits
def calculate_oracle_curve_with_limits(llms, lamb_range):
    llm_names = list(llms.keys())
    # Build matrices across all (model, token_limit) choices
    scores_blocks = []  # each (n_queries, n_limits_plus_unlimited)
    costs_blocks = []   # each (n_queries, n_limits_plus_unlimited)
    for name in llm_names:
        s_limited = llms[name]["true_test_score"]  # (n_queries, n_limits)
        c_limited = llms[name]["true_test_count"] * llms[name]["size"]  # (n_queries, n_limits)
        # Append unlimited as an additional column
        s_unlim = llms[name]["true_test_unlimited_score"][:, None]  # (n_queries, 1)
        c_unlim = (llms[name]["true_test_unlimited_count"] * llms[name]["size"])[:, None]  # (n_queries, 1)
        s = np.concatenate([s_limited, s_unlim], axis=1)
        c = np.concatenate([c_limited, c_unlim], axis=1)
        scores_blocks.append(s)
        costs_blocks.append(c)
    Y_score_all = np.concatenate(scores_blocks, axis=1)  # (n_queries, M*L)
    Y_cost_all  = np.concatenate(costs_blocks, axis=1)   # (n_queries, M*L)

    oracle_cost, oracle_perf = [], []
    n_queries = Y_score_all.shape[0]
    for lam in lamb_range:
        oracle_risk = (1 - lam) * Y_score_all - lam * Y_cost_all
        chosen = oracle_risk.argmax(axis=1)
        chosen_perf = Y_score_all[np.arange(n_queries), chosen]
        chosen_cost = Y_cost_all[np.arange(n_queries), chosen]
        oracle_perf.append(chosen_perf.mean())
        oracle_cost.append(chosen_cost.mean())
    return np.array(oracle_cost), np.array(oracle_perf)

routing_cost_oracle_limited, routing_perf_oracle_limited = calculate_oracle_curve_with_limits(llms, lamb_range)

# %% IRT with latent_dim=256

llm_texts = {name: f"LLM {name}, size {llms[name]['size']}." for name in llm_names}
irt_256 = IRTBaseline(llms, llm_texts=llm_texts, latent_dim=256)
irt_256.fit(epochs=50)
Y_hat_score_test_irt_256 = irt_256.predict()

fixed_tokens = 1000
Y_hat_count_test_irt_256 = np.full_like(Y_hat_score_test_irt_256, fixed_tokens, dtype=np.float32)

routing_cost_irt_256, routing_perf_irt_256 = route_baseline(
    Y_hat_score_test_irt_256,
    Y_hat_count_test_irt_256,
    carrot_baseline.Y_score_test_true,
    carrot_baseline.Y_count_test_true,
    lamb_range,
    sizes_vec
)

# %% IRT with latent_dim=8
irt_32 = IRTBaseline(llms, llm_texts=llm_texts, latent_dim=2)
irt_32.fit(epochs=50)
Y_hat_score_test_irt_32 = irt_32.predict()

Y_hat_count_test_irt_32 = np.full_like(Y_hat_score_test_irt_32, fixed_tokens, dtype=np.float32)

routing_cost_irt_32, routing_perf_irt_32 = route_baseline(
    Y_hat_score_test_irt_32,
    Y_hat_count_test_irt_32,
    carrot_baseline.Y_score_test_true,
    carrot_baseline.Y_count_test_true,
    lamb_range,
    sizes_vec
)

# %% NIRT with latent_dim=256
nirt_256 = NIRTBaseline(llms, llm_texts=llm_texts, latent_dim=256)
nirt_256.fit(epochs=50)
Y_hat_score_test_nirt_256 = nirt_256.predict()

Y_hat_count_test_nirt_256 = np.full_like(Y_hat_score_test_nirt_256, fixed_tokens, dtype=np.float32)

routing_cost_nirt_256, routing_perf_nirt_256 = route_baseline(
    Y_hat_score_test_nirt_256,
    Y_hat_count_test_nirt_256,
    carrot_baseline.Y_score_test_true,
    carrot_baseline.Y_count_test_true,
    lamb_range,
    sizes_vec
)

# %% NIRT with latent_dim=2
nirt_2 = NIRTBaseline(llms, llm_texts=llm_texts, latent_dim=2)
nirt_2.fit(epochs=50)
Y_hat_score_test_nirt_2 = nirt_2.predict()

Y_hat_count_test_nirt_2 = np.full_like(Y_hat_score_test_nirt_2, fixed_tokens, dtype=np.float32)

routing_cost_nirt_2, routing_perf_nirt_2 = route_baseline(
    Y_hat_score_test_nirt_2,
    Y_hat_count_test_nirt_2,
    carrot_baseline.Y_score_test_true,
    carrot_baseline.Y_count_test_true,
    lamb_range,
    sizes_vec
)

# %% Save LLM token statistics to CSV
token_limits = ['10', '20', '30', '40', '50', '80', '100', '150', '200', '300', '500', '800', '1200', '2000', '4000', 'unlimited']
data = []

for name in llm_names:
    # Limited token data
    for i, limit in enumerate(token_limits[:-1]):  # Exclude 'unlimited'
        avg_score = llms[name]["true_test_score"][:, i].mean()
        avg_tokens = llms[name]["true_test_count"][:, i].mean()
        data.append([name, llms[name]["size"], limit, avg_score, avg_tokens])
    
    # Unlimited data
    avg_score = llms[name]["true_test_unlimited_score"].mean()
    avg_tokens = llms[name]["true_test_unlimited_count"].mean()
    data.append([name, llms[name]["size"], 'unlimited', avg_score, avg_tokens])

df = pd.DataFrame(data, columns=['llm_name', 'model_size', 'token_limit', 'avg_score', 'avg_actual_tokens'])
df.to_csv('llm_token_statistics.csv', index=False)
print(f"Saved {len(data)} records to llm_token_statistics.csv")

# %% Save Our and Carrot curves to CSV (minimal)
curve_long = pd.DataFrame({
    'method': np.concatenate([
        np.full_like(lamb_range, 'our', dtype=object),
        np.full_like(lamb_range, 'carrot', dtype=object),
        np.full_like(lamb_range, 'oracle_unlimited', dtype=object),
        np.full_like(lamb_range, 'oracle_all_limits', dtype=object),
        np.full_like(lamb_range, 'irt_256', dtype=object),
        np.full_like(lamb_range, 'irt_32', dtype=object),
        np.full_like(lamb_range, 'nirt_256', dtype=object),
        np.full_like(lamb_range, 'nirt_2', dtype=object)
    ]),
    'lambda': np.concatenate([lamb_range, lamb_range, lamb_range, lamb_range, lamb_range, lamb_range, lamb_range, lamb_range]),
    'cost':   np.concatenate([routing_cost, routing_cost_carrot, routing_cost_oracle, routing_cost_oracle_limited, routing_cost_irt_256, routing_cost_irt_32, routing_cost_nirt_256, routing_cost_nirt_2]),
    'perf':   np.concatenate([routing_perf, routing_perf_carrot, routing_perf_oracle, routing_perf_oracle_limited, routing_perf_irt_256, routing_perf_irt_32, routing_perf_nirt_256, routing_perf_nirt_2])
})
curve_long.to_csv('curve_points.csv', index=False)
print(f"Saved {len(curve_long)} curve points to curve_points.csv")

# %%
plt.figure(figsize=(3.5, 3))
plt.plot(routing_cost_oracle, routing_perf_oracle, '-', label="Oracle (Unlimited)", linewidth=2, color="red", alpha=0.8)
plt.plot(routing_cost_oracle_limited, routing_perf_oracle_limited, '-', label="Oracle (All limits)", linewidth=2, color="brown", alpha=0.8)
plt.plot(routing_cost, routing_perf, '-', label="Our", linewidth=2, color="black")
plt.plot(routing_cost_carrot, routing_perf_carrot, '.-', label="Carrot", color="orange", linewidth=2)
plt.plot(routing_cost_linear, routing_perf_linear, '.-', label="Linear Carrot", color="green", linewidth=2)
plt.plot(routing_cost_irt_256, routing_perf_irt_256, '--', label="MIRT", color="purple", linewidth=2)
plt.plot(routing_cost_nirt_256, routing_perf_nirt_256, '--', label="NIRT", color="magenta", linewidth=2)
# plt.plot(routing_cost_irt_32, routing_perf_irt_32, '--', label="IRT (dim=32)", color="brown", linewidth=2)
# plt.plot(routing_cost_nirt_2, routing_perf_nirt_2, '--', label="NIRT (dim=2)", color="pink", linewidth=2)

for j, name in enumerate(llm_names):
    # Use unlimited_score data from load_llm
    score = llms[name]["true_test_unlimited_score"].mean()
    cost = (llms[name]["true_test_unlimited_count"] * llms[name]["size"]).mean()
    plt.scatter(cost, score, marker="*", s=160, color=colors[j % len(colors)], edgecolor="black", label=f"{name}")

    # Add per-token-limit scatter points (mean performance vs mean cost)
    true_test_score = llms[name]["true_test_score"]
    true_test_count = llms[name]["true_test_count"]
    for tok_idx in range(true_test_score.shape[1]):  # Use actual number of token limits from data
        sc = true_test_score[:, tok_idx].mean()
        cst = (true_test_count[:, tok_idx] * llms[name]["size"]).mean()
        plt.scatter(cst, sc, marker="o", s=40, color=colors[j % len(colors)], alpha=0.55)

plt.xlabel("Cost")
plt.ylabel("Average Performance")
plt.legend()
plt.grid(True)
plt.show()

# %% Plot individual LLM Oracle curves
def calculate_single_llm_oracle(llm_name, llm_data, lamb_range):
    """
    Calculate Oracle curve for a single LLM
    Returns: unlimited point and all_limits curve
    """
    # Unlimited point (single point)
    unlimited_score = llm_data["true_test_unlimited_score"]
    unlimited_count = llm_data["true_test_unlimited_count"]
    unlimited_cost = unlimited_count * llm_data["size"]
    unlimited_perf = unlimited_score.mean()
    unlimited_cost_mean = unlimited_cost.mean()
    
    # All limits curve (including limited tokens)
    scores_all = llm_data["true_test_score"]  # (n_queries, n_limits)
    costs_all = llm_data["true_test_count"] * llm_data["size"]  # (n_queries, n_limits)
    
    # Append unlimited as additional column
    scores_all = np.concatenate([scores_all, unlimited_score[:, None]], axis=1)
    costs_all = np.concatenate([costs_all, unlimited_cost[:, None]], axis=1)
    
    n_queries = scores_all.shape[0]
    all_limits_cost, all_limits_perf = [], []
    
    for lam in lamb_range:
        oracle_risk = (1 - lam) * scores_all - lam * costs_all
        chosen = oracle_risk.argmax(axis=1)
        
        chosen_perf = scores_all[np.arange(n_queries), chosen]
        chosen_cost = costs_all[np.arange(n_queries), chosen]
        
        all_limits_perf.append(chosen_perf.mean())
        all_limits_cost.append(chosen_cost.mean())
    
    # Calculate Oracle point: for each query, find max performance, then choose min cost among those
    oracle_perf_list, oracle_cost_list = [], []
    for query_idx in range(n_queries):
        query_scores = scores_all[query_idx, :]
        query_costs = costs_all[query_idx, :]
        
        # Find maximum performance
        max_perf = query_scores.max()
        
        # Among options with maximum performance, choose the one with minimum cost
        best_options = query_scores == max_perf
        best_costs = query_costs[best_options]
        min_cost = best_costs.min()
        
        oracle_perf_list.append(max_perf)
        oracle_cost_list.append(min_cost)
    
    oracle_perf_mean = np.mean(oracle_perf_list)
    oracle_cost_mean = np.mean(oracle_cost_list)
    
    return {
        'unlimited': (unlimited_cost_mean, unlimited_perf),
        'all_limits': (np.array(all_limits_cost), np.array(all_limits_perf)),
        'oracle': (oracle_cost_mean, oracle_perf_mean)
    }

# Calculate individual Oracle for each LLM
individual_oracles = {}
for llm_name, llm_data in llms.items():
    individual_oracles[llm_name] = calculate_single_llm_oracle(llm_name, llm_data, lamb_range)

# %% Calculate Pool Oracle point
def calculate_pool_oracle_point(llms):
    """
    Calculate Pool Oracle point: for each query, find max performance across all (LLM, token_limit) combinations,
    then choose min cost among those with max performance.
    """
    llm_names = list(llms.keys())
    # Build matrices across all (model, token_limit) choices
    scores_blocks = []
    costs_blocks = []
    for name in llm_names:
        s_limited = llms[name]["true_test_score"]  # (n_queries, n_limits)
        c_limited = llms[name]["true_test_count"] * llms[name]["size"]
        # Append unlimited as additional column
        s_unlim = llms[name]["true_test_unlimited_score"][:, None]
        c_unlim = (llms[name]["true_test_unlimited_count"] * llms[name]["size"])[:, None]
        s = np.concatenate([s_limited, s_unlim], axis=1)
        c = np.concatenate([c_limited, c_unlim], axis=1)
        scores_blocks.append(s)
        costs_blocks.append(c)
    Y_score_all = np.concatenate(scores_blocks, axis=1)  # (n_queries, M*L)
    Y_cost_all = np.concatenate(costs_blocks, axis=1)    # (n_queries, M*L)
    
    n_queries = Y_score_all.shape[0]
    oracle_perf_list, oracle_cost_list = [], []
    
    for query_idx in range(n_queries):
        query_scores = Y_score_all[query_idx, :]
        query_costs = Y_cost_all[query_idx, :]
        
        # Find maximum performance
        max_perf = query_scores.max()
        
        # Among options with maximum performance, choose the one with minimum cost
        best_options = query_scores == max_perf
        best_costs = query_costs[best_options]
        min_cost = best_costs.min()
        
        oracle_perf_list.append(max_perf)
        oracle_cost_list.append(min_cost)
    
    oracle_perf_mean = np.mean(oracle_perf_list)
    oracle_cost_mean = np.mean(oracle_cost_list)
    
    return oracle_cost_mean, oracle_perf_mean

pool_oracle_cost, pool_oracle_perf = calculate_pool_oracle_point(llms)

# Plot
plt.figure(figsize=(5, 4))
for j, (llm_name, oracle_data) in enumerate(individual_oracles.items()):
    color = colors[j % len(colors)]
    
    # Plot unlimited point (single point)
    unl_cost, unl_perf = oracle_data['unlimited']
    plt.scatter(unl_cost, unl_perf, marker="*", s=200, color=color, 
                edgecolor="black", linewidth=1.5, label=f"{llm_name} (unlimited)", zorder=5)
    
    # Plot all limits curve
    all_limits_cost, all_limits_perf = oracle_data['all_limits']
    plt.plot(all_limits_cost, all_limits_perf, '-', 
             label=f"{llm_name} (all limits)", linewidth=2, color=color, alpha=0.7)
    
    # Plot Oracle point (max performance, min cost among those)
    oracle_cost, oracle_perf = oracle_data['oracle']
    plt.scatter(oracle_cost, oracle_perf, marker="o", s=150, color=color, 
                edgecolor="black", linewidth=1.5, label=f"{llm_name} (oracle)", zorder=5)

# Plot Pool Oracle curves
plt.plot(routing_cost_oracle, routing_perf_oracle, '-', 
         label="Pool Oracle (unlimited)", linewidth=3, color="black", linestyle='--', alpha=0.9)

plt.plot(routing_cost_oracle_limited, routing_perf_oracle_limited, '-', 
         label="Pool Oracle (all limits)", linewidth=3, color="red", linestyle='--', alpha=0.9)

# Plot Pool Oracle point
plt.scatter(pool_oracle_cost, pool_oracle_perf, marker="D", s=200, color="orange", 
            edgecolor="black", linewidth=2, label="Pool Oracle (point)", zorder=6)

plt.xlabel("Cost")
plt.ylabel("Average Performance")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Save individual LLM oracle points to CSV
llm_oracle_data = []

for llm_name, oracle_data in individual_oracles.items():
    # Save unlimited point
    unl_cost, unl_perf = oracle_data['unlimited']
    llm_oracle_data.append({
        'llm_name': llm_name,
        'type': 'unlimited',
        'cost': unl_cost,
        'performance': unl_perf,
        'lambda': lamb_range[-1]  # Use the last lambda value
    })
    
    # Save final point from all_limits curve (the point with highest cost, i.e., largest lambda)
    all_limits_cost, all_limits_perf = oracle_data['all_limits']
    # The last point in the curve corresponds to the highest lambda value
    final_idx = len(all_limits_cost) - 1
    llm_oracle_data.append({
        'llm_name': llm_name,
        'type': 'final_all_limits',
        'cost': all_limits_cost[final_idx],
        'performance': all_limits_perf[final_idx],
        'lambda': lamb_range[final_idx]
    })
    
    # Save the Oracle point (for each query: max performance, then min cost among those)
    oracle_cost, oracle_perf = oracle_data['oracle']
    llm_oracle_data.append({
        'llm_name': llm_name,
        'type': 'oracle',
        'cost': oracle_cost,
        'performance': oracle_perf,
        'lambda': None  # Oracle point doesn't correspond to a specific lambda
    })

df_llm_oracle = pd.DataFrame(llm_oracle_data)
df_llm_oracle.to_csv('individual_llm_oracle_points.csv', index=False)
print(f"\nSaved {len(llm_oracle_data)} individual LLM oracle points to individual_llm_oracle_points.csv")

# %% Save full individual LLM Oracle curves to CSV
individual_llm_curve_data = []

for llm_name, oracle_data in individual_oracles.items():
    # Save the full all_limits curve for each LLM
    all_limits_cost, all_limits_perf = oracle_data['all_limits']
    for i, lam in enumerate(lamb_range):
        individual_llm_curve_data.append({
            'llm_name': llm_name,
            'type': 'all_limits_curve',
            'lambda': lam,
            'cost': all_limits_cost[i],
            'performance': all_limits_perf[i]
        })

df_individual_curves = pd.DataFrame(individual_llm_curve_data)
df_individual_curves.to_csv('individual_llm_oracle_curves.csv', index=False)
print(f"\nSaved {len(individual_llm_curve_data)} individual LLM oracle curve points to individual_llm_oracle_curves.csv")

# %% Save LLM Pool Oracle curves to CSV
pool_oracle_data = []

# Save unlimited oracle curve
for i, lam in enumerate(lamb_range):
    pool_oracle_data.append({
        'pool_type': 'pool_unlimited',
        'lambda': lam,
        'cost': routing_cost_oracle[i],
        'performance': routing_perf_oracle[i]
    })

# Save limited oracle curve
for i, lam in enumerate(lamb_range):
    pool_oracle_data.append({
        'pool_type': 'pool_all_limits',
        'lambda': lam,
        'cost': routing_cost_oracle_limited[i],
        'performance': routing_perf_oracle_limited[i]
    })

df_pool_oracle = pd.DataFrame(pool_oracle_data)
df_pool_oracle.to_csv('pool_oracle_curves.csv', index=False)
print(f"\nSaved {len(pool_oracle_data)} pool oracle curve points to pool_oracle_curves.csv")

# %% Save Pool Oracle point to CSV
pool_oracle_point_data = [{
    'pool_type': 'pool_oracle_point',
    'lambda': None,
    'cost': pool_oracle_cost,
    'performance': pool_oracle_perf
}]

df_pool_oracle_point = pd.DataFrame(pool_oracle_point_data)
df_pool_oracle_point.to_csv('pool_oracle_point.csv', index=False)
print(f"\nSaved pool oracle point to pool_oracle_point.csv")

# %%
# Metrics calculation functions
def calculate_audc(cost_curve, perf_curve):
    """
    Calculate Area Under the Deferral Curve (AUDC)
    AUDC measures the area under the cost-performance curve
    Cost is normalized to [0, 1] range
    """
    # Sort by cost to ensure proper curve
    sorted_indices = np.argsort(cost_curve)
    sorted_cost = cost_curve[sorted_indices]
    sorted_perf = perf_curve[sorted_indices]
    
    # Normalize cost to [0, 1] range
    if sorted_cost.max() > sorted_cost.min():
        normalized_cost = (sorted_cost - sorted_cost.min()) / (sorted_cost.max() - sorted_cost.min())
    else:
        normalized_cost = np.zeros_like(sorted_cost)
    
    # Calculate area under the curve using trapezoidal rule
    audc = np.trapezoid(sorted_perf, normalized_cost)
    return audc

def calculate_peak_accuracy(perf_curve):
    """
    Calculate Peak accuracy - the maximum performance achieved
    """
    return np.max(perf_curve)

def calculate_qnc(cost_curve, perf_curve):
    """
    Calculate Query-Normalized Cost (QNC)
    QNC is the normalized cost at which performance reaches 90% of peak performance
    Cost is normalized to [0, 1] range
    """
    peak_perf = np.max(perf_curve)
    target_perf = 0.9 * peak_perf
    
    # Find the cost at which performance reaches 90% of peak
    # Interpolate if needed
    sorted_indices = np.argsort(cost_curve)
    sorted_cost = cost_curve[sorted_indices]
    sorted_perf = perf_curve[sorted_indices]
    
    # Normalize cost to [0, 1] range
    if sorted_cost.max() > sorted_cost.min():
        normalized_cost = (sorted_cost - sorted_cost.min()) / (sorted_cost.max() - sorted_cost.min())
    else:
        normalized_cost = np.zeros_like(sorted_cost)
    
    # Find the first point where performance >= target_perf
    valid_indices = sorted_perf >= target_perf
    if np.any(valid_indices):
        first_valid_idx = np.where(valid_indices)[0][0]
        if first_valid_idx == 0:
            # If first point is already >= target, use it
            qnc = normalized_cost[0]
        else:
            # Interpolate between the point just before and at target
            prev_idx = first_valid_idx - 1
            # Linear interpolation
            x1, y1 = normalized_cost[prev_idx], sorted_perf[prev_idx]
            x2, y2 = normalized_cost[first_valid_idx], sorted_perf[first_valid_idx]
            
            # Interpolate x where y = target_perf
            if y2 != y1:  # Avoid division by zero
                qnc = x1 + (target_perf - y1) * (x2 - x1) / (y2 - y1)
            else:
                qnc = x1
    else:
        # If no point reaches 90% of peak, use the cost at maximum performance
        max_perf_idx = np.argmax(sorted_perf)
        qnc = normalized_cost[max_perf_idx]
    
    return qnc

# Calculate metrics for all methods
print("=" * 60)
print("METRICS COMPARISON")
print("=" * 60)

# Oracle metrics (Unlimited)
oracle_audc = calculate_audc(routing_cost_oracle, routing_perf_oracle)
oracle_peak = calculate_peak_accuracy(routing_perf_oracle)
oracle_qnc = calculate_qnc(routing_cost_oracle, routing_perf_oracle)

print(f"\nOracle (Unlimited Token Limits):")
print(f"  AUDC (Area Under Deferral Curve): {oracle_audc:.4f}")
print(f"  Peak Accuracy: {oracle_peak:.4f}")
print(f"  QNC (Query-Normalized Cost): {oracle_qnc:.4f}")

# Oracle metrics (All Limits)
oracle_limited_audc = calculate_audc(routing_cost_oracle_limited, routing_perf_oracle_limited)
oracle_limited_peak = calculate_peak_accuracy(routing_perf_oracle_limited)
oracle_limited_qnc = calculate_qnc(routing_cost_oracle_limited, routing_perf_oracle_limited)

print(f"\nOracle (All Token Limits):")
print(f"  AUDC (Area Under Deferral Curve): {oracle_limited_audc:.4f}")
print(f"  Peak Accuracy: {oracle_limited_peak:.4f}")
print(f"  QNC (Query-Normalized Cost): {oracle_limited_qnc:.4f}")

# Router metrics
router_audc = calculate_audc(routing_cost, routing_perf)
router_peak = calculate_peak_accuracy(routing_perf)
router_qnc = calculate_qnc(routing_cost, routing_perf)

print(f"\nRouter Method:")
print(f"  AUDC (Area Under Deferral Curve): {router_audc:.4f}")
print(f"  Peak Accuracy: {router_peak:.4f}")
print(f"  QNC (Query-Normalized Cost): {router_qnc:.4f}")

# Carrot baseline metrics
carrot_audc = calculate_audc(routing_cost_carrot, routing_perf_carrot)
carrot_peak = calculate_peak_accuracy(routing_perf_carrot)
carrot_qnc = calculate_qnc(routing_cost_carrot, routing_perf_carrot)

print(f"\nCarrot Baseline:")
print(f"  AUDC (Area Under Deferral Curve): {carrot_audc:.4f}")
print(f"  Peak Accuracy: {carrot_peak:.4f}")
print(f"  QNC (Query-Normalized Cost): {carrot_qnc:.4f}")

# IRT baseline metrics (dim=256)
irt_256_audc = calculate_audc(routing_cost_irt_256, routing_perf_irt_256)
irt_256_peak = calculate_peak_accuracy(routing_perf_irt_256)
irt_256_qnc = calculate_qnc(routing_cost_irt_256, routing_perf_irt_256)

print(f"\nIRT Baseline (dim=256):")
print(f"  AUDC (Area Under Deferral Curve): {irt_256_audc:.4f}")
print(f"  Peak Accuracy: {irt_256_peak:.4f}")
print(f"  QNC (Query-Normalized Cost): {irt_256_qnc:.4f}")

# IRT baseline metrics (dim=32)
irt_32_audc = calculate_audc(routing_cost_irt_32, routing_perf_irt_32)
irt_32_peak = calculate_peak_accuracy(routing_perf_irt_32)
irt_32_qnc = calculate_qnc(routing_cost_irt_32, routing_perf_irt_32)

print(f"\nIRT Baseline (dim=32):")
print(f"  AUDC (Area Under Deferral Curve): {irt_32_audc:.4f}")
print(f"  Peak Accuracy: {irt_32_peak:.4f}")
print(f"  QNC (Query-Normalized Cost): {irt_32_qnc:.4f}")

# NIRT baseline metrics (dim=256)
nirt_256_audc = calculate_audc(routing_cost_nirt_256, routing_perf_nirt_256)
nirt_256_peak = calculate_peak_accuracy(routing_perf_nirt_256)
nirt_256_qnc = calculate_qnc(routing_cost_nirt_256, routing_perf_nirt_256)

print(f"\nNIRT Baseline (dim=256):")
print(f"  AUDC (Area Under Deferral Curve): {nirt_256_audc:.4f}")
print(f"  Peak Accuracy: {nirt_256_peak:.4f}")
print(f"  QNC (Query-Normalized Cost): {nirt_256_qnc:.4f}")

# NIRT baseline metrics (dim=2)
nirt_2_audc = calculate_audc(routing_cost_nirt_2, routing_perf_nirt_2)
nirt_2_peak = calculate_peak_accuracy(routing_perf_nirt_2)
nirt_2_qnc = calculate_qnc(routing_cost_nirt_2, routing_perf_nirt_2)

print(f"\nNIRT Baseline (dim=2):")
print(f"  AUDC (Area Under Deferral Curve): {nirt_2_audc:.4f}")
print(f"  Peak Accuracy: {nirt_2_peak:.4f}")
print(f"  QNC (Query-Normalized Cost): {nirt_2_qnc:.4f}")

# Linear Carrot baseline metrics
linear_audc = calculate_audc(routing_cost_linear, routing_perf_linear)
linear_peak = calculate_peak_accuracy(routing_perf_linear)
linear_qnc = calculate_qnc(routing_cost_linear, routing_perf_linear)

print(f"\nLinear Carrot Baseline:")
print(f"  AUDC (Area Under Deferral Curve): {linear_audc:.4f}")
print(f"  Peak Accuracy: {linear_peak:.4f}")
print(f"  QNC (Query-Normalized Cost): {linear_qnc:.4f}")

# Summary table
print(f"\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Method':<20} {'AUDC':<10} {'Peak Acc':<10} {'QNC':<10}")
print("-" * 70)
print(f"{'Oracle (Unlimited)':<20} {oracle_audc:<10.4f} {oracle_peak:<10.4f} {oracle_qnc:<10.4f}")
print(f"{'Oracle (All Limits)':<20} {oracle_limited_audc:<10.4f} {oracle_limited_peak:<10.4f} {oracle_limited_qnc:<10.4f}")
print(f"{'Router':<20} {router_audc:<10.4f} {router_peak:<10.4f} {router_qnc:<10.4f}")
print(f"{'Carrot':<20} {carrot_audc:<10.4f} {carrot_peak:<10.4f} {carrot_qnc:<10.4f}")
print(f"{'Linear Carrot':<20} {linear_audc:<10.4f} {linear_peak:<10.4f} {linear_qnc:<10.4f}")
print(f"{'MIRT (dim=256)':<20} {irt_256_audc:<10.4f} {irt_256_peak:<10.4f} {irt_256_qnc:<10.4f}")
print(f"{'MIRT (dim=32)':<20} {irt_32_audc:<10.4f} {irt_32_peak:<10.4f} {irt_32_qnc:<10.4f}")
print(f"{'NIRT (dim=256)':<20} {nirt_256_audc:<10.4f} {nirt_256_peak:<10.4f} {nirt_256_qnc:<10.4f}")
print(f"{'NIRT (dim=2)':<20} {nirt_2_audc:<10.4f} {nirt_2_peak:<10.4f} {nirt_2_qnc:<10.4f}")

# Relative improvements (Router vs best baseline)
best_baseline_audc = max(carrot_audc, irt_256_audc, irt_32_audc, nirt_256_audc, nirt_2_audc, linear_audc)
best_baseline_peak = max(carrot_peak, irt_256_peak, irt_32_peak, nirt_256_peak, nirt_2_peak, linear_peak)
best_baseline_qnc = min(carrot_qnc, irt_256_qnc, irt_32_qnc, nirt_256_qnc, nirt_2_qnc, linear_qnc)  # Lower QNC is better

audc_improvement = ((router_audc - best_baseline_audc) / best_baseline_audc) * 100
peak_improvement = ((router_peak - best_baseline_peak) / best_baseline_peak) * 100
qnc_improvement = ((best_baseline_qnc - router_qnc) / best_baseline_qnc) * 100  # Negative is good for QNC

print(f"\n" + "=" * 60)
print("ROUTER vs BEST BASELINE IMPROVEMENTS")
print("=" * 60)
print(f"AUDC improvement: {audc_improvement:+.2f}%")
print(f"Peak accuracy improvement: {peak_improvement:+.2f}%")
print(f"QNC improvement (lower is better): {qnc_improvement:+.2f}%")

# Oracle efficiency analysis
print(f"\n" + "=" * 60)
print("ORACLE EFFICIENCY ANALYSIS")
print("=" * 60)

# Router vs Oracle (Unlimited)
router_oracle_audc_gap = ((oracle_audc - router_audc) / oracle_audc) * 100
router_oracle_peak_gap = ((oracle_peak - router_peak) / oracle_peak) * 100
router_oracle_qnc_gap = ((router_qnc - oracle_qnc) / oracle_qnc) * 100  # Positive means router is worse

print(f"Router vs Oracle (Unlimited Token Limits):")
print(f"  AUDC gap: {router_oracle_audc_gap:.2f}% (lower is better)")
print(f"  Peak accuracy gap: {router_oracle_peak_gap:.2f}% (lower is better)")
print(f"  QNC gap: {router_oracle_qnc_gap:.2f}% (lower is better)")

# Router vs Oracle (All Limits)
router_oracle_limited_audc_gap = ((oracle_limited_audc - router_audc) / oracle_limited_audc) * 100
router_oracle_limited_peak_gap = ((oracle_limited_peak - router_peak) / oracle_limited_peak) * 100
router_oracle_limited_qnc_gap = ((router_qnc - oracle_limited_qnc) / oracle_limited_qnc) * 100  # Positive means router is worse

print(f"\nRouter vs Oracle (All Token Limits):")
print(f"  AUDC gap: {router_oracle_limited_audc_gap:.2f}% (lower is better)")
print(f"  Peak accuracy gap: {router_oracle_limited_peak_gap:.2f}% (lower is better)")
print(f"  QNC gap: {router_oracle_limited_qnc_gap:.2f}% (lower is better)")

# Oracle (Unlimited) vs Oracle (All Limits)
oracle_comparison_audc = ((oracle_limited_audc - oracle_audc) / oracle_audc) * 100
oracle_comparison_peak = ((oracle_limited_peak - oracle_peak) / oracle_peak) * 100
oracle_comparison_qnc = ((oracle_qnc - oracle_limited_qnc) / oracle_limited_qnc) * 100

print(f"\nOracle (All Limits) vs Oracle (Unlimited):")
print(f"  AUDC improvement: {oracle_comparison_audc:.2f}% (positive means All Limits is better)")
print(f"  Peak accuracy improvement: {oracle_comparison_peak:.2f}% (positive means All Limits is better)")
print(f"  QNC improvement: {oracle_comparison_qnc:.2f}% (positive means All Limits is better)")


# %%
