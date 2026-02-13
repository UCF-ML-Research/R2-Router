# CLAUDE.md

## Project Overview

**R2-Router** — an LLM routing system that selects the optimal (LLM, token_budget) pair for each query, balancing accuracy and cost. Submitted to the **RouterArena** leaderboard.

## Routing Architecture

### Formula

```
risk(M, b) = (1-λ) × pred_quality(x, M, b) - λ × pred_output_tokens(x, M) × output_price_M / 1e6
(M*, b*) = argmax risk
```

### Category-Aware Routing

1. **Classifier** (SVM-RBF): Assigns each query to 1 of 7 categories (Code, Math, Knowledge, NLU, Translation, Trivia, Domain)
2. **Quality predictor** (KNN per category×model×budget): Predicts accuracy for each (model, budget) pair
3. **Token predictor** (LinearRegression per category×model): Predicts output token count
4. **Routing**: Compute risk for all (model, budget), pick argmax

### Shrinkage (optional, controlled by `--shrinkage_k`)

- **Quality**: `alpha = clip(CV_R2 × k, 0, 1)`, blend = `alpha × predictor + (1-alpha) × category_mean`
- **Token**: Binary switch — use predictor if `token_R2 > 0`, else fall back to category mean

### LLM Pool (12 models)

| Short Name | Model | Output $/M | Source |
|-----------|-------|-----------|--------|
| 235b | Qwen3-235B-A22B-Instruct-2507 | $0.463 | vLLM (4 GPU) |
| 80b | Qwen3-Next-80B-A3B-Instruct | $1.10 | vLLM (2 GPU) |
| 30b | Qwen3-30B-A3B-Instruct-2507 | $0.33 | vLLM (1 GPU) |
| coder-next | Qwen3-Coder-Next | $0.30 | vLLM (4 GPU) |
| coder-30b | Qwen3-Coder-30B-A3B-Instruct | $0.27 | vLLM (1 GPU) |
| ministral-14b | Ministral-3-14B-Instruct-2512 | $0.20 | vLLM (1 GPU) |
| ministral-8b | Ministral-3-8B-Instruct-2512 | $0.15 | vLLM (1 GPU) |
| ministral-3b | Ministral-3-3B-Instruct-2512 | $0.10 | vLLM (1 GPU) |
| gpt4o | GPT-4o | $10.0 | OpenRouter |
| gemini-flash | Gemini 2.5 Flash | $2.50 | OpenRouter |
| haiku | Claude 3 Haiku | $1.25 | OpenRouter |
| gemma-3n-e4b | Gemma 3N E4B IT | $0.04 | vLLM (1 GPU) |

**Key insight**: 80B is the most expensive vLLM model (2.4x of 235B). gpt4o is most expensive overall ($10/M).

### RouterArena Arena Score

```
S = ((1 + β) × A × C) / (β × A + C) × 100
```
- `A` = mean accuracy (0~1)
- `C` = normalized cost = `(log₂(200) - log₂(cost_per_1kq)) / (log₂(200) - log₂(0.0044))`
- `β` = 0.1 — β < 1 favors accuracy, β > 1 favors cost

## Current Status (Feb 13, 2026)

### Feb 13 Submission (Current Best)

Submitted to RouterArena PR #68 on branch `r2-router-submission`.
Uses **Global KNN** routing (trained on sub_10 only, ~809 queries).

| | Value |
|--|---|
| **Method** | Global KNN (cosine, distance-weighted) |
| **Models** | 6 (235b, 80b, 30b, coder-next, gemini-flash, haiku) |
| **Budgets** | 4 (concise, budget_200, budget_400, budget_800) |
| **λ** | 0.999 |
| **K** | 80 |
| **Accuracy** | 71.16% |
| **Cost/1kq** | $0.035 |
| **Arena(β=0.1)** | **71.93** |

Submission files:
- `routerarena_submission/r2-router-feb13.json` — 12,445 entries (8400 regular + 4045 optimality)
- Export script: `scripts/route_knn_export.py`

```bash
# Reproduce
.venv/bin/python scripts/route_knn_export.py \
    --models 235b 80b 30b coder-next gemini-flash haiku \
    --exclude-budgets budget_10 budget_20 budget_40 budget_80 budget_150 budget_1500 budget_unlimited \
    --lambda_val 0.999 --k 80 \
    --export routerarena_submission/r2-router-feb13.json
```

### Feb 10 Submission (Baseline)

Submitted to RouterArena via commit `7788685` in RouterArena repo.

| | Value |
|--|---|
| **Method** | Category-Aware KNN (per-category predictors) |
| **Models** | 4 (235b=91.8%, gemini-flash=5.2%, 80b=2.9%, haiku=0.1%) |
| **λ** | 0.999 |
| **Shrinkage** | No |
| **Budgets** | 9 (budget_10~800 + concise, no unlimited/1500) |
| **RouterArena Score** | **72.20** (Acc=71.94%, Cost=$0.065/1kq) |

Submitted file backed up at: `routerarena_submission/r2-router-feb10-submitted.json`

### Evaluation Protocol

RouterArena 是开源平台，所有人都能拿到 full query set 和 answers。官方只提供 sub_10（~840 queries，10%）作为训练集，他们在 full 8400 queries 上验证。**用 full data 训练 = 作弊。**

- **只能用 sub_10 训练，full set 测试。** 不做 CV，不做 full training，任何时候都不用 full data。
- 提交也必须用 sub_10 训练的模型。

### Optimization Findings (Feb 13)

- **Semi-supervised methods all underperform KNN**: Label Propagation (71.36), Cluster-Then-Route (70.79), Neighbor Voting (70.18) — all worse than KNN baseline (71.88).
- **Global KNN > Category-Aware KNN**: With only ~80 training samples per category, global KNN with k=80 generalizes better.
- **6-model pool > 4-model pool**: Adding 30b and coder-next improves routing diversity.
- **Oracle upper bound**: 84.55 Arena (perfect per-query routing, 4 models, concise-only). Current gap: ~12.6pp.

### Budget Sweep Data: COMPLETE

All 11 models × 11 budgets evaluated on 8400 RouterArena queries.
Sweep files: `/orange/qi855292.ucf/ah872032.ucf/budget_sweep/{model}/{budget}.json`
Each file has per-entry `accuracy` and `cost` from `eval_sweep.py`.

**Budgets**: `[budget_10, budget_20, budget_40, budget_80, budget_150, budget_200, budget_400, budget_800, budget_1500, budget_unlimited, concise]`

### Training Data

`/orange/qi855292.ucf/ah872032.ucf/category_router/training_data.pkl`:
- `embeddings`: (8400, 1024) — Qwen3-0.6B
- `global_indices`: [str, ...] — aligned with RouterArena `router_data.json`
- `categories`: (8400,) — 0~6
- `models`: {model: {budget: {accuracy: (8400,), output_tokens: (8400,)}}}

## Code Organization

```
scripts/                               # Main pipeline
├── category_config.py                 # Shared config (models, paths, categories)
├── route_and_eval.py                  # Route queries + evaluate against sweep GT
├── train_category_predictors.py       # Train KNN quality + token predictors
├── train_category_classifier.py       # Train SVM category classifier
├── build_category_training_data.py    # Build training_data.pkl from sweep files
├── inference_budget_sweep.py          # Budget sweep inference (vLLM)
├── inference_routerarena.py           # vLLM/API inference engine
├── inference_api.py                   # OpenRouter API inference
├── eval_sweep.py                      # Evaluate sweep files (writes accuracy/cost)
├── eval_model.py                      # Evaluate single model predictions
├── merge_predictions.py               # Merge per-model results into submission
├── env.sh                             # Environment variables for sbatch
├── sweep_model.sbatch                 # Unified sweep template
├── inference_*_chat.sbatch            # Per-model inference jobs
├── train_*.sbatch                     # Training jobs
└── eval_*.sbatch                      # Evaluation jobs

checkpoints/category_router/           # Active checkpoint
├── classifier.joblib                  # SVM-RBF category classifier
├── predictors/{Category}/             # Per-category predictor files
│   ├── {model}_quality_meta.json      # Budget list, CV R²
│   ├── {model}_{budget}_quality.joblib # KNN quality predictor
│   └── {model}_token.joblib           # LinearRegression token predictor
├── predictor_results.json             # CV R² per category×model
├── category_means.pkl                 # Category-level accuracy/token means
└── train_test_split.pkl               # Train/test indices

routerarena_submission/                # Submission files
├── r2-router-feb13.json               # Feb 13 submission (current best)
├── r2-router-feb10-submitted.json     # Feb 10 baseline
├── routerarena_embeddings.pkl         # 8400 query embeddings (Qwen3-0.6B)
└── routerarena_robustness_embeddings.pkl

experiments/                           # Experiment scripts
├── quick_semisupervised.py            # KNN/LP/Cluster/NV comparison
└── semisupervised_routing.py          # Full semi-supervised routing

main/                                  # IID evaluation pipeline (R2-Bench)
├── r2/                                # R2-Router predictors (sklearn)
├── baselines/                         # CARROT, IRT baselines
├── shared/                            # DatasetManager, utils
└── evaluation/                        # Compare methods

data_collection/                       # R2-Bench data pipeline
```

## Key Commands

```bash
# Generate submission (Global KNN, sub_10 training)
.venv/bin/python scripts/route_knn_export.py \
    --models 235b 80b 30b coder-next gemini-flash haiku \
    --exclude-budgets budget_10 budget_20 budget_40 budget_80 budget_150 budget_1500 budget_unlimited \
    --lambda_val 0.999 --k 80 \
    --export routerarena_submission/r2-router-feb13.json

# Route and evaluate locally (against sweep ground truth)
.venv/bin/python scripts/route_and_eval.py --lambda_val 0.98 --shrinkage_k 3.0
.venv/bin/python scripts/route_and_eval.py --lambda_val 0.999 --shrinkage_k 0 \
    --models 235b 80b gemini-flash haiku --exclude-budgets budget_unlimited budget_1500

# Train predictors (KNN quality + token)
sbatch scripts/train_predictors.sbatch
sbatch scripts/train_classifier.sbatch      # SVM category classifier

# Budget sweep inference (unified template)
sbatch --partition=hpg-b200 --gres=gpu:4 \
    --export=MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507",MODEL_SHORT=235b,TP_SIZE=4 \
    scripts/sweep_model.sbatch

# Evaluate sweep files
sbatch scripts/eval_sweep_all.sbatch        # All models
```

## HiPerGator Environment

- **Account**: qi855292.ucf | **User**: ah872032.ucf
- **Group limits**: 48 CPUs, 8 GPUs, 375GB mem total
- **Partitions**: `hpg-b200` (B200 GPU), `hpg-turin` (L4 GPU), `hpg-default` (CPU)
- **Python**: uv-managed `.venv/` (Python 3.11), run via `.venv/bin/python`
- **Storage**: Code in `~/jiaqi/router/`, data on `/orange/qi855292.ucf/ah872032.ucf/`
- **Caches**: All on Orange (`HF_HOME`, `VLLM_CACHE_ROOT`, etc.) — see `scripts/env.sh`
- **NEVER compute on login node** — always use `sbatch`
- **Mem limits**: 32gb for inference, 64gb for eval (code_accuracy leaks memory)

### SLURM Quick Reference

```bash
sbatch scripts/job.sbatch      # Submit
squeue -u $USER                # Check status
tail -f logs/job_JOBID.log     # Watch output
scancel JOBID                  # Cancel
```

## Per-Model Sampling Params

| Model | temperature | top_p | top_k | repetition_penalty |
|-------|-----------|-------|-------|-------------------|
| Qwen3 (default) | 0.7 | 0.8 | 20 | 1.05 |
| Coder-Next | 1.0 | 0.95 | 40 | 1.05 |
| Ministral (all) | 0.1 | — | — | — |

## Known Issues

- **Sub_10 training data is the bottleneck**: Only ~809 training queries. Semi-supervised methods (LP, Clustering, NV) cannot close the gap. Oracle upper bound is 84.55 Arena.
- **Token predictor is catastrophic**: R2 avg = -19.7. Global KNN approach side-steps this by using λ=0.999 (risk formula dominated by cost penalty, not token prediction).
- **93% binary targets**: Accuracy values are mostly 0/1 — classification may outperform regression. Potential Phase 2 improvement.
- **code_accuracy memory leak**: LiveCodeBench code eval causes OOM after 2-3 models at 64gb. Use per-model eval jobs.
- **Budget prompt is counter-productive**: Concise prompt (median=6 tokens) beats budget=10 (median=36 tokens) for controlling output length.
- **Node c1000a-s15** has GPU issues — exclude with `#SBATCH --exclude=c1000a-s15`
- **vLLM API versions**: v0.7 uses `LLM(task="embed")`, v0.15+ uses `LLM(runner="pooling")`
