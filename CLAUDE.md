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
2. **Quality predictor** (Ridge per category×model×budget): Predicts accuracy for each (model, budget) pair
3. **Token predictor** (LinearRegression per category×model): Predicts output token count
4. **Routing**: Compute risk for all (model, budget), pick argmax

### Shrinkage (optional, controlled by `--shrinkage_k`)

- **Quality**: `alpha = clip(CV_R2 × k, 0, 1)`, blend = `alpha × predictor + (1-alpha) × category_mean`
- **Token**: Binary switch — use predictor if `token_R2 > 0`, else fall back to category mean

### Quality Predictor Analysis

- **R2 avg = 0.091** (weak but positive signal). Best: Trivia (alpha=1.0). Worst: Math, Knowledge (alpha~0.14-0.19)
- **93% of accuracy targets are binary** (0 or 1) — regression on classification targets
- **Token predictor R2 avg = -19.7** (catastrophically bad). 85.7% fall back to category mean under shrinkage

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

## Current Status (Feb 11, 2026)

### Feb 10 Submission (Baseline)

Submitted to RouterArena via commit `7788685` in RouterArena repo.

| | Value |
|--|---|
| **Models** | 4 (235b=91.8%, gemini-flash=5.2%, 80b=2.9%, haiku=0.1%) |
| **λ** | 0.999 |
| **Shrinkage** | No |
| **Budgets** | 9 (budget_10~800 + concise, no unlimited/1500) |
| **RouterArena Score** | **72.20** (Acc=71.94%, Cost=$0.065/1kq) |
| **Local sweep eval** | Arena=71.97 (Acc=71.94%, Cost=$0.063/1kq) — matches |

Submitted file backed up at: `routerarena_submission/r2-router-feb10-submitted.json`

### Current Checkpoint

Trained with `--full --ridge-only` (all 8400 queries, no held-out test set).
**Cannot reliably evaluate locally** — any local metrics are optimistic (training set evaluation).

Local results on training data (for reference only, NOT real performance):

| Setting | Acc | Cost/1kq (sweep) | Arena (sweep) |
|---------|-----|-------------------|---------------|
| raw λ=0.999, 4 models, excl unlimited+1500 | 73.05% | $0.061 | 73.07 |
| raw λ=0.999, all 12 models | 74.13% | $0.088 | 74.11 |
| raw λ=0.98, all 12 models | 75.64% | $0.216 | 75.50 |
| shrinkage k=3, λ=0.98, all 12 models | 73.28% | $0.168 | 73.20 |

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
├── train_category_predictors.py       # Train Ridge quality + token predictors
├── train_category_classifier.py       # Train SVM category classifier
├── build_category_training_data.py    # Build training_data.pkl from sweep files
├── inference_budget_sweep.py          # Budget sweep inference (vLLM)
├── inference_routerarena.py           # vLLM/API inference engine
├── inference_api.py                   # OpenRouter API inference
├── eval_sweep.py                      # Evaluate sweep files (writes accuracy/cost)
├── eval_model.py                      # Evaluate single model predictions
├── merge_predictions.py               # Merge per-model results into submission
├── env.sh                             # Environment variables for sbatch
├── sweep_model.sbatch                 # Unified sweep template (--export=MODEL=...,MODEL_SHORT=...,TP_SIZE=N)
├── gemma3n_full_pipeline.sbatch       # Gemma 3N sweep+eval pipeline
├── inference_*_chat.sbatch            # Per-model inference jobs
├── train_*.sbatch                     # Training jobs
└── eval_*.sbatch                      # Evaluation jobs

checkpoints/category_router/           # Active checkpoint (26MB)
├── classifier.joblib                  # SVM-RBF category classifier
├── predictors/{Category}/             # Per-category predictor files
│   ├── {model}_quality_meta.json      # Budget list, CV R²
│   ├── {model}_{budget}_quality.joblib # Ridge quality predictor
│   └── {model}_token.joblib           # LinearRegression token predictor
├── predictor_results.json             # CV R² per category×model
├── category_means.pkl                 # Category-level accuracy/token means
└── train_test_split.pkl               # Train/test indices (currently full=8400/0)

routerarena_submission/                # Submission files
├── r2-router-feb10-submitted.json     # Feb 10 baseline (backed up from git)
├── routerarena_embeddings.pkl         # 8400 query embeddings (Qwen3-0.6B)
└── routerarena_robustness_embeddings.pkl

main/                                  # IID evaluation pipeline (R2-Bench)
├── r2/                                # R2-Router predictors (PyTorch + sklearn)
├── baselines/                         # CARROT, IRT baselines
├── shared/                            # DatasetManager, utils
└── evaluation/                        # Compare methods

data_collection/                       # R2-Bench data pipeline
```

## Key Commands

```bash
# Route and evaluate locally (against sweep ground truth)
.venv/bin/python scripts/route_and_eval.py --lambda_val 0.98 --shrinkage_k 3.0
.venv/bin/python scripts/route_and_eval.py --lambda_val 0.999 --shrinkage_k 0 \
    --models 235b 80b gemini-flash haiku --exclude-budgets budget_unlimited budget_1500

# Train predictors
sbatch scripts/train_predictors.sbatch      # Ridge quality + token
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

- **Predictor trained on full data** (`--full`): Current checkpoint uses all 8400 queries for training. Local eval is optimistic. Must retrain with 80/20 split or rely on RouterArena eval.
- **Token predictor is catastrophic**: R2 avg = -19.7. Shrinkage binary switch (token_R2 > 0 → predictor, else → mean) effectively disables it for 85.7% of cases.
- **93% binary targets**: Accuracy values are mostly 0/1 — classification (Logistic Regression) may outperform Ridge regression.
- **code_accuracy memory leak**: LiveCodeBench code eval causes OOM after 2-3 models at 64gb. Use per-model eval jobs.
- **Budget prompt is counter-productive**: Concise prompt (median=6 tokens) beats budget=10 (median=36 tokens) for controlling output length.
- **Node c1000a-s15** has GPU issues — exclude with `#SBATCH --exclude=c1000a-s15`
- **vLLM API versions**: v0.7 uses `LLM(task="embed")`, v0.15+ uses `LLM(runner="pooling")`
