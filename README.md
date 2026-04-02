# R2-Router for RouterArena

**R2-Router** introduces *reasoning* into LLM routing. Instead of treating each LLM as a fixed quality-cost point, R2-Router reasons about how quality varies with output length, jointly selecting the best LLM **and** token budget. This transforms routing from selecting among points to searching over quality-cost curves.

> Under review at ICML 2026.

## Overview

Existing LLM routers assume each model has a single fixed quality-cost profile per query. This causes them to exclude powerful LLMs when estimated cost exceeds the budget, missing the opportunity that these LLMs could still deliver high quality with shorter outputs.

R2-Router addresses this by:
1. **Reasoning about cost-dependent quality**: Predicting how each LLM's quality varies with output length
2. **Routing on curves, not points**: Searching over all (LLM, token budget) combinations
3. **Enforcing budget via instructions**: Using length-constrained prompts (e.g., "use at most K tokens")

This enables R2-Router to discover that a powerful LLM with constrained output can outperform a weaker LLM at comparable cost -- efficient configurations invisible to prior methods.

## Key Features

- **Reasoning-based Routing**: Models each LLM as a quality-cost curve rather than a single point
- **Multi-Model Routing**: Selects from a heterogeneous pool of LLMs (0.6B to 235B parameters)
- **Token Budget Optimization**: Searches over multiple token budgets + concise prompt
- **KNN-based Prediction**: Uses cosine-distance-weighted KNN for quality prediction per (category, model, budget)
- **Baseline Comparisons**: Includes CARROT (KNN, Linear), IRT (MIRT, NIRT), and UniRouter baselines
- **RouterArena Submission**: Submitted to the [RouterArena](https://github.com/RouteWorks/RouterArena) leaderboard

## Scope

This public release focuses on the **RouterArena branch** of R2-Router:

- category-aware routing on RouterArena queries
- per-model, per-budget quality prediction
- offline evaluation against sweep ground truth
- RouterArena-format submission export

## Installation

### Prerequisites

- Python 3.11+
- scikit-learn, numpy, joblib

### Setup

```bash
git clone https://github.com/jqxue1999/router.git
cd router

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Public Release Notes

This working repository contains research code plus local-only assets used during development. For the public artifact:

- keep `scripts/`, `ood_evaluation/`, and `unirouter/`
- exclude `demo/`, `old_demo/`, `hf_space/`, and `hf_upload/`
- do not commit checkpoints, submission JSONs, local logs, or sweep outputs
- configure dataset and sweep locations through environment variables in [.env.example](/home/ji757406.ucf/router/.env.example)

The RouterArena-oriented scripts preserve the current local defaults, but can now be redirected with environment variables such as `R2_SWEEP_ROOT`, `R2_ROUTER_DATA_PATH`, and `R2_CHECKPOINT_DIR`.

Public artifacts:

- RouterArena data: [JiaqiXue/r2-router-routerarena-data](https://huggingface.co/datasets/JiaqiXue/r2-router-routerarena-data)
- RouterArena checkpoints: [JiaqiXue/r2-router-routerarena-checkpoints](https://huggingface.co/JiaqiXue/r2-router-routerarena-checkpoints)

## Project Structure

```
r2-router/
├── scripts/                     # RouterArena pipeline
│   ├── category_config.py         # Shared config (models, paths, categories)
│   ├── route_and_eval.py          # Route queries + evaluate against sweep GT
│   ├── route_knn_export.py        # Export RouterArena submission JSON (Global KNN)
│   ├── sweep_lambda_global_knn.py # Lambda sweep with Global KNN routing
│   ├── train_category_predictors.py # Train KNN quality + token predictors
│   ├── train_category_classifier.py # Train SVM category classifier
│   ├── build_category_training_data.py # Build training_data.pkl from sweep files
│   ├── inference_budget_sweep.py   # Budget sweep inference (vLLM)
│   ├── inference_routerarena.py    # vLLM/API inference engine
│   ├── eval_sweep.py              # Evaluate sweep files (writes accuracy/cost)
│   └── *.sbatch                   # SLURM job scripts
├── routerarena_submission/      # Local submission/embedding artifacts (not for git release)
├── ood_evaluation/              # Out-of-distribution evaluation
├── unirouter/                   # UniRouter integration
├── DATA_RELEASE.md              # Data packaging and licensing guidance
├── reproduce/                   # Minimal reproduction entrypoints
└── artifacts/                   # Notes for assembling the public artifact
```

## Quick Start

### RouterArena Evaluation

```bash
# Route and evaluate locally (against sweep ground truth)
.venv/bin/python scripts/route_and_eval.py --lambda_val 0.98 --shrinkage_k 3.0

# Train KNN predictors
sbatch scripts/train_predictors.sbatch

# Export submission for RouterArena
.venv/bin/python scripts/route_knn_export.py \
    --models 235b ministral-3b gemini-flash --lambda_val 0.85 \
    --export routerarena_submission/submission.json
```

### RouterArena Reproduction

For the public RouterArena branch, prepare the released data package and set the paths in [.env.example](/home/ji757406.ucf/router/.env.example).

Minimum required environment variables:

```bash
export R2_SWEEP_ROOT=/path/to/routerarena_data_release/budget_sweep
export R2_TRAINING_DATA_PATH=/path/to/routerarena_data_release/category_router/training_data.pkl
export R2_EMBEDDINGS_PATH=/path/to/routerarena_data_release/embeddings/routerarena_embeddings.pkl
export R2_ROUTER_DATA_PATH=/path/to/routerarena_data_release/routerarena_meta/router_data.json
export R2_ROUTER_DATA_10_PATH=/path/to/routerarena_data_release/routerarena_meta/router_data_10.json
export R2_MODEL_COST_PATH=/path/to/routerarena_data_release/routerarena_meta/model_cost.json
```

Then run:

```bash
bash reproduce/routerarena_train.sh
bash reproduce/routerarena_eval.sh
```

The helper scripts are:

- [routerarena_train.sh](/home/ji757406.ucf/router/reproduce/routerarena_train.sh): builds consolidated training data and trains RouterArena predictors
- [routerarena_eval.sh](/home/ji757406.ucf/router/reproduce/routerarena_eval.sh): runs offline evaluation and exports a submission JSON

## Evaluation Metrics

Following the evaluation protocol from UniRouter (Jitkrittum et al., 2025):

- **AUDC (Area Under Deferral Curve)**: Overall quality-cost tradeoff (higher is better)
- **Peak Quality**: Maximum achievable quality [0, 1] (higher is better)
- **QNC (Query-Normalized Cost)**: Minimum relative cost to match the best single LLM's performance (lower is better)

## Main Results

R2-Router is released here as a RouterArena-oriented routing system with:

- per-category KNN quality predictors
- token-aware routing objectives
- Global KNN submission export

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{r2router2026,
  title={R2-Router: A New Paradigm for LLM Routing with Reasoning},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

MIT License

## Data And Artifact Release

This repository does not yet bundle the full research dataset. Use the guidance in [DATA_RELEASE.md](/home/ji757406.ucf/router/DATA_RELEASE.md) and [README.md](/home/ji757406.ucf/router/artifacts/README.md) to package a public release safely:

- publish code separately from large data artifacts
- verify redistribution rights for third-party benchmark content
- release derived metadata, splits, and reconstructions when raw prompts/answers cannot be mirrored directly

## Acknowledgments

- [SPROUT](https://arxiv.org/abs/2502.03261) benchmark for the routing evaluation framework
- [UniRouter](https://arxiv.org/abs/2502.08773) for dynamic LLM pool routing
- [vLLM](https://github.com/vllm-project/vllm) for efficient LLM serving
- [OpenRouter](https://openrouter.ai/) for LLM API access and pricing data
