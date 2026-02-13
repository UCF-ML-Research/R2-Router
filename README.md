# R2-Router: A New Paradigm for LLM Routing with Reasoning

**R2-Router** introduces *reasoning* into LLM routing. Instead of treating each LLM as a fixed quality-cost point, R2-Router reasons about how quality varies with output length, jointly selecting the best LLM **and** token budget. This transforms routing from selecting among points to searching over quality-cost curves, achieving state-of-the-art performance at **4-5x lower cost**.

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

## Dataset: R2-Bench

R2-Bench is the first LLM routing dataset capturing behavior across diverse output length budgets:
- **30,968 queries** across **20 categories** from 6 benchmarks
- **6 benchmarks**: MMLU-Pro, OpenHermes, MATH, GPQA, MuSR, RAGBench
- **15 LLMs**: From Qwen3-0.6B to Qwen3-235B (general-purpose and domain-specific)
- **16 token budgets**: {10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000, default}
- Quality scored by Qwen3-80B-Instruct (validated against 30 human annotators, Pearson r=0.82)

R2-Bench raises the Oracle upper bound by **15% in AUDC** compared to prior single-response datasets.

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
├── main/                        # IID evaluation pipeline (R2-Bench)
│   ├── r2/                        # R2-Router predictors (sklearn)
│   ├── baselines/                 # CARROT, IRT baselines
│   ├── shared/                    # DatasetManager, utils
│   ├── evaluation/                # Compare methods
│   └── run_experiment.sh          # Automated pipeline
├── routerarena_submission/      # RouterArena submission files
├── data_collection/             # R2-Bench data pipeline
├── ood_evaluation/              # Out-of-distribution evaluation
├── unirouter/                   # UniRouter integration
└── demo/                        # Interactive web demo (Gradio)
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

### IID Evaluation (R2-Bench)

```bash
bash main/run_experiment.sh
```

## Evaluation Metrics

Following the evaluation protocol from UniRouter (Jitkrittum et al., 2025):

- **AUDC (Area Under Deferral Curve)**: Overall quality-cost tradeoff (higher is better)
- **Peak Quality**: Maximum achievable quality [0, 1] (higher is better)
- **QNC (Query-Normalized Cost)**: Minimum relative cost to match the best single LLM's performance (lower is better)

## Main Results

R2-Router achieves comparable quality at **4-5x lower cost** compared to reactive baselines:

| Method | AUDC | QNC | Peak Quality |
|--------|------|-----|-------------|
| MIRT | 0.74 | 0.78 | 0.81 |
| CARROT-L | 0.77 | 0.66 | 0.80 |
| **R2-Router** | **0.80** | **0.29** | **0.81** |

## Citation

If you use this code or R2-Bench, please cite:

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

## Acknowledgments

- [SPROUT](https://arxiv.org/abs/2502.03261) benchmark for the routing evaluation framework
- [UniRouter](https://arxiv.org/abs/2502.08773) for dynamic LLM pool routing
- [vLLM](https://github.com/vllm-project/vllm) for efficient LLM serving
- [OpenRouter](https://openrouter.ai/) for LLM API access and pricing data
