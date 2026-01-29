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
- **Multi-Model Routing**: Selects from a heterogeneous pool of 15 LLMs (0.6B to 235B parameters)
- **Token Budget Optimization**: Searches over 16 token budgets {10, 20, 30, ..., 4000, default}
- **Continuous Interpolation**: Approximates continuous quality-cost curves from sparse anchor points
- **Plug-in Module**: Can be integrated with existing routers (e.g., UniRouter) as a drop-in enhancement
- **Baseline Comparisons**: Includes CARROT (KNN, Linear), IRT (MIRT, NIRT), and UniRouter baselines
- **Interactive Demo**: Gradio web interface for real-time routing demonstrations

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

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)

### Setup

```bash
git clone https://github.com/anonymous/r2-router.git
cd r2-router

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

### 1. IID Evaluation

Run the complete training and evaluation pipeline:

```bash
bash main/run_experiment.sh
```

This will:
1. Train R2-Router quality-cost predictors for all LLMs
2. Train baseline methods (CARROT-KNN, CARROT-Linear, MIRT, NIRT)
3. Evaluate routing across different lambda values
4. Generate comparison metrics (AUDC, Peak Quality, QNC) and plots

Results saved to `comparison_results/main/`

### 2. Out-of-Distribution (OOD) Evaluation

Test generalization to unseen query categories (leave-one-category-out):

```bash
# Default: hold out MMLU-Pro
python ood_evaluation/run_ood.py

# Quick test with 1 model
python ood_evaluation/run_ood.py --quick

# Hold out a different category
python ood_evaluation/run_ood.py --category "lighteval/MATH/all"
```

Results saved to `comparison_results/ood_evaluation/{category}/`

### 3. UniRouter Integration

Compare R2-Router integrated with UniRouter for dynamic LLM pools:

```bash
bash unirouter/run_unirouter_experiment.sh
```

Results saved to `comparison_results/unirouter/`

### 4. Interactive Demo

```bash
export OPENROUTER_API_KEY="your-api-key-here"
cd demo && python app.py
```

Browser opens at `http://localhost:7860`

## Project Structure

```
r2-router/
├── main/                        # Main evaluation code
│   ├── core/                    # R2-Router predictor implementations
│   │   ├── predictor.py         # PyTorch MLP (3-layer, [256,128,64])
│   │   └── predictor_sklearn.py # Ridge regression (faster)
│   ├── baselines/               # Baseline methods
│   │   ├── carrot/              # CARROT-KNN, CARROT-Linear
│   │   └── irt/                 # MIRT, NIRT
│   ├── shared/                  # Shared utilities
│   │   ├── dataset_manager.py   # Centralized train/test split
│   │   ├── llm_loader.py        # LLM data loader
│   │   ├── router_dataset.py    # Dataset wrapper
│   │   └── utils.py             # Utility functions
│   ├── evaluation/              # Evaluation scripts
│   │   ├── compare_methods.py   # Method comparison
│   │   └── results.py           # IID evaluation
│   └── run_experiment.sh        # Automated pipeline
├── ood_evaluation/              # Out-of-distribution evaluation
│   ├── run_ood.py               # Main OOD script
│   └── ood_dataset_manager.py   # Category-based splits
├── unirouter/                   # UniRouter integration
│   ├── eval_compare.py          # UniRouter vs Uni-R2Router
│   ├── uni_core.py              # Uni-R2Router implementation
│   └── unirouter_original.py    # Original UniRouter
├── demo/                        # Interactive web demo
│   ├── app.py                   # Gradio interface
│   ├── router.py                # R2-Router routing logic
│   ├── baselines.py             # Baseline routers
│   ├── llm_client.py            # OpenRouter API client
│   ├── judge.py                 # Quality evaluation
│   └── config.py                # Configuration
├── data/                        # Dataset (not included, see below)
├── checkpoints/                 # Trained models (not included)
└── README.md
```

## Training

### Train R2-Router Predictors

```bash
# Ridge regression (fast, recommended)
python -m main.core.train_core \
    --model_type sklearn \
    --model "Model-Name" "0.85" "data/Model-Name.csv" "checkpoints/Model-Name"

# PyTorch MLP (higher accuracy)
python -m main.core.train_core \
    --model_type torch_mlp \
    --model "Model-Name" "0.85" "data/Model-Name.csv" "checkpoints/Model-Name"
```

### Train Baselines

```bash
# CARROT baselines (KNN + Linear)
python -m main.baselines.carrot.train_carrot --model ...

# IRT baselines (MIRT + NIRT)
python -m main.baselines.irt.train_irt --model ...
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

## Data Format

Data files are not included due to size. To use your own data:

1. **CSV files** (`data/{model}.csv`): One row per query (30,968 total)
   - `prompts_id`, `key`, `original_prompt`
   - `{budget}_score`: Quality score [0, 1] for each token budget
   - `{budget}_count`: Actual token count used

2. **Embeddings** (`data/prompt_embeddings.pkl`):
   - NumPy array of shape `(30968, 768)`
   - Generated using sentence-transformers (all-mpnet-base-v2)

## Configuration

Key parameters:

- **lambda**: Cost-quality tradeoff coefficient [0, 1]
  - lambda=0: Maximize quality only
  - lambda->1: Minimize cost
- **LLM Pool**: 15 models from 0.6B to 235B parameters
- **Token Budgets**: 16 levels from 10 to 4000 tokens + default (unlimited)
- **Train/Test Split**: 80/20 with seed=42

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
