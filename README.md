# CoRE: Constrained Response Evaluator

**CoRE** is an intelligent LLM routing system that optimizes the cost-performance tradeoff by predicting which (LLM, token_limit) combination will best answer each query.

## Overview

CoRE addresses a fundamental challenge in LLM deployment: **How do we balance response quality with computational cost?**

Instead of always using the most expensive model with unlimited tokens, CoRE:
- Predicts the performance of different LLMs under various token budgets
- Routes each query to the optimal (model, token_limit) pair
- Achieves comparable performance to expensive models while significantly reducing costs

## Key Features

- **Multi-Model Routing**: Intelligently selects from a pool of LLMs (e.g., GPT-4, Llama, Qwen)
- **Token Budget Optimization**: Chooses optimal token limits (10, 20, 50, ..., unlimited)
- **Cost-Quality Tradeoff**: Configurable λ parameter balances performance vs. cost
- **Multiple Routing Methods**:
  - **CoRE**: Ridge regression-based predictor (our method)
  - **CARROT**: KNN and Linear baselines
  - **IRT**: Item Response Theory baselines (MIRT, NIRT)
- **Interactive Demo**: Web interface for real-time routing demonstrations

## Dataset: CoRD (Constrained Response Dataset)

CoRE is evaluated on **CoRD**, an extension of the SPROUT benchmark:
- **30,968 queries** across **20 categories**
- **6 diverse benchmarks**: MMLU-Pro, OpenHermes, MATH, GPQA, etc.
- **16 token budgets**: {10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000, unlimited}
- Responses from multiple LLMs under each budget

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/core-router.git
cd core-router

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Evaluation (IID Setting)

Run the complete evaluation pipeline:

```bash
# Train all methods and evaluate
bash main/run_experiment.sh
```

This will:
1. Train CoRE predictors for all LLMs
2. Train baseline methods (CARROT, IRT)
3. Evaluate routing performance across different λ values
4. Generate comparison metrics and plots

Results saved to `comparison_results/main/`

### 2. Out-of-Distribution (OOD) Evaluation

Test generalization to unseen query categories:

```bash
# Evaluate on MMLU-Pro (default held-out category)
python ood_evaluation/run_ood.py

# Quick test with 1 model
python ood_evaluation/run_ood.py --quick

# Test on different category
python ood_evaluation/run_ood.py --category "lighteval/MATH/all"
```

Results saved to `comparison_results/ood_evaluation/{category}/`

### 3. Interactive Demo

Launch the web interface:

```bash
# Set API key
export OPENROUTER_API_KEY="your-api-key-here"

# Start demo
cd demo
python app.py
```

Browser opens at `http://localhost:7860`

## Project Structure

```
core-router/
├── main/                      # Main evaluation code
│   ├── core/                  # CoRE predictor implementations
│   │   ├── predictor.py       # PyTorch neural network
│   │   └── predictor_sklearn.py  # Ridge regression (faster)
│   ├── baselines/             # Baseline methods
│   │   ├── carrot/            # CARROT (KNN, Linear)
│   │   └── irt/               # IRT (MIRT, NIRT)
│   ├── shared/                # Shared utilities
│   │   ├── dataset_manager.py # Train/test split management
│   │   └── llm_loader.py      # LLM data loader
│   ├── evaluation/            # Evaluation scripts
│   └── run_experiment.sh      # Automated pipeline
├── ood_evaluation/            # Out-of-distribution evaluation
├── unirouter/                 # UniRouter comparison
├── demo/                      # Interactive web demo
│   ├── app.py                 # Gradio interface
│   ├── router.py              # CoRE routing logic
│   ├── baselines.py           # Baseline routers
│   └── config.py              # Configuration
├── data/                      # Dataset (not included in repo)
│   ├── prompt_embeddings.pkl  # Query embeddings
│   └── {model-name}.csv       # Performance data per LLM
├── checkpoints/               # Trained models (not included)
└── README.md                  # This file
```

## Training Your Own Models

### Train CoRE Predictors

```bash
# Ridge regression (fast, recommended)
python -m main.core.train_core \
    --model_type sklearn \
    --model "Model-Name" "0.85" "data/Model-Name.csv" "checkpoints/Model-Name"

# Neural network (more accurate)
python -m main.core.train_core \
    --model_type torch_mlp \
    --model "Model-Name" "0.85" "data/Model-Name.csv" "checkpoints/Model-Name"
```

### Train Baselines

```bash
# CARROT baselines
python -m main.baselines.carrot.train_carrot --model ...

# IRT baselines
python -m main.baselines.irt.train_irt --model ...
```

## Evaluation Metrics

- **Peak Accuracy**: Maximum performance achieved [0, 1]
- **AUDC (Area Under Deferral Curve)**: Integral of cost-performance curve
- **QNC (Query-Normalized Cost)**: Cost to match best single LLM performance
  - Normalized to [0, 1] using global min/max across all methods
  - Lower is better (0 = minimum cost, 1 = maximum cost)

## Configuration

Key parameters in code:

- **λ (lambda)**: Cost-quality tradeoff [0, 1e-4]
  - λ=0: Maximize quality (ignore cost)
  - λ→1e-4: Minimize cost (preserve quality)
- **LLM Pool**: Available models for routing
- **Token Limits**: Budget levels to consider
- **Train/Test Split**: Default 80/20 with seed=42

## Data Format

Not included in repository due to size. To use your own data:

1. **CSV files** (`data/{model}.csv`):
   - Columns: `prompts_id`, `key`, `original_prompt`
   - Performance: `{limit}_score` (0-1 scores)
   - Token usage: `{limit}_count`

2. **Embeddings** (`data/prompt_embeddings.pkl`):
   - Numpy array: shape (N, 768)
   - Generated using sentence-transformers

## Citation

If you use this code, please cite our paper:

```bibtex
@article{core2025,
  title={CoRE: Constrained Response Evaluator for Intelligent LLM Routing},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## License

[Specify your license here - e.g., MIT, Apache 2.0]

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue.

## Acknowledgments

This project builds on:
- **SPROUT benchmark** for query diversity
- **vLLM** for efficient LLM serving
- **OpenRouter** for API access to multiple LLMs
