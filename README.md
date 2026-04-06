# R2-Router

**R2-Router** introduces *reasoning* into LLM routing. Instead of treating each LLM as a fixed quality-cost point, R2-Router reasons about how quality varies with output length, jointly selecting the best LLM **and** token budget.

## How It Works

Given a query, R2-Router:
1. **Embeds** the query using Qwen3-0.6B (1024d)
2. **Predicts** quality at each (model, budget) using per-LLM Ridge regressors
3. **Routes** by maximizing: `risk = (1-λ) × quality - λ × cost`
4. **Generates** a response from the selected LLM with budget-constrained prompt

Each LLM has 17 Ridge regressors: 15 for limited budgets (10-4000 tokens) + 1 unlimited quality + 1 unlimited token count. Total: 11 models x 17 = 187 regressors, all shipping as 1.3MB of checkpoints.

## Installation

```bash
git clone -b release-routerarena-public https://github.com/jqxue1999/router.git
cd router
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python -e ".[embed]"
```

For compatibility with the released checkpoints, this package pins `scikit-learn==1.7.2`.

Checkpoints (1.3MB) are included in the repo — no separate download needed.

## Quick Start

### 1. Start embedding server

```bash
./.venv/bin/vllm serve Qwen/Qwen3-0.6B --runner pooling --port 8000
```

### 2. Route and generate

```python
from r2_router import R2Router

router = R2Router.from_pretrained(
    "./r2_router",
    embed_url="http://localhost:8000",              # Qwen3-0.6B embedding
    llm_api_base="https://openrouter.ai/api/v1",   # LLM API
    llm_api_key="sk-or-...",                        # your OpenRouter key
)

# End-to-end: embed -> route -> generate
result = router.route_and_generate("Write a Python function to calculate factorial.")
print(result["model"])      # e.g., "Qwen3-235B-A22B-Instruct-2507"
print(result["budget"])     # e.g., 100  (or "unlimited")
print(result["response"])   # LLM's answer
```

### Route only (no generation)

```python
decision = router.route_text("Solve the equation: 2x + 5 = 13")
print(decision["model"], decision["budget"])
# -> "Qwen2.5-Math-7B-Instruct", 200
```

### CLI

```bash
# Route only (human-readable output)
python route.py --query "Write a Python function to calculate factorial." --embed-url http://localhost:8000

# Output:
# Candidate LLMs:
# Qwen3-235B-A22B-Instruct-2507, GLM-4.5-Air, Llama-3.1-70B-Instruct, ...
#
# Selected LLM: Qwen3-235B-A22B-Instruct-2507
#
# Selected budget: 100

# Route + generate
python route.py --query "Write a Python function to calculate factorial." \
    --embed-url http://localhost:8000 \
    --llm-api-base https://openrouter.ai/api/v1 \
    --llm-api-key sk-or-...

# Output:
# Candidate LLMs:
# Qwen3-235B-A22B-Instruct-2507, GLM-4.5-Air, ...
#
# Selected LLM: Qwen3-235B-A22B-Instruct-2507
#
# Selected budget: 100
#
# ----------------------------------------
#
# Response:
# The capital of France is Paris.

# Adjust lambda (0=quality, 1=cost, default=0.99999)
python route.py --query "Solve the equation: 2x + 5 = 13" \
    --embed-url http://localhost:8000 \
    --llm-api-base https://openrouter.ai/api/v1 \
    --llm-api-key sk-or-... \
    --lambda_val 0.3

# Structured JSON output (for programmatic use)
python route.py --query "Write a Python function to calculate factorial." \
    --embed-url http://localhost:8000 --json

# Show all (model, budget) candidates ranked by risk
python route.py --query "Write a Python function to calculate factorial." \
    --embed-url http://localhost:8000 --verbose
```

## LLM Pool (11 models)

| Model | Input $/M | Output $/M | OpenRouter |
|-------|-----------|------------|------------|
| Qwen3-235B-A22B-Instruct-2507 | $0.071 | $0.10 | qwen/qwen3-235b-a22b-2507 |
| GLM-4.5-Air | $0.13 | $0.85 | z-ai/glm-4.5-air |
| Llama-3.1-70B-Instruct | $0.40 | $0.40 | meta-llama/llama-3.1-70b-instruct |
| Qwen2.5-Math-7B-Instruct | $0.10 | $0.10 | self-host* |
| Qwen2.5-Math-1.5B-Instruct | $0.04 | $0.04 | self-host* |
| gemma-3-4b-it | $0.04 | $0.08 | google/gemma-3-4b-it |
| Llama-3.2-3B-Instruct | $0.051 | $0.34 | meta-llama/llama-3.2-3b-instruct |
| Mistral-7B-Instruct-v0.2 | $0.11 | $0.19 | mistralai/mistral-7b-instruct |
| Qwen3-0.6B | $0.02 | $0.02 | self-host* |
| gemma-3-1b-it | $0.02 | $0.04 | self-host* |
| gemma-3-270m-it | $0.01 | $0.02 | self-host* |

*Models marked "self-host" are not on OpenRouter; prices are estimated. Edit `r2_router/config.json` to adjust.

Cost is computed as: `cost = input_tokens x input_price/1M + output_tokens x output_price/1M` (real USD).

The current online router uses absolute USD cost directly in
`risk = (1-λ) × quality - λ × cost`. Since `quality` is in `[0,1]` but
`cost` is usually much smaller in magnitude, meaningful cost-sensitive routing
typically requires `λ` to be very close to `1`. The default is therefore set to
`0.99999` instead of `0.5`.

## Architecture

```
query --> Qwen3-0.6B --> embedding (1024d)
              |
              v
         R2-Router (per-LLM Ridge regressors)
              |
              |-- For each (model, budget):
              |     quality = Ridge.predict(embedding)
              |     cost    = input_tokens x in_price + output_tokens x out_price
              |     risk    = (1-lambda) x quality - lambda x cost
              |
              v
         Best (model*, budget*) = argmax risk
              |
              v
         Call model* via OpenRouter with budget prompt
              |
              v
         Response
```

## Project Structure

```
r2-router/
├── route.py                     # CLI entry point
├── r2_router/                   # Core package (self-contained)
│   ├── __init__.py
│   ├── router.py                  # R2Router class
│   ├── config.json                # 11 models, prices, OpenRouter IDs
│   └── checkpoints/               # 11 models x Ridge regressors (~1.2MB total)
│       ├── Qwen3-235B-A22B-Instruct-2507_ridge_alpha10.0/
│       │   ├── limited_score_predictors.joblib    # 15 budget predictors
│       │   ├── unlimited_score_predictor.joblib   # unlimited quality
│       │   └── unlimited_token_predictor.joblib   # unlimited token count
│       └── ...
└── pyproject.toml
```

## R2-Bench Dataset

The training data for R2-Router is available as the **R2-Bench** dataset:

- [R2-Bench on RouterArena](https://huggingface.co/datasets/JiaqiXue/R2-Bench-RouterArena)
- [R2-Bench for paper](https://huggingface.co/datasets/JiaqiXue/R2-Bench)

R2-Bench contains 30,968 queries evaluated across 10 LLMs at 16 token budget levels (10-8000 tokens), with LLM-judge quality scores. Each evaluation includes the original prompt, LLM response, actual token count, and judge correctness score (0.0-1.0).

## Citation

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
