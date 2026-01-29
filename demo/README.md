# CoRE Router Demo

Interactive web demo for testing the CoRE (Constrained Response Evaluator) routing system against CARROT baselines.

## Overview

This demo allows users to:
- Submit queries and see which LLM CoRE selects
- Compare CoRE against CARROT-KNN and CARROT-Linear baselines
- Adjust the cost-quality tradeoff parameter (λ)
- Set budget constraints
- View actual vs predicted performance metrics

## Project Structure

```
demo/
├── app.py              # Main Gradio application
├── config.py           # Configuration (LLM pool, paths, API keys)
├── embedder.py         # Query embedding module
├── router.py           # CoRE routing logic
├── baselines.py        # CARROT baseline routing
├── llm_client.py       # OpenRouter API client
├── judge.py            # Response evaluation module
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd demo
pip install -r requirements.txt
```

### 2. Set OpenRouter API Key

Get your API key from [OpenRouter](https://openrouter.ai/) and set it as an environment variable:

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key-here"
```

Or add it to your `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export OPENROUTER_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Configure LLM Pool

Edit `config.py` to configure which LLMs are available:

```python
LLM_POOL = {
    "GLM-4.5-Air": {
        "size": 0.85,
        "openrouter_id": "01-ai/yi-large",  # Update with correct model ID
        "checkpoint": CHECKPOINTS_DIR / "GLM-4.5-Air_ridge_alpha10.0",
        "csv": DATA_DIR / "GLM-4.5-Air.csv"
    },
    # Add more LLMs...
}
```

**Important:** Update the `openrouter_id` fields with the correct OpenRouter model IDs for your LLMs.

### 4. Verify Checkpoints

Make sure your trained CoRE and CARROT checkpoints exist:

```bash
# Check CoRE checkpoints
ls -l ../checkpoints/GLM-4.5-Air_ridge_alpha10.0/
# Should see: limited_score_predictors.joblib, unlimited_score_predictor.joblib, unlimited_token_predictor.joblib

# Check CARROT checkpoints
ls -l ../checkpoints/carrot_knn/
ls -l ../checkpoints/carrot_linear/
```

If checkpoints don't exist, train them first:

```bash
cd ..
bash run_experiment.sh
```

### 5. Run the Demo

```bash
python app.py
```

The demo will launch at `http://localhost:7860`

## Configuration Options

### Mock Mode (Testing Without API/Checkpoints)

To test the demo without real API calls or trained checkpoints:

1. Edit `config.py`:
```python
ENABLE_MOCK_MODE = True
```

2. Run the demo:
```bash
python app.py
```

Mock mode uses:
- Deterministic fake embeddings
- Mock LLM responses
- Mock routing decisions
- Heuristic judge scores

### Judge LLM

Configure which model evaluates responses in `config.py`:

```python
JUDGE_MODEL = "openai/gpt-4o-mini"  # Fast and cheap
# Or:
# JUDGE_MODEL = "openai/gpt-4o"  # Best quality
# JUDGE_MODEL = "anthropic/claude-3.5-sonnet"  # Good balance
```

### Embedding Model

Configure the embedding model in `config.py`:

```python
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # Fast (384 dim)
# Or:
# SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"  # Better quality (768 dim)
```

**Important:** The embedding dimension must match your training data!

## Usage Guide

### Basic Usage

1. **Enter a query** in the text box
2. **Adjust Lambda (λ)**:
   - λ = 0.0: Prioritize quality (may be expensive)
   - λ = 0.5: Balanced
   - λ = 1.0: Prioritize cost (cheapest option)
3. **Set budget** (optional): Maximum cost in tokens × model_size
4. **Select methods** to compare (CoRE, CARROT-KNN, CARROT-Linear)
5. **Click "Route Query"**

### Understanding Results

The demo shows:

**Comparison Table:**
- Selected LLM and token limit for each method
- Actual score from judge (0-1, higher is better)
- Actual cost (tokens × model_size)
- Cost-effectiveness (score / cost)
- Winner highlighted in green

**Detailed Cards:**
- Full LLM response
- Predicted vs actual scores and costs
- Judge reasoning
- Prediction errors (MAE)

### Example Scenarios

**Scenario 1: Quality-Focused (λ = 0)**
- Demo will select larger, more capable models
- Higher costs but better responses

**Scenario 2: Cost-Focused (λ = 1.0)**
- Demo will select smaller, cheaper models
- Lower costs, potentially lower quality

**Scenario 3: Budget Constrained**
- Set budget = 5.0
- Only options with cost ≤ 5.0 will be considered
- Forces cheaper models/token limits

## Deployment

### Local Network Access

To allow access from other devices on your network:

```bash
# app.py already configured with server_name="0.0.0.0"
python app.py
```

Access from other devices: `http://your-ip:7860`

### Public Link (Temporary)

To create a public link (via Gradio tunneling):

Edit `app.py`:
```python
demo.launch(
    share=True,  # Enable public link
    ...
)
```

You'll get a `https://xxxxx.gradio.live` link that works for 72 hours.

### Production Deployment

For permanent deployment, consider:

**Option 1: Docker**
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

```bash
docker build -t core-router-demo .
docker run -p 7860:7860 -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY core-router-demo
```

**Option 2: Hugging Face Spaces**
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Upload all demo files
3. Add `OPENROUTER_API_KEY` to Space secrets
4. Your demo will be live at `https://huggingface.co/spaces/your-username/core-router`

**Option 3: Cloud (AWS/GCP/Azure)**
- Deploy on EC2/Compute Engine/Azure VM
- Use HTTPS with nginx reverse proxy
- Add authentication if needed

## Troubleshooting

### Error: "No models loaded"

**Cause:** Checkpoint directories don't exist

**Fix:**
1. Check `config.py` paths
2. Train models: `cd .. && bash run_experiment.sh`
3. Or enable mock mode: `ENABLE_MOCK_MODE = True` in config.py

### Error: "OpenRouter API key not set"

**Cause:** Environment variable not set

**Fix:**
```bash
export OPENROUTER_API_KEY="your-key"
# Then restart the app
```

### Error: "sentence-transformers not installed"

**Cause:** Missing dependency

**Fix:**
```bash
pip install sentence-transformers torch
```

### Slow Performance

**Causes:**
- First query loads embedding model (one-time delay)
- LLM API latency
- CPU-only embedding (no GPU)

**Optimizations:**
1. Use GPU for embeddings (if available):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. Use faster embedding model:
   ```python
   SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # Fastest
   ```

3. Cache embeddings (implement in `embedder.py`)

### Wrong Model IDs

**Symptom:** LLM calls fail with "model not found"

**Cause:** Incorrect OpenRouter model IDs in `config.py`

**Fix:** Check [OpenRouter Models](https://openrouter.ai/docs#models) for correct IDs

## API Cost Estimates

Typical costs per query (with OpenRouter pricing):

| Component | Model | Cost per Query |
|-----------|-------|----------------|
| Judge | gpt-4o-mini | ~$0.0001 |
| LLM (small) | Llama-3.2-3B | ~$0.0001 |
| LLM (medium) | Llama-3.1-70B | ~$0.001 |
| LLM (large) | GPT-4 | ~$0.01 |
| **Total per comparison** | | **$0.001 - $0.03** |

Budget accordingly for demos!

## Advanced Configuration

### Custom Judge Prompt

Edit `config.py`:

```python
JUDGE_PROMPT_TEMPLATE = """
Your custom evaluation prompt...

Query: {query}
Response: {response}

Return JSON: {{"score": <0-1>, "reasoning": "<text>"}}
"""
```

### Adding New LLMs

1. Add to `config.LLM_POOL`:
```python
"New-Model": {
    "size": 0.25,  # Relative size for cost calculation
    "openrouter_id": "provider/model-name",
    "checkpoint": CHECKPOINTS_DIR / f"New-Model_{TRAINING_SCHEME}",
    "csv": DATA_DIR / "New-Model.csv"
}
```

2. Train CoRE model for it:
```bash
cd ..
# Add to run_experiment.sh LLM_POOL
bash run_experiment.sh
```

3. Retrain CARROT (automatically triggered by pool change)

### Batch Mode (Future Enhancement)

To process multiple queries:

1. Modify `app.py` to accept multiple queries
2. Process in parallel using `asyncio`
3. Show aggregate statistics

## Development

### Testing Individual Modules

Each module can be tested standalone:

```bash
# Test embedder
python embedder.py

# Test LLM client
python llm_client.py

# Test judge
python judge.py

# Test router
python router.py

# Test baselines
python baselines.py
```

### Adding Features

Common enhancements:

1. **Query History**: Store queries and results in SQLite
2. **Batch Processing**: Process multiple queries at once
3. **Visualizations**: Add charts (plotly/matplotlib)
4. **Export Results**: Download as CSV/JSON
5. **User Accounts**: Add authentication
6. **A/B Testing**: Compare different λ values automatically

## License

Same as main CoRE project.

## Citation

If you use this demo in research:

```bibtex
@software{core_router_demo,
  title = {CoRE Router Demo},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/...}
}
```

## Support

For issues:
1. Check this README troubleshooting section
2. Check `DEMO_REQUIREMENTS.md` in parent directory
3. Open an issue on GitHub

---

**Happy Routing! 🚀**
