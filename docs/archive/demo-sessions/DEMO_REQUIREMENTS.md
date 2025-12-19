# CoRE Router Demo - Project Requirements

## Project Overview

Build an interactive web demo that allows users to test the CoRE (Constrained Response Evaluator) routing system against CARROT baselines. Users submit queries with cost constraints, and the system intelligently selects the best LLM to answer, then evaluates the response quality.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web Frontend                             │
│  User inputs: Query(s), Cost Budget, Lambda (λ)                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend API Server                          │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Query Embedding                                        │  │
│  │    - Convert query to embedding vector                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                        │                                         │
│                        ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2. Router Selection (Parallel)                           │  │
│  │                                                           │  │
│  │    ┌─────────────────────┐   ┌─────────────────────┐    │  │
│  │    │  CoRE Router        │   │  CARROT Baseline    │    │  │
│  │    ├─────────────────────┤   ├─────────────────────┤    │  │
│  │    │ • Load checkpoints  │   │ • KNN Baseline      │    │  │
│  │    │ • Predict scores    │   │ • Linear Baseline   │    │  │
│  │    │ • Predict costs     │   │ • Route with λ      │    │  │
│  │    │ • Apply λ & budget  │   │ • Apply budget      │    │  │
│  │    │ • Select best LLM   │   │ • Select best LLM   │    │  │
│  │    └─────────────────────┘   └─────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                        │                                         │
│                        ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 3. LLM Inference                                          │  │
│  │    - Forward query to selected LLM(s)                     │  │
│  │    - Get responses                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                        │                                         │
│                        ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 4. Response Evaluation                                    │  │
│  │    - Judge LLM evaluates response quality                 │  │
│  │    - Extract correctness score (0-1)                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                        │                                         │
└────────────────────────┼─────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Display Results                               │
│  - Selected LLM for each method                                  │
│  - LLM responses                                                 │
│  - Quality scores from judge                                     │
│  - Cost breakdown                                                │
│  - Side-by-side comparison                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Functional Requirements

### 1. User Input Interface

**Query Input:**
- [ ] Single query input (text area)
- [ ] Multiple queries input (batch mode)
- [ ] Support for up to N queries per submission (define N, e.g., 10)

**Cost Control:**
- [ ] Lambda (λ) slider: Range [0.0, 1.0]
  - λ = 0: Prioritize quality (accuracy)
  - λ = 1: Prioritize cost (cheapest)
  - Default: 0.5 (balanced)
  - Step: 0.1 or continuous
- [ ] Cost budget input: Optional maximum cost constraint
  - Units: Token count or USD equivalent
  - Default: Unlimited
  - When set: Only consider LLMs within budget

**Router Selection:**
- [ ] Method selection checkboxes:
  - [x] CoRE (default checked)
  - [x] CARROT-KNN
  - [x] CARROT-Linear
  - [ ] Oracle (optional, if you want to show theoretical best)

### 2. Query Embedding Service

**Requirements:**
- [ ] Convert user query to embedding vector
- [ ] Use same embedding model as training (consistency critical)
- [ ] Handle batch embedding for multiple queries
- [ ] Cache embeddings for repeated queries (optional optimization)

**Implementation Options:**
1. Local embedding model (e.g., sentence-transformers)
2. API-based embedding (e.g., OpenAI embeddings)
3. Pre-computed embedding lookup (if using fixed query set)

**Embedding Model:**
- Must match training: Which model did you use for `prompt_embeddings.pkl`?
- Dimension: Must match predictor input dimension

### 3. Routing Logic

#### CoRE Router

**Input:**
- Query embedding (vector)
- Lambda (λ) value
- Cost budget (optional)
- Available LLM pool with sizes

**Processing:**
1. Load trained CoRE checkpoints for each LLM in pool
2. For each LLM:
   - For each token limit (10, 20, 30, ..., unlimited):
     - Predict performance score: `s_{llm,limit}`
     - Predict token count: `c_{llm,limit}`
     - Calculate cost: `cost = c × model_size`
     - Calculate risk: `risk = (1-λ) × s - λ × cost`
     - Filter out if cost > budget
3. Select (LLM, token_limit) with maximum risk

**Output:**
- Selected LLM name
- Selected token limit
- Predicted score
- Predicted cost
- Risk value

#### CARROT Baseline

**Input:**
- Query embedding (vector)
- Lambda (λ) value
- Cost budget (optional)
- Method: "knn" or "linear"

**Processing:**
1. Load trained CARROT model (KNN or Linear)
2. Predict scores and costs for all (LLM, token_limit) combinations
3. Apply same risk calculation: `risk = (1-λ) × s - λ × cost`
4. Filter by budget, select maximum risk

**Output:**
- Selected LLM name
- Selected token limit
- Predicted score
- Predicted cost
- Risk value

### 4. LLM Inference Service

**Requirements:**
- [ ] Forward query to selected LLM(s)
- [ ] Support for token limit enforcement
- [ ] Handle API rate limits and errors
- [ ] Timeout handling (max wait time per query)

**LLM Pool:**
Which LLMs from your pool will be available in the demo?
- GLM-4.5-Air
- Llama-3.1-70B-Instruct
- Llama-3.2-3B-Instruct
- Qwen3-0.6B
- Qwen3-235B-A22B-Instruct
- gemma-3-4b-it
- ... (define subset)

**API Access:**
- [ ] API keys for each LLM provider
- [ ] Model endpoint URLs
- [ ] Token limit implementation (max_tokens parameter)
- [ ] Consistent prompt formatting

### 5. Response Evaluation (Judge LLM)

**Requirements:**
- [ ] Use a judge LLM to evaluate response quality
- [ ] Extract correctness score in range [0, 1]
- [ ] Consistent evaluation criteria across all responses

**Judge LLM Selection:**
Options:
1. GPT-4 (expensive but high quality)
2. Claude (good balance)
3. Qwen-Max (cost-effective)
4. Your existing judge from training data

**Evaluation Prompt:**
- Should match your training data evaluation methodology
- Extract score using `ultimate_json_score_extractor()` from utils.py
- Handle JSON parsing errors gracefully

**Judge Input:**
```
Query: {user_query}
Response: {llm_response}

Evaluate the correctness and quality of the response.
Return a JSON with:
{
  "score": <float 0-1>,
  "reasoning": "<explanation>"
}
```

### 6. Results Display

**Comparison View:**
- [ ] Side-by-side comparison table:

| Metric | CoRE | CARROT-KNN | CARROT-Linear |
|--------|------|------------|---------------|
| Selected LLM | GLM-4.5-Air | Llama-3.1-70B | Qwen3-0.6B |
| Token Limit | 100 | unlimited | 50 |
| Predicted Score | 0.85 | 0.82 | 0.78 |
| Actual Score (Judge) | 0.88 | 0.79 | 0.80 |
| Predicted Cost | 85 tokens | 450 tokens | 30 tokens |
| Actual Cost | 92 tokens | 467 tokens | 28 tokens |
| Risk Value | 0.425 | 0.185 | 0.610 |

**Response Display:**
- [ ] Full LLM response text for each method
- [ ] Highlight best method (highest actual score)
- [ ] Show cost-effectiveness metric

**Visualizations (Optional):**
- [ ] Cost vs Quality scatter plot
- [ ] Prediction accuracy (predicted vs actual)
- [ ] Budget utilization gauge

### 7. Batch Processing (Multiple Queries)

**Requirements:**
- [ ] Process multiple queries in one submission
- [ ] Aggregate statistics across queries
- [ ] Show per-query and overall performance

**Aggregate Metrics:**
- Average actual score
- Total cost
- Prediction accuracy (MAE between predicted and actual scores)
- Best method win rate (how often each method produces highest quality)

---

## Non-Functional Requirements

### Performance

- [ ] Query processing time: < 10 seconds per query (including LLM inference)
- [ ] Embedding generation: < 1 second
- [ ] Routing decision: < 100ms per method
- [ ] Support concurrent users (define: 10? 100?)

### Reliability

- [ ] Handle LLM API failures gracefully (timeout, rate limit, errors)
- [ ] Fallback to default LLM if routing fails
- [ ] Error messages user-friendly
- [ ] Validation for user inputs

### Scalability

- [ ] Cache loaded models (don't reload checkpoints per request)
- [ ] Async LLM inference (parallel requests to different LLMs)
- [ ] Request queuing if high load

### Security

- [ ] API key management (don't expose in frontend)
- [ ] Input sanitization (prevent injection attacks)
- [ ] Rate limiting per user (prevent abuse)
- [ ] CORS configuration for web frontend

---

## Technical Stack Recommendations

### Frontend

**Option 1: Simple (Gradio or Streamlit)**
```python
# Gradio demo
import gradio as gr

def route_and_evaluate(query, lambda_val, budget, methods):
    # Your routing logic
    return results_html

demo = gr.Interface(
    fn=route_and_evaluate,
    inputs=[
        gr.Textbox(label="Query", lines=3),
        gr.Slider(0, 1, value=0.5, label="Lambda (λ)"),
        gr.Number(label="Cost Budget (tokens, optional)"),
        gr.CheckboxGroup(["CoRE", "CARROT-KNN", "CARROT-Linear"],
                        value=["CoRE"], label="Methods")
    ],
    outputs=gr.HTML(label="Results")
)
demo.launch()
```

**Option 2: Full Web App (React + FastAPI)**
- Frontend: React, Vue, or Svelte
- Backend: FastAPI with async support
- Communication: REST API or WebSocket for streaming

### Backend

**Framework:**
- FastAPI (recommended for async LLM calls)
- Flask (simpler, but synchronous)

**Model Loading:**
```python
# Load once at startup, reuse across requests
from predictor_sklearn import TokenPerformancePredictor
from baselines_carrot import CarrotKNNBaseline, CarrotLinearBaseline

# Global model cache
CORE_MODELS = {}
CARROT_MODELS = {}

@app.on_event("startup")
async def load_models():
    for llm_name in LLM_POOL:
        checkpoint = f"./checkpoints/{llm_name}_ridge_alpha10.0"
        CORE_MODELS[llm_name] = TokenPerformancePredictor(load_dir=checkpoint)

    CARROT_MODELS["knn"] = CarrotKNNBaseline(load_dir="./checkpoints/carrot_knn")
    CARROT_MODELS["linear"] = CarrotLinearBaseline(load_dir="./checkpoints/carrot_linear")
```

**LLM API Integration:**
```python
import openai
from anthropic import Anthropic

async def call_llm(llm_name: str, query: str, token_limit: int):
    if llm_name.startswith("gpt"):
        # OpenAI API
        response = await openai.ChatCompletion.acreate(...)
    elif llm_name.startswith("claude"):
        # Anthropic API
        client = Anthropic(api_key=...)
        response = await client.messages.create(...)
    # ... other providers
    return response.text, actual_tokens_used
```

### Database (Optional)

**Purpose:**
- Store query history
- Cache embeddings
- Log routing decisions for analysis

**Options:**
- SQLite (simple, file-based)
- PostgreSQL (production-ready)
- Redis (fast caching)

---

## User Flow Example

### Example 1: Single Query

**User Input:**
```
Query: "What is the capital of France?"
Lambda: 0.3 (prioritize quality)
Budget: 200 tokens
Methods: [CoRE, CARROT-KNN]
```

**System Processing:**
1. Generate embedding for query
2. CoRE routing:
   - Evaluates all (LLM, token_limit) options
   - Selects: GLM-4.5-Air with 50 token limit
   - Predicted score: 0.95, predicted cost: 42 tokens, risk: 0.652
3. CARROT-KNN routing:
   - Evaluates all options
   - Selects: Llama-3.1-70B with 100 token limit
   - Predicted score: 0.93, predicted cost: 95 tokens, risk: 0.622
4. Call selected LLMs:
   - GLM-4.5-Air responds: "Paris"
   - Llama-3.1-70B responds: "The capital of France is Paris, which is also its largest city..."
5. Judge evaluation:
   - GLM-4.5-Air score: 1.0 (correct, concise)
   - Llama-3.1-70B score: 1.0 (correct, detailed)
6. Display results with comparison

**Output Display:**
```
╔════════════════════════════════════════════════════════════╗
║                    Routing Comparison                       ║
╠════════════════════════════════════════════════════════════╣
║ Query: "What is the capital of France?"                    ║
║ Lambda: 0.3 (Quality-focused)                              ║
║ Budget: 200 tokens                                         ║
╚════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────┐
│ CoRE (Our Method)                                         │
├──────────────────────────────────────────────────────────┤
│ Selected: GLM-4.5-Air (limit: 50 tokens)                 │
│ Response: "Paris"                                         │
│                                                           │
│ Predicted Score: 0.95    Actual Score: 1.00 ✓            │
│ Predicted Cost:  42      Actual Cost:  38                │
│ Risk Value:      0.652                                    │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ CARROT-KNN Baseline                                       │
├──────────────────────────────────────────────────────────┤
│ Selected: Llama-3.1-70B (limit: 100 tokens)              │
│ Response: "The capital of France is Paris, which is      │
│            also its largest city and a major European     │
│            center for art, fashion, and culture."         │
│                                                           │
│ Predicted Score: 0.93    Actual Score: 1.00 ✓            │
│ Predicted Cost:  95      Actual Cost:  89                │
│ Risk Value:      0.622                                    │
└──────────────────────────────────────────────────────────┘

Winner: CoRE (Better cost-effectiveness: 1.00 score / 38 tokens)
```

### Example 2: Batch Queries with Tight Budget

**User Input:**
```
Queries:
1. "Solve: 2x + 5 = 13"
2. "What is photosynthesis?"
3. "Write a haiku about spring"

Lambda: 0.7 (prioritize cost)
Budget: 50 tokens per query
Methods: [CoRE, CARROT-KNN, CARROT-Linear]
```

**System Processing:**
- Process 3 queries × 3 methods = 9 routing decisions
- 9 LLM API calls
- 9 judge evaluations
- Aggregate statistics

**Output Display:**
```
╔════════════════════════════════════════════════════════════╗
║              Batch Results (3 queries)                      ║
╚════════════════════════════════════════════════════════════╝

Overall Performance:

Method          Avg Score  Total Cost  Avg Pred Error  Win Rate
─────────────────────────────────────────────────────────────
CoRE               0.92      127        0.04           67% (2/3)
CARROT-KNN         0.88      135        0.07           33% (1/3)
CARROT-Linear      0.85      122        0.09           0% (0/3)

[Detailed per-query breakdown below...]
```

---

## Data Requirements

### Model Checkpoints

**CoRE:**
- Path: `./checkpoints/{LLM_name}_{training_scheme}/`
- Files needed per LLM:
  - `limited_score_predictors.joblib`
  - `unlimited_score_predictor.joblib`
  - `unlimited_token_predictor.joblib`

**CARROT:**
- Path: `./checkpoints/carrot_knn/` and `./checkpoints/carrot_linear/`
- Files needed:
  - `knn_score.joblib`, `knn_count.joblib`
  - `linear_score.joblib`, `linear_count.joblib`

### LLM Pool Configuration

**Define which LLMs are available:**
```python
LLM_POOL = [
    {"name": "GLM-4.5-Air", "size": 0.85, "api": "zhipu"},
    {"name": "Llama-3.1-70B", "size": 0.40, "api": "together"},
    {"name": "Qwen3-0.6B", "size": 0.0173, "api": "dashscope"},
    # ... add more
]

TOKEN_LIMITS = [10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000, "unlimited"]
```

### Embedding Model

**Which embedding model?**
- Sentence-BERT?
- OpenAI text-embedding-ada-002?
- Custom model?

**Must match training data embeddings!**

---

## Development Phases

### Phase 1: MVP (Minimum Viable Product)
- [ ] Single query input
- [ ] CoRE + CARROT-KNN only
- [ ] Fixed λ = 0.5, no budget constraint
- [ ] Simple Gradio interface
- [ ] 2-3 LLMs from pool
- [ ] Mock judge scores (or manual entry)

### Phase 2: Core Features
- [ ] Lambda slider
- [ ] Budget constraint
- [ ] All routing methods (CoRE, KNN, Linear)
- [ ] Real judge LLM integration
- [ ] Better UI with result comparison

### Phase 3: Advanced Features
- [ ] Batch query processing
- [ ] Result caching
- [ ] Query history
- [ ] Visualizations (charts, graphs)
- [ ] Export results (CSV, JSON)

### Phase 4: Production-Ready
- [ ] Full web app (React + FastAPI)
- [ ] User authentication
- [ ] Database for logging
- [ ] Performance monitoring
- [ ] Deployment (Docker, cloud hosting)

---

## Open Questions to Resolve

1. **Embedding Model:** Which embedding model did you use for training? (Need exact match)

2. **LLM APIs:** Which LLMs will be accessible via API for the demo?
   - Some models require specific providers
   - Need API keys and rate limits

3. **Judge LLM:** Which LLM should evaluate responses?
   - Should it match your training data judge?
   - Trade-off: quality vs cost vs speed

4. **Deployment:** Where will the demo be hosted?
   - Local (Gradio share link)
   - Cloud (AWS, GCP, Azure)
   - University server

5. **Budget Units:** Cost in tokens or USD?
   - Tokens easier to implement
   - USD more user-friendly

6. **Token Limits:** How to enforce in API calls?
   - `max_tokens` parameter (if supported)
   - Post-processing truncation

7. **Scale:** Expected number of users?
   - Affects infrastructure choices
   - Rate limiting strategy

---

## Success Metrics

**Demo should demonstrate:**
1. CoRE outperforms CARROT baselines on average (quality or cost-effectiveness)
2. Lambda control works: λ=0 picks expensive/good, λ=1 picks cheap/ok
3. Budget constraints are respected
4. Predictions are reasonably accurate (within 10-15% of actual)
5. User-friendly interface, easy to understand results

**Quantitative Goals:**
- Prediction MAE < 0.1 for scores
- Routing decision time < 100ms
- End-to-end response time < 10 seconds
- CoRE wins ≥ 60% of queries on quality or cost-effectiveness

---

## Next Steps

1. **Clarify open questions** (embedding model, LLM access, judge LLM)
2. **Start with Phase 1 MVP** (simple Gradio demo)
3. **Create demo project structure:**
   ```
   demo/
   ├── app.py              # Main Gradio/FastAPI app
   ├── router.py           # CoRE routing logic
   ├── baselines.py        # CARROT routing logic
   ├── llm_client.py       # LLM API wrapper
   ├── judge.py            # Response evaluation
   ├── embedder.py         # Query embedding
   ├── config.py           # LLM pool, paths, API keys
   └── requirements.txt    # Dependencies
   ```
4. **Implement core routing pipeline**
5. **Test with sample queries**
6. **Iterate based on feedback**

Would you like me to help you build any specific component of this demo?