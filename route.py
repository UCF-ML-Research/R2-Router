#!/usr/bin/env python3
"""
R2-Router: Route a query to the optimal (LLM, token_budget) pair and generate a response.

Examples:
    # Route only
    python route.py --query "What is the capital of France?" \
        --embed-url http://localhost:8000

    # Route + generate via OpenRouter
    python route.py --query "What is the capital of France?" \
        --embed-url http://localhost:8000 \
        --llm-api-base https://openrouter.ai/api/v1 \
        --llm-api-key sk-or-...

    # Batch from file
    python route.py --file queries.txt --embed-url http://localhost:8000
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="R2-Router: Route queries to optimal (LLM, token_budget) and generate responses",
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--query", type=str, help="Single query text")
    input_group.add_argument("--file", type=str, help="File with one query per line")

    # Checkpoint source
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Local dir with config.json + checkpoints/ (default: ./r2_router)")
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HuggingFace repo ID to download checkpoints")

    # Embedding
    parser.add_argument("--embed-url", type=str, default=None,
                        help="vLLM embedding server (e.g., http://localhost:8000)")

    # LLM generation (optional)
    parser.add_argument("--llm-api-base", type=str, default=None,
                        help="OpenAI-compatible API base (e.g., https://openrouter.ai/api/v1)")
    parser.add_argument("--llm-api-key", type=str, default=None,
                        help="API key for LLM endpoint")

    # Routing params
    parser.add_argument("--lambda_val", type=float, default=0.5,
                        help="Cost-quality tradeoff (0=quality, 1=cost, default: 0.5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all candidate options ranked by risk")

    args = parser.parse_args()

    # Resolve checkpoint path
    if args.hf_repo:
        ckpt_path = args.hf_repo
    elif args.checkpoint_dir:
        ckpt_path = args.checkpoint_dir
    else:
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r2_router")
        if os.path.isdir(default_dir) and os.path.exists(os.path.join(default_dir, "config.json")):
            ckpt_path = default_dir
        else:
            print(
                "Error: No checkpoint directory found.\n"
                "  Use --checkpoint-dir, --hf-repo, or place checkpoints in ./r2_router/",
                file=sys.stderr,
            )
            sys.exit(1)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from r2_router import R2Router

    router = R2Router.from_pretrained(
        ckpt_path,
        lambda_val=args.lambda_val,
        embed_url=args.embed_url,
        llm_api_base=args.llm_api_base,
        llm_api_key=args.llm_api_key,
    )

    # Load queries
    if args.query:
        queries = [args.query]
    else:
        with open(args.file) as f:
            queries = [line.strip() for line in f if line.strip()]

    # Route (and optionally generate)
    do_generate = args.llm_api_base is not None

    for query in queries:
        if do_generate:
            result = router.route_and_generate(query)
            output = {
                "query": result["query"],
                "model": result["model"],
                "openrouter_id": result["openrouter_id"],
                "budget": result["budget"],
                "predicted_quality": result["predicted_quality"],
                "risk": result["risk"],
                "response": result["response"],
                "usage": result["usage"],
            }
        else:
            result = router.route_text(query)
            output = {
                "query": query,
                "model": result["model"],
                "openrouter_id": result["openrouter_id"],
                "budget": result["budget"],
                "predicted_quality": result["predicted_quality"],
                "predicted_cost_usd": result["predicted_cost_usd"],
                "risk": result["risk"],
            }
            if args.verbose:
                output["all_options"] = result["all_options"]

        print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
