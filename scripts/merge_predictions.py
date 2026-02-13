#!/usr/bin/env python3
"""
Merge generated_result from budget sweep files into the routed r2-router.json.

For each entry in r2-router.json, looks up the corresponding budget sweep file
based on (model, budget_label), finds the matching global_index, and copies
the generated_result.

Usage:
    python scripts/merge_predictions.py [--dry-run]
"""

import json
import os
import sys
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


SWEEP_ROOT = "/orange/qi855292.ucf/ah872032.ucf/budget_sweep"
PREDICTIONS_DIR = "/home/ah872032.ucf/jiaqi/RouterArena/router_inference/predictions"
ROUTER_FILE = os.path.join(PREDICTIONS_DIR, "r2-router.json")

# Map RouterArena model name -> sweep directory name
MODEL_DIR_MAP = {
    "qwen/qwen3-235b-a22b-2507": "235b",
    "qwen/qwen3-next-80b-a3b-instruct": "80b",
    "qwen/qwen3-30b-a3b-instruct-2507": "30b",
    "Qwen/Qwen3-Coder-Next": "Qwen3-Coder-Next",
    "qwen/qwen3-coder-30b-a3b-instruct": "coder-30b",
    "mistralai/ministral-3b": "ministral-3b",
    "mistralai/ministral-8b": "ministral-8b",
    "mistralai/ministral-14b": "ministral-14b",
    "gpt-4o": "gpt4o",
    "gemini-2.5-flash": "gemini-flash",
    "claude-3-haiku-20240307": "haiku",
}


def budget_label_to_filename(label: str) -> str:
    """Convert budget label from routing to sweep filename."""
    if label == "concise":
        return "concise.json"
    elif label == "unlimited":
        return "budget_unlimited.json"
    else:
        return f"budget_{label}.json"


def load_sweep_file(model_ra_name: str, budget_label: str) -> dict:
    """Load a sweep file and build global_index -> entry mapping.

    Returns: {global_index: entry} for regular (non-optimality) entries only.
    """
    model_dir = MODEL_DIR_MAP[model_ra_name]
    fname = budget_label_to_filename(budget_label)
    fpath = os.path.join(SWEEP_ROOT, model_dir, fname)

    with open(fpath) as f:
        data = json.load(f)

    # Index by global_index (regular entries only)
    index = {}
    for entry in data:
        if not entry.get("for_optimality", False):
            index[entry["global index"]] = entry

    return index


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write output, just report stats")
    parser.add_argument("--split", choices=["full", "robustness"], default="full",
                        help="Dataset split (default: full)")
    args = parser.parse_args()

    if args.split == "robustness":
        router_file = os.path.join(PREDICTIONS_DIR, "r2-router-robustness.json")
    else:
        router_file = ROUTER_FILE

    log(f"Loading routing from: {router_file}")
    with open(router_file) as f:
        router_data = json.load(f)
    log(f"  {len(router_data)} entries")

    # Group entries by (model, budget) to batch-load sweep files
    groups = defaultdict(list)  # (model, budget) -> [(idx, global_index), ...]
    for i, entry in enumerate(router_data):
        model = entry["prediction"]
        budget = entry.get("_r2_budget_label", "concise")
        groups[(model, budget)].append((i, entry["global index"]))

    log(f"  {len(groups)} unique (model, budget) combinations")

    # Process each group
    total_filled = 0
    total_missing = 0

    for (model, budget), entries in sorted(groups.items(), key=lambda x: -len(x[1])):
        sweep_index = load_sweep_file(model, budget)
        filled = 0
        missing = 0

        for idx, gidx in entries:
            sweep_entry = sweep_index.get(gidx)
            if sweep_entry is None:
                missing += 1
                continue

            gr = sweep_entry.get("generated_result")
            if gr is None or not gr.get("success"):
                missing += 1
                continue

            router_data[idx]["generated_result"] = gr
            filled += 1

        model_short = MODEL_DIR_MAP.get(model, model)
        log(f"  {model_short}/{budget}: {filled} filled, {missing} missing (of {len(entries)})")
        total_filled += filled
        total_missing += missing

    log(f"\nTotal: {total_filled} filled, {total_missing} missing")

    if total_missing > 0:
        log(f"WARNING: {total_missing} entries could not be filled!")

    if not args.dry_run:
        log(f"Saving to: {router_file}")
        with open(router_file, "w") as f:
            json.dump(router_data, f, ensure_ascii=False, indent=2)
        log("Done!")
    else:
        log("Dry run — no file written")


if __name__ == "__main__":
    main()
