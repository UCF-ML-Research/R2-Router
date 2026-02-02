#!/usr/bin/env python3
"""
Quick version of interpolation experiment for testing.
Tests fewer head counts for faster results.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from main.r2.experiment_interpolation import (
    run_experiment,
    plot_results
)


def main():
    """Run quick interpolation experiment."""
    print("=" * 80)
    print("R2-Router Interpolation Experiment (Quick Version)")
    print("=" * 80)

    # Use only ONE model for quick testing
    models = [
        ('GLM-4.5-Air', 0.85, 'data/GLM-4.5-Air.csv'),
    ]

    # Test fewer head counts: 2, 6, 10, 14, 16
    n_heads_list = [2, 6, 10, 14, 16]

    # Lambda range for routing
    lambda_range = np.linspace(0, 1e-4, 100)

    # Run experiment
    print(f"\nTesting {len(models)} model with {len(n_heads_list)} different head counts")
    results_df = run_experiment(
        models=models,
        n_heads_list=n_heads_list,
        lambda_range=lambda_range,
        strategy='uniform'
    )

    # Save results
    output_dir = 'comparison_results/interpolation_experiment'
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, 'interpolation_results_quick.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved results to {results_path}")

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Plot results
    plot_results(results_df, output_dir)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
