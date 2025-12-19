#!/usr/bin/env python3
"""
Visualize embedding model ablation study results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_comparison(summary_df: pd.DataFrame, output_dir: str):
    """Generate comparison plots for embedding models."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.4)

    # Sort by AUDC
    summary_df = summary_df.sort_values('audc', ascending=False)

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # Plot 1: AUDC comparison
    ax1.barh(summary_df['embedding_model'], summary_df['audc'], color=colors)
    ax1.set_xlabel('AUDC (Normalized) ↑', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Embedding Model', fontsize=13, fontweight='bold')
    ax1.set_title('Routing Performance', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_axisbelow(True)

    # Add values on bars
    for i, v in enumerate(summary_df['audc']):
        ax1.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=11)

    # Plot 2: QNC comparison (lower is better)
    ax2.barh(summary_df['embedding_model'], summary_df['qnc'], color=colors)
    ax2.set_xlabel('QNC (Normalized) ↓', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Embedding Model', fontsize=13, fontweight='bold')
    ax2.set_title('Cost Efficiency', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_axisbelow(True)

    # Add values on bars
    for i, v in enumerate(summary_df['qnc']):
        ax2.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=11)

    # Plot 3: Peak Accuracy comparison
    ax3.barh(summary_df['embedding_model'], summary_df['peak_accuracy'], color=colors)
    ax3.set_xlabel('Peak Accuracy ↑', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Embedding Model', fontsize=13, fontweight='bold')
    ax3.set_title('Peak Performance', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.set_axisbelow(True)

    # Add values on bars
    for i, v in enumerate(summary_df['peak_accuracy']):
        ax3.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=11)

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'embedding_ablation_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")

    output_path_pdf = os.path.join(output_dir, 'embedding_ablation_comparison.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved PDF to {output_path_pdf}")

    plt.close()


def plot_curves(curves_df: pd.DataFrame, output_dir: str):
    """Plot cost-performance curves for all embedding models."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot curve for each embedding model
    for model in curves_df['embedding_model'].unique():
        model_data = curves_df[curves_df['embedding_model'] == model]

        # Sort by cost
        model_data = model_data.sort_values('cost')

        ax.plot(model_data['cost'], model_data['performance'],
                linewidth=2.5, marker='o', markersize=4, alpha=0.8,
                label=model)

    ax.set_xlabel('Cost ↓', fontsize=13, fontweight='bold')
    ax.set_ylabel('Performance (Accuracy) ↑', fontsize=13, fontweight='bold')
    ax.set_title('Cost-Performance Tradeoff Curves\n(Different Embedding Models)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, framealpha=0.95, loc='best')

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'embedding_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved curves plot to {output_path}")

    plt.close()


def main():
    """Main visualization pipeline."""
    print("=" * 80)
    print("Embedding Model Ablation Study - Visualization")
    print("=" * 80)

    # Load results
    results_dir = 'ablation/results'
    summary_df = pd.read_csv(os.path.join(results_dir, 'embedding_comparison.csv'))
    curves_df = pd.read_csv(os.path.join(results_dir, 'embedding_curves.csv'))

    print("\nEmbedding Model Comparison:")
    print(summary_df.to_string(index=False))

    # Generate plots
    print("\nGenerating visualizations...")
    plot_comparison(summary_df, results_dir)
    plot_curves(curves_df, results_dir)

    print("\n" + "=" * 80)
    print("Visualization completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
