#!/usr/bin/env python3
"""
Categoria 7.1: Nested Clusters (Clusters dentro de Clusters)

Objetivo: Demonstrar DDC em estrutura hierárquica
Hipótese: DDC captura melhor estrutura hierárquica que Random
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from experiments.ddc_advantage.utils import (
    fit_random_coreset, fit_ddc_coreset_optimized,
    compute_all_metrics, compute_spatial_coverage,
    plot_spatial_coverage_2d, plot_marginal_distributions,
    RANDOM_STATE
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')


def generate_nested_clusters(n_samples=20_000, n_features=10, random_state=None):
    """
    Generate nested clusters: large clusters containing sub-clusters.
    
    Structure:
    - 2 large clusters
    - Each large cluster contains 3 sub-clusters
    - Total: 6 sub-clusters
    """
    if random_state is None:
        random_state = RANDOM_STATE
    
    rng = np.random.RandomState(random_state)
    
    # Large cluster centers (far apart)
    large_centers = np.array([
        [-5, 0] + [0] * (n_features - 2),
        [5, 0] + [0] * (n_features - 2),
    ])
    
    # Sub-cluster centers within each large cluster
    sub_centers_per_large = 3
    sub_centers = []
    
    for large_idx, large_center in enumerate(large_centers):
        # Sub-clusters around large center
        for sub_idx in range(sub_centers_per_large):
            angle = 2 * np.pi * sub_idx / sub_centers_per_large
            offset = np.zeros(n_features)
            offset[0] = 1.5 * np.cos(angle)
            offset[1] = 1.5 * np.sin(angle)
            sub_centers.append(large_center + offset)
    
    sub_centers = np.array(sub_centers)
    
    # Generate points
    samples_per_sub = n_samples // len(sub_centers)
    X_list = []
    labels_list = []
    
    for sub_idx, sub_center in enumerate(sub_centers):
        large_idx = sub_idx // sub_centers_per_large
        
        # Generate points in sub-cluster
        X_sub = rng.randn(samples_per_sub, n_features) * 0.5 + sub_center
        X_list.append(X_sub)
        labels_list.append(np.full(samples_per_sub, sub_idx))
    
    X = np.vstack(X_list)
    labels = np.hstack(labels_list)
    
    # Shuffle
    indices = np.arange(len(X))
    rng.shuffle(indices)
    X = X[indices]
    labels = labels[indices]
    
    return X, labels


def run_experiment():
    """Run nested clusters experiment."""
    print("=" * 70)
    print("Experiment 7.1: Nested Clusters")
    print("=" * 70)
    
    # Parameters
    n_samples = 20_000
    n_features = 10
    k_reps = 1000
    
    # Generate data
    print(f"\nGenerating nested clusters dataset...")
    print(f"  n_samples={n_samples:,}, n_features={n_features}")
    print(f"  Structure: 2 large clusters, each with 3 sub-clusters (6 total)")
    
    X, labels = generate_nested_clusters(n_samples, n_features, random_state=RANDOM_STATE)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit coresets
    print(f"\nFitting coresets (k={k_reps})...")
    
    print("  Random sampling...")
    S_random, w_random = fit_random_coreset(X_scaled, k_reps, random_state=RANDOM_STATE)
    
    print("  DDC coreset...")
    S_ddc, w_ddc, _ = fit_ddc_coreset_optimized(X_scaled, k_reps, n0=None, random_state=RANDOM_STATE)
    
    # Compute metrics
    print("\nComputing metrics...")
    
    metrics_random = compute_all_metrics(X_scaled, S_random, w_random, 'Random')
    metrics_ddc = compute_all_metrics(X_scaled, S_ddc, w_ddc, 'DDC')
    
    # Spatial coverage
    coverage_random = compute_spatial_coverage(X_scaled, S_random, labels_full=labels)
    coverage_ddc = compute_spatial_coverage(X_scaled, S_ddc, labels_full=labels)
    
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    
    print(f"\nCovariance Error:")
    print(f"  Random: {metrics_random['cov_err_fro']:.4f}")
    print(f"  DDC:    {metrics_ddc['cov_err_fro']:.4f}")
    improvement = (metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100
    print(f"  DDC Improvement: {improvement:+.1f}%")
    
    print(f"\nWasserstein-1 Mean:")
    print(f"  Random: {metrics_random['W1_mean']:.4f}")
    print(f"  DDC:    {metrics_ddc['W1_mean']:.4f}")
    improvement_w1 = (metrics_random['W1_mean'] / metrics_ddc['W1_mean'] - 1) * 100
    print(f"  DDC Improvement: {improvement_w1:+.1f}%")
    
    print(f"\nSub-cluster Coverage:")
    n_clusters = len(np.unique(labels))
    random_covered = len([k for k, v in coverage_random['coverage_per_cluster'].items() if v > 0])
    ddc_covered = len([k for k, v in coverage_ddc['coverage_per_cluster'].items() if v > 0])
    print(f"  Random: {random_covered}/{n_clusters} clusters")
    print(f"  DDC:    {ddc_covered}/{n_clusters} clusters")
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame([metrics_random, metrics_ddc])
    results_df.to_csv(results_dir / "nested_clusters_metrics.csv", index=False)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Spatial coverage (2D projection)
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=RANDOM_STATE, n_jobs=1)
        X_2d = reducer.fit_transform(X_scaled)
        S_random_2d = reducer.transform(S_random)
        S_ddc_2d = reducer.transform(S_ddc)
        
        plot_spatial_coverage_2d(
            X_2d, S_random_2d, w_random, S_ddc_2d, w_ddc,
            title="Nested Clusters: Spatial Coverage",
            output_path=output_dir / "nested_clusters_spatial.png",
            labels_full=labels,
        )
    except Exception as e:
        print(f"  Warning: Could not generate spatial plot: {e}")
    
    # Marginals
    plot_marginal_distributions(
        X_scaled, S_random, w_random, S_ddc, w_ddc,
        n_features=min(5, n_features),
        title="Nested Clusters: Marginal Distributions",
        output_path=output_dir / "nested_clusters_marginals.png",
    )
    
    # Metrics bar chart (simple version)
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_to_plot = ['cov_err_fro', 'corr_err_fro', 'W1_mean', 'KS_mean']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    random_vals = [metrics_random[m] for m in metrics_to_plot]
    ddc_vals = [metrics_ddc[m] for m in metrics_to_plot]
    
    ax.bar(x - width/2, random_vals, width, label='Random', alpha=0.7)
    ax.bar(x + width/2, ddc_vals, width, label='DDC', alpha=0.7)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Nested Clusters: Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "nested_clusters_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Comparison table
    comparison_table = f"""
Nested Clusters Experiment Results
==================================

Dataset:
  n_samples: {n_samples:,}
  n_features: {n_features}
  Structure: 2 large clusters, each with 3 sub-clusters (6 total)
  k: {k_reps}

Metrics Comparison:
                   Random      DDC        Improvement
Cov Error (Fro)    {metrics_random['cov_err_fro']:.4f}     {metrics_ddc['cov_err_fro']:.4f}     {improvement:+.1f}%
Corr Error (Fro)   {metrics_random['corr_err_fro']:.4f}     {metrics_ddc['corr_err_fro']:.4f}     {(metrics_random['corr_err_fro']/metrics_ddc['corr_err_fro']-1)*100:+.1f}%
W1 Mean            {metrics_random['W1_mean']:.4f}     {metrics_ddc['W1_mean']:.4f}     {improvement_w1:+.1f}%
W1 Max             {metrics_random['W1_max']:.4f}     {metrics_ddc['W1_max']:.4f}     {(metrics_random['W1_max']/metrics_ddc['W1_max']-1)*100:+.1f}%
KS Mean            {metrics_random['KS_mean']:.4f}     {metrics_ddc['KS_mean']:.4f}     {(metrics_random['KS_mean']/metrics_ddc['KS_mean']-1)*100:+.1f}%

Spatial Coverage:
  Sub-clusters covered:
    Random: {random_covered}/{n_clusters}
    DDC:    {ddc_covered}/{n_clusters}
"""
    
    with open(results_dir / "nested_clusters_comparison_table.txt", 'w') as f:
        f.write(comparison_table)
    
    print("\n" + comparison_table)
    print("\n" + "=" * 70)
    print("Experiment completed!")
    print("=" * 70)
    
    return [metrics_random, metrics_ddc]


if __name__ == "__main__":
    run_experiment()

