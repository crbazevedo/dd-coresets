#!/usr/bin/env python3
"""
Categoria 10.1: Rare but Important Clusters

Objetivo: Clusters raros mas importantes (1% do dataset)
Hipótese: DDC garante representação de clusters raros
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


def generate_rare_clusters(n_samples=20_000, n_features=10, rare_ratio=0.01, random_state=None):
    """
    Generate dataset with rare but important cluster.
    
    Structure:
    - 3 common clusters (balanced)
    - 1 rare cluster (1% of data) but important
    """
    if random_state is None:
        random_state = RANDOM_STATE
    
    rng = np.random.RandomState(random_state)
    
    # Common clusters (well separated)
    n_common = 3
    n_common_samples = int((1 - rare_ratio) * n_samples)
    samples_per_common = n_common_samples // n_common
    
    # Rare cluster
    n_rare_samples = n_samples - n_common_samples
    
    # Generate common clusters
    X_common, labels_common = make_blobs(
        n_samples=n_common_samples,
        n_features=n_features,
        centers=n_common,
        cluster_std=0.8,
        random_state=random_state,
    )
    
    # Generate rare cluster (far from common clusters)
    rare_center = np.zeros(n_features)
    rare_center[0] = 8.0  # Far from common clusters
    
    X_rare = rng.randn(n_rare_samples, n_features) * 0.5 + rare_center
    labels_rare = np.full(n_rare_samples, n_common)  # Label = 3
    
    # Combine
    X = np.vstack([X_common, X_rare])
    labels = np.hstack([labels_common, labels_rare])
    
    # Shuffle
    indices = np.arange(len(X))
    rng.shuffle(indices)
    X = X[indices]
    labels = labels[indices]
    
    return X, labels


def run_experiment():
    """Run rare clusters experiment."""
    print("=" * 70)
    print("Experiment 10.1: Rare but Important Clusters")
    print("=" * 70)
    
    # Parameters
    n_samples = 20_000
    n_features = 10
    k_reps = 1000
    rare_ratio = 0.01  # 1% rare cluster
    
    # Generate data
    print(f"\nGenerating rare clusters dataset...")
    print(f"  n_samples={n_samples:,}, n_features={n_features}")
    print(f"  Structure: 3 common clusters + 1 rare cluster ({rare_ratio:.1%} of data)")
    
    X, labels = generate_rare_clusters(n_samples, n_features, rare_ratio, random_state=RANDOM_STATE)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check rare cluster presence
    rare_label = len(np.unique(labels)) - 1
    rare_mask = (labels == rare_label)
    print(f"\n  Rare cluster: {np.sum(rare_mask):,} samples ({np.sum(rare_mask)/len(X):.2%})")
    
    # Fit coresets
    print(f"\nFitting coresets (k={k_reps})...")
    
    print("  Random sampling...")
    S_random, w_random = fit_random_coreset(X_scaled, k_reps, random_state=RANDOM_STATE)
    
    print("  DDC coreset...")
    S_ddc, w_ddc, _ = fit_ddc_coreset_optimized(X_scaled, k_reps, n0=None, random_state=RANDOM_STATE)
    
    # Check rare cluster representation
    from scipy.spatial.distance import cdist
    
    rare_points = X_scaled[rare_mask]
    rare_center = rare_points.mean(axis=0)
    
    # Find closest points in coresets to rare cluster
    dists_random_to_rare = cdist(S_random, rare_points).min(axis=1)
    dists_ddc_to_rare = cdist(S_ddc, rare_points).min(axis=1)
    
    threshold = 2.0  # Distance threshold
    random_close_to_rare = np.sum(dists_random_to_rare < threshold)
    ddc_close_to_rare = np.sum(dists_ddc_to_rare < threshold)
    
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
    
    print(f"\nRare Cluster Representation:")
    print(f"  Random: {random_close_to_rare}/{k_reps} points close to rare cluster ({random_close_to_rare/k_reps:.1%})")
    print(f"  DDC:    {ddc_close_to_rare}/{k_reps} points close to rare cluster ({ddc_close_to_rare/k_reps:.1%})")
    
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
    
    print(f"\nCluster Coverage:")
    n_clusters = len(np.unique(labels))
    random_covered = len([k for k, v in coverage_random['coverage_per_cluster'].items() if v > 0])
    ddc_covered = len([k for k, v in coverage_ddc['coverage_per_cluster'].items() if v > 0])
    print(f"  Random: {random_covered}/{n_clusters} clusters")
    print(f"  DDC:    {ddc_covered}/{n_clusters} clusters")
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame([metrics_random, metrics_ddc])
    results_df.to_csv(results_dir / "rare_clusters_metrics.csv", index=False)
    
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
            title="Rare Clusters: Spatial Coverage",
            output_path=output_dir / "rare_clusters_spatial.png",
            labels_full=labels,
        )
    except Exception as e:
        print(f"  Warning: Could not generate spatial plot: {e}")
    
    # Marginals
    plot_marginal_distributions(
        X_scaled, S_random, w_random, S_ddc, w_ddc,
        n_features=min(5, n_features),
        title="Rare Clusters: Marginal Distributions",
        output_path=output_dir / "rare_clusters_marginals.png",
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
    ax.set_title('Rare Clusters: Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "rare_clusters_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Comparison table
    comparison_table = f"""
Rare Clusters Experiment Results
=================================

Dataset:
  n_samples: {n_samples:,}
  n_features: {n_features}
  Structure: 3 common clusters + 1 rare cluster ({rare_ratio:.1%})
  k: {k_reps}

Rare Cluster Representation:
  Random: {random_close_to_rare}/{k_reps} points close ({random_close_to_rare/k_reps:.1%})
  DDC:    {ddc_close_to_rare}/{k_reps} points close ({ddc_close_to_rare/k_reps:.1%})

Metrics Comparison:
                   Random      DDC        Improvement
Cov Error (Fro)    {metrics_random['cov_err_fro']:.4f}     {metrics_ddc['cov_err_fro']:.4f}     {improvement:+.1f}%
Corr Error (Fro)   {metrics_random['corr_err_fro']:.4f}     {metrics_ddc['corr_err_fro']:.4f}     {(metrics_random['corr_err_fro']/metrics_ddc['corr_err_fro']-1)*100:+.1f}%
W1 Mean            {metrics_random['W1_mean']:.4f}     {metrics_ddc['W1_mean']:.4f}     {improvement_w1:+.1f}%
W1 Max             {metrics_random['W1_max']:.4f}     {metrics_ddc['W1_max']:.4f}     {(metrics_random['W1_max']/metrics_ddc['W1_max']-1)*100:+.1f}%
KS Mean            {metrics_random['KS_mean']:.4f}     {metrics_ddc['KS_mean']:.4f}     {(metrics_random['KS_mean']/metrics_ddc['KS_mean']-1)*100:+.1f}%

Spatial Coverage:
  Clusters covered:
    Random: {random_covered}/{n_clusters}
    DDC:    {ddc_covered}/{n_clusters}
"""
    
    with open(results_dir / "rare_clusters_comparison_table.txt", 'w') as f:
        f.write(comparison_table)
    
    print("\n" + comparison_table)
    print("\n" + "=" * 70)
    print("Experiment completed!")
    print("=" * 70)
    
    return [metrics_random, metrics_ddc]


if __name__ == "__main__":
    run_experiment()

