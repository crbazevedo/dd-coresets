#!/usr/bin/env python3
"""
Categoria 7.2: Varying Cluster Separability

Objetivo: Testar DDC com diferentes níveis de separação
Hipótese: DDC mantém vantagem mesmo com clusters menos separados
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
    RANDOM_STATE
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_clusters_varying_separability(n_samples=20_000, n_features=10, n_clusters=4, 
                                          separability_multipliers=[0.5, 1.0, 2.0, 5.0], 
                                          random_state=None):
    """
    Generate Gaussian mixtures with varying cluster separability.
    
    separability_multipliers: Multipliers for cluster_std to control separation
    """
    if random_state is None:
        random_state = RANDOM_STATE
    
    datasets = {}
    
    for sep_mult in separability_multipliers:
        # cluster_std controls separation (smaller = more separated)
        # We use 1.0 / sep_mult so larger multiplier = more separation
        cluster_std = 1.0 / sep_mult
        
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=cluster_std,
            random_state=random_state,
        )
        
        datasets[f'sep_{sep_mult}x'] = (X, y, sep_mult)
    
    return datasets


def run_experiment():
    """Run varying separability experiment."""
    print("=" * 70)
    print("Experiment 7.2: Varying Cluster Separability")
    print("=" * 70)
    
    # Parameters
    n_samples = 20_000
    n_features = 10
    n_clusters = 4
    k_reps = 1000
    separability_multipliers = [0.5, 1.0, 2.0, 5.0]  # 0.5x = less separated, 5x = well separated
    
    # Generate datasets
    print(f"\nGenerating datasets with varying separability...")
    print(f"  n_samples={n_samples:,}, n_features={n_features}, n_clusters={n_clusters}")
    print(f"  Separability multipliers: {separability_multipliers}")
    
    datasets = generate_clusters_varying_separability(
        n_samples, n_features, n_clusters, separability_multipliers, random_state=RANDOM_STATE
    )
    
    # Run experiments
    all_results = []
    
    for sep_name, (X, y, sep_mult) in datasets.items():
        print(f"\n{'='*70}")
        print(f"Testing separability: {sep_mult}x")
        print(f"{'='*70}")
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit coresets
        print(f"  Fitting coresets (k={k_reps})...")
        S_random, w_random = fit_random_coreset(X_scaled, k_reps, random_state=RANDOM_STATE)
        S_ddc, w_ddc, _ = fit_ddc_coreset_optimized(X_scaled, k_reps, n0=None, random_state=RANDOM_STATE)
        
        # Compute metrics
        metrics_random = compute_all_metrics(X_scaled, S_random, w_random, 'Random')
        metrics_ddc = compute_all_metrics(X_scaled, S_ddc, w_ddc, 'DDC')
        
        # Coverage
        coverage_random = compute_spatial_coverage(X_scaled, S_random, labels_full=y)
        coverage_ddc = compute_spatial_coverage(X_scaled, S_ddc, labels_full=y)
        
        n_clusters_covered_random = len([k for k, v in coverage_random['coverage_per_cluster'].items() if v > 0])
        n_clusters_covered_ddc = len([k for k, v in coverage_ddc['coverage_per_cluster'].items() if v > 0])
        
        improvement_cov = (metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100
        improvement_w1 = (metrics_random['W1_mean'] / metrics_ddc['W1_mean'] - 1) * 100
        
        print(f"  Cov Error - Random: {metrics_random['cov_err_fro']:.4f}, DDC: {metrics_ddc['cov_err_fro']:.4f} ({improvement_cov:+.1f}%)")
        print(f"  W1 Mean - Random: {metrics_random['W1_mean']:.4f}, DDC: {metrics_ddc['W1_mean']:.4f} ({improvement_w1:+.1f}%)")
        print(f"  Clusters covered - Random: {n_clusters_covered_random}/{n_clusters}, DDC: {n_clusters_covered_ddc}/{n_clusters}")
        
        all_results.append({
            'separability': sep_mult,
            'random_cov_err': metrics_random['cov_err_fro'],
            'ddc_cov_err': metrics_ddc['cov_err_fro'],
            'cov_improvement_%': improvement_cov,
            'random_w1': metrics_random['W1_mean'],
            'ddc_w1': metrics_ddc['W1_mean'],
            'w1_improvement_%': improvement_w1,
            'random_clusters_covered': n_clusters_covered_random,
            'ddc_clusters_covered': n_clusters_covered_ddc,
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    results_df = pd.DataFrame(all_results)
    print("\nResults by Separability:")
    print(results_df.to_string(index=False))
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(results_dir / "varying_separability_summary.csv", index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.plot(results_df['separability'], results_df['cov_improvement_%'], 'o-', label='Cov Improvement', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Separability Multiplier')
    ax.set_ylabel('Covariance Improvement (%)')
    ax.set_title('DDC Covariance Improvement vs Cluster Separability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(results_df['separability'], results_df['w1_improvement_%'], 'o-', label='W1 Improvement', linewidth=2, color='orange')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Separability Multiplier')
    ax.set_ylabel('W1 Improvement (%)')
    ax.set_title('DDC W1 Improvement vs Cluster Separability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "varying_separability.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 70)
    print("Experiment completed!")
    print("=" * 70)
    
    return results_df


if __name__ == "__main__":
    run_experiment()

