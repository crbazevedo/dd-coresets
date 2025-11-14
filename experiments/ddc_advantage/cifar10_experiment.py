#!/usr/bin/env python3
"""
Categoria 12.1: CIFAR-10 Experiment

Objetivo: Dataset real com classes bem definidas (vs MNIST complexo)
Hipótese: DDC funciona melhor quando classes são bem separadas
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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


def load_cifar10(n_samples=10_000, n_components=50, random_state=None):
    """
    Load CIFAR-10 dataset.
    
    Falls back to synthetic data if CIFAR-10 is not available.
    """
    if random_state is None:
        random_state = RANDOM_STATE
    
    try:
        # Try to load CIFAR-10
        from tensorflow import keras
        from tensorflow.keras.datasets import cifar10
        
        print("Loading CIFAR-10...")
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
        # Combine train and test
        X_full = np.vstack([X_train, X_test])
        y_full = np.hstack([y_train.flatten(), y_test.flatten()])
        
        # Flatten images
        X_flat = X_full.reshape(X_full.shape[0], -1).astype(np.float32)
        
        # Sample subset
        if len(X_flat) > n_samples:
            indices = np.random.RandomState(random_state).choice(
                len(X_flat), size=n_samples, replace=False
            )
            X_flat = X_flat[indices]
            y_full = y_full[indices]
        
        # Reduce dimensionality with PCA
        print(f"  Reducing to {n_components} dimensions with PCA...")
        pca = PCA(n_components=n_components, random_state=random_state)
        X_reduced = pca.fit_transform(X_flat)
        
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"  Explained variance: {explained_var:.2%}")
        
        return X_reduced, y_full, True
        
    except ImportError:
        print("TensorFlow/Keras not available. Using synthetic CIFAR-10-like data...")
        # Fallback: Generate synthetic data with 10 well-separated clusters
        from sklearn.datasets import make_blobs
        
        X_synthetic, y_synthetic = make_blobs(
            n_samples=n_samples,
            n_features=n_components,
            centers=10,  # 10 classes like CIFAR-10
            cluster_std=1.5,
            random_state=random_state,
        )
        
        return X_synthetic, y_synthetic, False
    
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}")
        print("Using synthetic CIFAR-10-like data...")
        # Fallback: Generate synthetic data
        from sklearn.datasets import make_blobs
        
        X_synthetic, y_synthetic = make_blobs(
            n_samples=n_samples,
            n_features=n_components,
            centers=10,
            cluster_std=1.5,
            random_state=random_state,
        )
        
        return X_synthetic, y_synthetic, False


def run_experiment():
    """Run CIFAR-10 experiment."""
    print("=" * 70)
    print("Experiment 12.1: CIFAR-10")
    print("=" * 70)
    
    # Parameters
    n_samples = 10_000
    n_components = 50  # PCA components
    k_reps = 1000
    
    # Load data
    print(f"\nLoading CIFAR-10 dataset...")
    print(f"  n_samples={n_samples:,}, n_components={n_components}")
    
    X, y, is_real = load_cifar10(n_samples, n_components, random_state=RANDOM_STATE)
    
    if is_real:
        print("  Using real CIFAR-10 data")
    else:
        print("  Using synthetic CIFAR-10-like data (10 well-separated clusters)")
    
    print(f"  Final shape: {X.shape}")
    print(f"  Classes: {len(np.unique(y))}")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"    Class {cls}: {count:,} samples ({count/len(y):.1%})")
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit coresets
    print(f"\nFitting coresets (k={k_reps})...")
    
    print("  Random sampling...")
    S_random, w_random = fit_random_coreset(X_scaled, k_reps, random_state=RANDOM_STATE)
    
    print("  DDC coreset...")
    S_ddc, w_ddc, _ = fit_ddc_coreset_optimized(X_scaled, k_reps, n0=None, random_state=RANDOM_STATE)
    
    # Check class distribution in coresets
    from scipy.spatial.distance import cdist
    
    # Map coreset points back to original classes (approximate)
    distances_random = cdist(S_random, X_scaled)
    closest_random = distances_random.argmin(axis=1)
    y_random_coreset = y[closest_random]
    
    distances_ddc = cdist(S_ddc, X_scaled)
    closest_ddc = distances_ddc.argmin(axis=1)
    y_ddc_coreset = y[closest_ddc]
    
    print("\n  Class distribution in coresets:")
    print("    Random coreset:")
    unique_r, counts_r = np.unique(y_random_coreset, return_counts=True)
    for cls, count in zip(unique_r, counts_r):
        print(f"      Class {cls}: {count:,} ({count/k_reps:.1%})")
    
    print("    DDC coreset:")
    unique_d, counts_d = np.unique(y_ddc_coreset, return_counts=True)
    for cls, count in zip(unique_d, counts_d):
        print(f"      Class {cls}: {count:,} ({count/k_reps:.1%})")
    
    # Compute metrics
    print("\nComputing metrics...")
    
    metrics_random = compute_all_metrics(X_scaled, S_random, w_random, 'Random')
    metrics_ddc = compute_all_metrics(X_scaled, S_ddc, w_ddc, 'DDC')
    
    # Spatial coverage
    coverage_random = compute_spatial_coverage(X_scaled, S_random, labels_full=y)
    coverage_ddc = compute_spatial_coverage(X_scaled, S_ddc, labels_full=y)
    
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
    
    print(f"\nClass Coverage:")
    n_classes = len(np.unique(y))
    random_covered = len([k for k, v in coverage_random['coverage_per_cluster'].items() if v > 0])
    ddc_covered = len([k for k, v in coverage_ddc['coverage_per_cluster'].items() if v > 0])
    print(f"  Random: {random_covered}/{n_classes} classes")
    print(f"  DDC:    {ddc_covered}/{n_classes} classes")
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame([metrics_random, metrics_ddc])
    results_df.to_csv(results_dir / "cifar10_metrics.csv", index=False)
    
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
            title="CIFAR-10: Spatial Coverage",
            output_path=output_dir / "cifar10_spatial.png",
            labels_full=y,
        )
    except Exception as e:
        print(f"  Warning: Could not generate spatial plot: {e}")
    
    # Marginals
    plot_marginal_distributions(
        X_scaled, S_random, w_random, S_ddc, w_ddc,
        n_features=min(5, X_scaled.shape[1]),
        title="CIFAR-10: Marginal Distributions",
        output_path=output_dir / "cifar10_marginals.png",
    )
    
    # Metrics bar chart
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
    ax.set_title('CIFAR-10: Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "cifar10_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Comparison table
    comparison_table = f"""
CIFAR-10 Experiment Results
============================

Dataset:
  n_samples: {n_samples:,}
  n_features (after PCA): {n_components}
  n_classes: {n_classes}
  k: {k_reps}
  Data type: {'Real CIFAR-10' if is_real else 'Synthetic (10 clusters)'}

Class Coverage:
  Random: {random_covered}/{n_classes} classes
  DDC:    {ddc_covered}/{n_classes} classes

Metrics Comparison:
                   Random      DDC        Improvement
Cov Error (Fro)    {metrics_random['cov_err_fro']:.4f}     {metrics_ddc['cov_err_fro']:.4f}     {improvement:+.1f}%
Corr Error (Fro)   {metrics_random['corr_err_fro']:.4f}     {metrics_ddc['corr_err_fro']:.4f}     {(metrics_random['corr_err_fro']/metrics_ddc['corr_err_fro']-1)*100:+.1f}%
W1 Mean            {metrics_random['W1_mean']:.4f}     {metrics_ddc['W1_mean']:.4f}     {improvement_w1:+.1f}%
W1 Max             {metrics_random['W1_max']:.4f}     {metrics_ddc['W1_max']:.4f}     {(metrics_random['W1_max']/metrics_ddc['W1_max']-1)*100:+.1f}%
KS Mean            {metrics_random['KS_mean']:.4f}     {metrics_ddc['KS_mean']:.4f}     {(metrics_random['KS_mean']/metrics_ddc['KS_mean']-1)*100:+.1f}%
"""
    
    with open(results_dir / "cifar10_comparison_table.txt", 'w') as f:
        f.write(comparison_table)
    
    print("\n" + comparison_table)
    print("\n" + "=" * 70)
    print("Experiment completed!")
    print("=" * 70)
    
    return [metrics_random, metrics_ddc]


if __name__ == "__main__":
    run_experiment()

