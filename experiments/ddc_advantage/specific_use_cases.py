"""
Categoria 6: Casos de Uso Específicos

Experimentos demonstrando vantagem do DDC em casos específicos:
- Preservação de outliers
- Preservação de regiões de baixa densidade
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from experiments.ddc_advantage.utils import (
    compute_all_metrics, fit_random_coreset, fit_ddc_coreset_optimized,
    plot_spatial_coverage_2d, compute_spatial_coverage,
    save_results, RANDOM_STATE
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_with_outliers(n_samples, n_features, outlier_fraction=0.05, outlier_distance=10.0, random_state=None):
    """Generate dataset with outliers."""
    if random_state is None:
        random_state = RANDOM_STATE
    
    rng = np.random.RandomState(random_state)
    
    # Main data (Gaussian)
    n_main = int(n_samples * (1 - outlier_fraction))
    X_main = rng.randn(n_main, n_features)
    
    # Outliers (far from main data)
    n_outliers = n_samples - n_main
    X_outliers = rng.randn(n_outliers, n_features) * outlier_distance
    
    X = np.vstack([X_main, X_outliers])
    labels = np.hstack([np.zeros(n_main), np.ones(n_outliers)])  # 0=normal, 1=outlier
    
    # Shuffle
    indices = np.arange(len(X))
    rng.shuffle(indices)
    X = X[indices]
    labels = labels[indices]
    
    return X, labels


def generate_low_density_clusters(n_samples, n_features, n_clusters, size_ratios=[1, 1, 10], random_state=None):
    """Generate mixture with clusters of very different sizes."""
    if random_state is None:
        random_state = RANDOM_STATE
    
    rng = np.random.RandomState(random_state)
    X_list = []
    labels_list = []
    
    total_ratio = sum(size_ratios)
    
    for i, ratio in enumerate(size_ratios):
        size = int(n_samples * ratio / total_ratio)
        center = rng.randn(1, n_features) * 5 + i * 3
        
        X_cluster = rng.randn(size, n_features) * 0.8 + center
        X_list.append(X_cluster)
        labels_list.append(np.full(size, i))
    
    X = np.vstack(X_list)
    labels = np.hstack(labels_list)
    
    # Shuffle
    indices = np.arange(len(X))
    rng.shuffle(indices)
    X = X[indices]
    labels = labels[indices]
    
    return X, labels


def count_outliers_in_coreset(S, X_full, labels_full, threshold_percentile=95):
    """Count how many outliers are in the coreset."""
    # Compute distances from coreset points to full data
    try:
        from scipy.spatial.distance import cdist
        distances = cdist(S, X_full)
    except ImportError:
        # Fallback: manual computation
        distances = np.sqrt(((S[:, None, :] - X_full[None, :, :]) ** 2).sum(axis=2))
    
    # Find outlier points in full data (points far from main cluster)
    outlier_mask = (labels_full == 1)
    if not outlier_mask.any():
        return 0, 0
    
    # Count coreset points that are closest to outliers
    min_distances = distances.min(axis=1)
    outlier_distances = distances[:, outlier_mask]
    
    # Count coreset points closer to outliers than to main data
    outlier_count = 0
    for i in range(len(S)):
        min_outlier_dist = outlier_distances[i].min() if outlier_distances.shape[1] > 0 else np.inf
        min_main_dist = distances[i, ~outlier_mask].min() if (~outlier_mask).any() else np.inf
        
        if min_outlier_dist < min_main_dist:
            outlier_count += 1
    
    return outlier_count, len(S)


def run_experiment_6_1_outliers():
    """Experiment 6.1: Outlier preservation."""
    print("=" * 70)
    print("Experiment 6.1: Outlier Preservation")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = 20_000
    n_features = 10
    outlier_fraction = 0.05
    k_reps = 1000
    
    # Generate data with outliers
    X, labels = generate_with_outliers(n_samples, n_features, outlier_fraction=outlier_fraction,
                                      random_state=RANDOM_STATE)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit coresets
    S_random, w_random = fit_random_coreset(X_scaled, k_reps, random_state=RANDOM_STATE)
    S_ddc, w_ddc, _ = fit_ddc_coreset_optimized(X_scaled, k_reps, n0=None, 
                                                 random_state=RANDOM_STATE)
    
    # Compute metrics
    metrics_random = compute_all_metrics(X_scaled, S_random, w_random, 'Random')
    metrics_ddc = compute_all_metrics(X_scaled, S_ddc, w_ddc, 'DDC')
    
    # Count outliers in coresets
    random_outliers, _ = count_outliers_in_coreset(S_random, X_scaled, labels)
    ddc_outliers, _ = count_outliers_in_coreset(S_ddc, X_scaled, labels)
    
    # Compute tail metrics (quantiles)
    from experiments.ddc_advantage.complex_marginals import compute_tail_metrics
    tail_random = compute_tail_metrics(X_scaled, S_random, w_random)
    tail_ddc = compute_tail_metrics(X_scaled, S_ddc, w_ddc)
    
    # Save results
    exp_name = "use_case_outliers"
    save_results(metrics_random, metrics_ddc, exp_name, results_dir)
    
    # Visualizations
    if n_features > 2:
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=RANDOM_STATE)
            X_2d = reducer.fit_transform(X_scaled)
            S_random_2d = reducer.transform(S_random)
            S_ddc_2d = reducer.transform(S_ddc)
        except ImportError:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=RANDOM_STATE)
            X_2d = reducer.fit_transform(X_scaled)
            S_random_2d = reducer.transform(S_random)
            S_ddc_2d = reducer.transform(S_ddc)
    else:
        X_2d = X_scaled
        S_random_2d = S_random
        S_ddc_2d = S_ddc
    
    plot_spatial_coverage_2d(
        X_2d, S_random_2d, w_random, S_ddc_2d, w_ddc,
        labels_full=labels,
        title="Spatial Coverage: Outlier Preservation",
        output_path=output_dir / f"{exp_name}_spatial.png"
    )
    
    print(f"\nOutliers in coreset:")
    print(f"  Random: {random_outliers}/{k_reps} ({random_outliers/k_reps*100:.1f}%)")
    print(f"  DDC: {ddc_outliers}/{k_reps} ({ddc_outliers/k_reps*100:.1f}%)")
    print(f"\nTail Metrics (Q0.95 error):")
    print(f"  Random: {tail_random['quantile_0.95_error']:.4f}")
    print(f"  DDC: {tail_ddc['quantile_0.95_error']:.4f}")
    print(f"  Improvement: {(tail_random['quantile_0.95_error'] / tail_ddc['quantile_0.95_error'] - 1) * 100:+.1f}%")
    
    return metrics_random, metrics_ddc


def run_experiment_6_2_low_density():
    """Experiment 6.2: Low-density region coverage."""
    print("\n" + "=" * 70)
    print("Experiment 6.2: Low-Density Region Coverage")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    results_dir = Path(__file__).parent / "results"
    
    n_samples = 20_000
    n_features = 10
    n_clusters = 3
    size_ratios = [1, 1, 10]  # Third cluster is 10x smaller
    k_reps = 1000
    
    # Generate data
    X, labels = generate_low_density_clusters(n_samples, n_features, n_clusters, 
                                              size_ratios=size_ratios, random_state=RANDOM_STATE)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit coresets
    S_random, w_random = fit_random_coreset(X_scaled, k_reps, random_state=RANDOM_STATE)
    S_ddc, w_ddc, _ = fit_ddc_coreset_optimized(X_scaled, k_reps, n0=None, 
                                                 random_state=RANDOM_STATE)
    
    # Compute metrics
    metrics_random = compute_all_metrics(X_scaled, S_random, w_random, 'Random')
    metrics_ddc = compute_all_metrics(X_scaled, S_ddc, w_ddc, 'DDC')
    
    # Check coverage per cluster
    coverage_random = compute_spatial_coverage(X_scaled, S_random, labels_full=labels)
    coverage_ddc = compute_spatial_coverage(X_scaled, S_ddc, labels_full=labels)
    
    # Save results
    exp_name = "use_case_low_density"
    save_results(metrics_random, metrics_ddc, exp_name, results_dir)
    
    # Visualizations
    if n_features > 2:
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=RANDOM_STATE)
            X_2d = reducer.fit_transform(X_scaled)
            S_random_2d = reducer.transform(S_random)
            S_ddc_2d = reducer.transform(S_ddc)
        except ImportError:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=RANDOM_STATE)
            X_2d = reducer.fit_transform(X_scaled)
            S_random_2d = reducer.transform(S_random)
            S_ddc_2d = reducer.transform(S_ddc)
    else:
        X_2d = X_scaled
        S_random_2d = S_random
        S_ddc_2d = S_ddc
    
    plot_spatial_coverage_2d(
        X_2d, S_random_2d, w_random, S_ddc_2d, w_ddc,
        labels_full=labels,
        title="Spatial Coverage: Low-Density Regions",
        output_path=output_dir / f"{exp_name}_spatial.png"
    )
    
    print(f"\nCoverage per cluster (Random): {coverage_random['coverage_per_cluster']}")
    print(f"Coverage per cluster (DDC): {coverage_ddc['coverage_per_cluster']}")
    
    # Check if small cluster is covered
    small_cluster_idx = np.argmin(size_ratios)
    random_small_coverage = coverage_random['coverage_per_cluster'].get(small_cluster_idx, 0)
    ddc_small_coverage = coverage_ddc['coverage_per_cluster'].get(small_cluster_idx, 0)
    
    print(f"\nSmall cluster (cluster {small_cluster_idx}) coverage:")
    print(f"  Random: {random_small_coverage:.2%}")
    print(f"  DDC: {ddc_small_coverage:.2%}")
    print(f"  DDC W1 Mean: {metrics_ddc['W1_mean']:.4f} "
          f"({(metrics_random['W1_mean'] / metrics_ddc['W1_mean'] - 1) * 100:+.1f}% vs Random)")
    
    return metrics_random, metrics_ddc


def main():
    """Run all specific use case experiments."""
    print("Running all Specific Use Case experiments...")
    print("=" * 70)
    
    results_6_1 = run_experiment_6_1_outliers()
    results_6_2 = run_experiment_6_2_low_density()
    
    print("\n" + "=" * 70)
    print("All Specific Use Case experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

