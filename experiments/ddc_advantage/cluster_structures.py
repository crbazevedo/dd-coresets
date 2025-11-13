"""
Categoria 1: Estruturas de Clusters

Experimentos demonstrando vantagem do DDC em:
- Gaussian Mixtures variadas (2, 4, 8, 16 clusters)
- Clusters desbalanceados
- Clusters com formas diferentes
- Clusters com densidades diferentes
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
    plot_spatial_coverage_2d, plot_marginal_distributions, plot_metrics_comparison,
    save_results, RANDOM_STATE
)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def generate_gaussian_mixture(n_samples, n_features, n_clusters, cluster_std=0.8, 
                              random_state=None, centers=None):
    """Generate Gaussian mixture dataset."""
    if random_state is None:
        random_state = RANDOM_STATE
    
    X, labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters if centers is None else centers,
        cluster_std=cluster_std,
        random_state=random_state,
        return_centers=False
    )
    
    return X, labels


def generate_imbalanced_clusters(n_samples, n_features, n_clusters, imbalance_ratio=10.0,
                                 random_state=None):
    """Generate Gaussian mixture with imbalanced cluster sizes."""
    if random_state is None:
        random_state = RANDOM_STATE
    
    # Create cluster sizes with imbalance
    cluster_sizes = []
    total = 0
    for i in range(n_clusters):
        size = int(n_samples / (n_clusters + imbalance_ratio * (n_clusters - 1 - i)))
        cluster_sizes.append(size)
        total += size
    
    # Adjust to match n_samples exactly
    diff = n_samples - total
    cluster_sizes[0] += diff
    
    # Generate data
    X_list = []
    labels_list = []
    rng = np.random.RandomState(random_state)
    
    for i, size in enumerate(cluster_sizes):
        center = rng.randn(1, n_features) * 5 + i * 3  # Separate centers
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


def generate_different_shapes(n_samples, n_features, n_clusters, random_state=None):
    """Generate clusters with different shapes (spherical vs elliptical)."""
    if random_state is None:
        random_state = RANDOM_STATE
    
    rng = np.random.RandomState(random_state)
    X_list = []
    labels_list = []
    
    for i in range(n_clusters):
        size = n_samples // n_clusters
        center = rng.randn(1, n_features) * 5 + i * 3
        
        if i % 2 == 0:
            # Spherical cluster
            X_cluster = rng.randn(size, n_features) * 0.8 + center
        else:
            # Elliptical cluster (stretch first dimension)
            X_cluster = rng.randn(size, n_features)
            X_cluster[:, 0] *= 2.0  # Stretch
            X_cluster = X_cluster * 0.8 + center
        
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


def generate_different_densities(n_samples, n_features, n_clusters, density_ratios=[1, 5, 10],
                                 random_state=None):
    """Generate clusters with different densities."""
    if random_state is None:
        random_state = RANDOM_STATE
    
    rng = np.random.RandomState(random_state)
    X_list = []
    labels_list = []
    
    total_ratio = sum(density_ratios)
    
    for i, ratio in enumerate(density_ratios):
        size = int(n_samples * ratio / total_ratio)
        center = rng.randn(1, n_features) * 5 + i * 3
        
        # Different std based on density ratio (inverse relationship)
        std = 0.8 / np.sqrt(ratio)  # Lower density = higher spread
        
        X_cluster = rng.randn(size, n_features) * std + center
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


def run_experiment_1_1_varied_clusters():
    """Experiment 1.1: Gaussian Mixtures with varying number of clusters."""
    print("=" * 70)
    print("Experiment 1.1: Gaussian Mixtures Varied (2, 4, 8, 16 clusters)")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = 20_000
    n_features = 10
    k_reps = 1000
    
    all_results = []
    
    for n_clusters in [2, 4, 8, 16]:
        print(f"\nTesting with {n_clusters} clusters...")
        
        # Generate data
        X, labels = generate_gaussian_mixture(n_samples, n_features, n_clusters, 
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
        
        # Save results
        exp_name = f"cluster_varied_{n_clusters}clusters"
        save_results(metrics_random, metrics_ddc, exp_name, results_dir)
        
        # Visualizations (2D projection if needed)
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
            title=f"Spatial Coverage: {n_clusters} Clusters",
            output_path=output_dir / f"{exp_name}_spatial.png"
        )
        
        plot_marginal_distributions(
            X_scaled, S_random, w_random, S_ddc, w_ddc,
            n_features=4,
            title=f"Marginal Distributions: {n_clusters} Clusters",
            output_path=output_dir / f"{exp_name}_marginals.png"
        )
        
        plot_metrics_comparison(
            metrics_random, metrics_ddc,
            output_path=output_dir / f"{exp_name}_metrics.png"
        )
        
        # Store for summary
        result = {
            'n_clusters': n_clusters,
            'random_cov_err': metrics_random['cov_err_fro'],
            'ddc_cov_err': metrics_ddc['cov_err_fro'],
            'random_W1_mean': metrics_random['W1_mean'],
            'ddc_W1_mean': metrics_ddc['W1_mean'],
            'ddc_improvement_cov': (metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100,
            'ddc_improvement_W1': (metrics_random['W1_mean'] / metrics_ddc['W1_mean'] - 1) * 100,
        }
        all_results.append(result)
        
        print(f"  DDC Cov Error: {metrics_ddc['cov_err_fro']:.4f} "
              f"({result['ddc_improvement_cov']:+.1f}% vs Random)")
        print(f"  DDC W1 Mean: {metrics_ddc['W1_mean']:.4f} "
              f"({result['ddc_improvement_W1']:+.1f}% vs Random)")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary: Varied Number of Clusters")
    print("=" * 70)
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(results_dir / "cluster_varied_summary.csv", index=False)
    
    return all_results


def run_experiment_1_2_imbalanced():
    """Experiment 1.2: Imbalanced clusters."""
    print("\n" + "=" * 70)
    print("Experiment 1.2: Imbalanced Clusters (1:10 ratio)")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    results_dir = Path(__file__).parent / "results"
    
    n_samples = 20_000
    n_features = 10
    n_clusters = 4
    k_reps = 1000
    
    # Generate imbalanced data
    X, labels = generate_imbalanced_clusters(n_samples, n_features, n_clusters, 
                                             imbalance_ratio=10.0, random_state=RANDOM_STATE)
    
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
    from experiments.ddc_advantage.utils import compute_spatial_coverage
    coverage_random = compute_spatial_coverage(X_scaled, S_random, labels_full=labels)
    coverage_ddc = compute_spatial_coverage(X_scaled, S_ddc, labels_full=labels)
    
    # Save results
    exp_name = "cluster_imbalanced"
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
        title="Spatial Coverage: Imbalanced Clusters (1:10 ratio)",
        output_path=output_dir / f"{exp_name}_spatial.png"
    )
    
    print(f"\nCoverage per cluster (Random): {coverage_random['coverage_per_cluster']}")
    print(f"Coverage per cluster (DDC): {coverage_ddc['coverage_per_cluster']}")
    print(f"DDC Cov Error: {metrics_ddc['cov_err_fro']:.4f} "
          f"({(metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100:+.1f}% vs Random)")
    
    return metrics_random, metrics_ddc


def run_experiment_1_3_different_shapes():
    """Experiment 1.3: Clusters with different shapes."""
    print("\n" + "=" * 70)
    print("Experiment 1.3: Clusters with Different Shapes (Spherical vs Elliptical)")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    results_dir = Path(__file__).parent / "results"
    
    n_samples = 20_000
    n_features = 10
    n_clusters = 4
    k_reps = 1000
    
    # Generate data
    X, labels = generate_different_shapes(n_samples, n_features, n_clusters, 
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
    
    # Save results
    exp_name = "cluster_different_shapes"
    save_results(metrics_random, metrics_ddc, exp_name, results_dir)
    
    print(f"DDC Cov Error: {metrics_ddc['cov_err_fro']:.4f} "
          f"({(metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100:+.1f}% vs Random)")
    
    return metrics_random, metrics_ddc


def run_experiment_1_4_different_densities():
    """Experiment 1.4: Clusters with different densities."""
    print("\n" + "=" * 70)
    print("Experiment 1.4: Clusters with Different Densities (1:5:10 ratio)")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    results_dir = Path(__file__).parent / "results"
    
    n_samples = 20_000
    n_features = 10
    n_clusters = 3
    k_reps = 1000
    
    # Generate data
    X, labels = generate_different_densities(n_samples, n_features, n_clusters,
                                            density_ratios=[1, 5, 10], random_state=RANDOM_STATE)
    
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
    from experiments.ddc_advantage.utils import compute_spatial_coverage
    coverage_random = compute_spatial_coverage(X_scaled, S_random, labels_full=labels)
    coverage_ddc = compute_spatial_coverage(X_scaled, S_ddc, labels_full=labels)
    
    # Save results
    exp_name = "cluster_different_densities"
    save_results(metrics_random, metrics_ddc, exp_name, results_dir)
    
    print(f"\nCoverage per cluster (Random): {coverage_random['coverage_per_cluster']}")
    print(f"Coverage per cluster (DDC): {coverage_ddc['coverage_per_cluster']}")
    print(f"DDC W1 Mean: {metrics_ddc['W1_mean']:.4f} "
          f"({(metrics_random['W1_mean'] / metrics_ddc['W1_mean'] - 1) * 100:+.1f}% vs Random)")
    
    return metrics_random, metrics_ddc


def main():
    """Run all cluster structure experiments."""
    import pandas as pd
    
    print("Running all Cluster Structure experiments...")
    print("=" * 70)
    
    # Run all experiments
    results_1_1 = run_experiment_1_1_varied_clusters()
    results_1_2 = run_experiment_1_2_imbalanced()
    results_1_3 = run_experiment_1_3_different_shapes()
    results_1_4 = run_experiment_1_4_different_densities()
    
    print("\n" + "=" * 70)
    print("All Cluster Structure experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

