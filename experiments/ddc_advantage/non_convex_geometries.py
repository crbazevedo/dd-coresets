"""
Categoria 3: Geometrias Não-Convexas

Experimentos demonstrando vantagem do DDC em:
- Swiss Roll
- S-Curve
- Concentric Rings
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_swiss_roll, make_s_curve
from experiments.ddc_advantage.utils import (
    compute_all_metrics, fit_random_coreset, fit_ddc_coreset_optimized,
    plot_spatial_coverage_2d, plot_marginal_distributions, plot_metrics_comparison,
    save_results, RANDOM_STATE
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_swiss_roll_data(n_samples=10000, noise=0.1, random_state=None):
    """Generate Swiss Roll dataset."""
    if random_state is None:
        random_state = RANDOM_STATE
    
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, t


def generate_s_curve_data(n_samples=10000, noise=0.1, random_state=None):
    """Generate S-Curve dataset."""
    if random_state is None:
        random_state = RANDOM_STATE
    
    X, t = make_s_curve(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, t


def generate_concentric_rings(n_samples=10000, n_rings=3, random_state=None):
    """Generate concentric rings dataset."""
    if random_state is None:
        random_state = RANDOM_STATE
    
    rng = np.random.RandomState(random_state)
    X_list = []
    labels_list = []
    
    samples_per_ring = n_samples // n_rings
    
    for ring_idx in range(n_rings):
        radius = 1.0 + ring_idx * 2.0
        angles = rng.uniform(0, 2 * np.pi, samples_per_ring)
        
        x = radius * np.cos(angles) + rng.normal(0, 0.1, samples_per_ring)
        y = radius * np.sin(angles) + rng.normal(0, 0.1, samples_per_ring)
        
        X_ring = np.column_stack([x, y])
        X_list.append(X_ring)
        labels_list.append(np.full(samples_per_ring, ring_idx))
    
    X = np.vstack(X_list)
    labels = np.hstack(labels_list)
    
    # Shuffle
    indices = np.arange(len(X))
    rng.shuffle(indices)
    X = X[indices]
    labels = labels[indices]
    
    return X, labels


def run_experiment_3_1_swiss_roll():
    """Experiment 3.1: Swiss Roll manifold."""
    print("=" * 70)
    print("Experiment 3.1: Swiss Roll Manifold")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = 10_000
    k_reps = 1000
    
    # Generate data
    X, t = generate_swiss_roll_data(n_samples=n_samples, random_state=RANDOM_STATE)
    
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
    exp_name = "geometry_swiss_roll"
    save_results(metrics_random, metrics_ddc, exp_name, results_dir)
    
    # Visualizations (3D -> 2D projection)
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
    
    plot_spatial_coverage_2d(
        X_2d, S_random_2d, w_random, S_ddc_2d, w_ddc,
        labels_full=t,
        title="Spatial Coverage: Swiss Roll",
        output_path=output_dir / f"{exp_name}_spatial.png"
    )
    
    plot_metrics_comparison(
        metrics_random, metrics_ddc,
        output_path=output_dir / f"{exp_name}_metrics.png"
    )
    
    print(f"DDC Cov Error: {metrics_ddc['cov_err_fro']:.4f} "
          f"({(metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100:+.1f}% vs Random)")
    print(f"DDC W1 Mean: {metrics_ddc['W1_mean']:.4f} "
          f"({(metrics_random['W1_mean'] / metrics_ddc['W1_mean'] - 1) * 100:+.1f}% vs Random)")
    
    return metrics_random, metrics_ddc


def run_experiment_3_2_s_curve():
    """Experiment 3.2: S-Curve manifold."""
    print("\n" + "=" * 70)
    print("Experiment 3.2: S-Curve Manifold")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    results_dir = Path(__file__).parent / "results"
    
    n_samples = 10_000
    k_reps = 1000
    
    # Generate data
    X, t = generate_s_curve_data(n_samples=n_samples, random_state=RANDOM_STATE)
    
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
    exp_name = "geometry_s_curve"
    save_results(metrics_random, metrics_ddc, exp_name, results_dir)
    
    # Visualizations
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
    
    plot_spatial_coverage_2d(
        X_2d, S_random_2d, w_random, S_ddc_2d, w_ddc,
        labels_full=t,
        title="Spatial Coverage: S-Curve",
        output_path=output_dir / f"{exp_name}_spatial.png"
    )
    
    print(f"DDC Cov Error: {metrics_ddc['cov_err_fro']:.4f} "
          f"({(metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100:+.1f}% vs Random)")
    
    return metrics_random, metrics_ddc


def run_experiment_3_3_concentric_rings():
    """Experiment 3.3: Concentric rings."""
    print("\n" + "=" * 70)
    print("Experiment 3.3: Concentric Rings")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    results_dir = Path(__file__).parent / "results"
    
    n_samples = 10_000
    n_rings = 3
    k_reps = 1000
    
    # Generate data
    X, labels = generate_concentric_rings(n_samples=n_samples, n_rings=n_rings, 
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
    
    # Check coverage per ring
    from experiments.ddc_advantage.utils import compute_spatial_coverage
    coverage_random = compute_spatial_coverage(X_scaled, S_random, labels_full=labels)
    coverage_ddc = compute_spatial_coverage(X_scaled, S_ddc, labels_full=labels)
    
    # Save results
    exp_name = "geometry_concentric_rings"
    save_results(metrics_random, metrics_ddc, exp_name, results_dir)
    
    # Visualizations (already 2D)
    plot_spatial_coverage_2d(
        X_scaled, S_random, w_random, S_ddc, w_ddc,
        labels_full=labels,
        title=f"Spatial Coverage: {n_rings} Concentric Rings",
        output_path=output_dir / f"{exp_name}_spatial.png"
    )
    
    print(f"\nCoverage per ring (Random): {coverage_random['coverage_per_cluster']}")
    print(f"Coverage per ring (DDC): {coverage_ddc['coverage_per_cluster']}")
    print(f"DDC Cov Error: {metrics_ddc['cov_err_fro']:.4f} "
          f"({(metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100:+.1f}% vs Random)")
    
    return metrics_random, metrics_ddc


def main():
    """Run all non-convex geometry experiments."""
    print("Running all Non-Convex Geometry experiments...")
    print("=" * 70)
    
    results_3_1 = run_experiment_3_1_swiss_roll()
    results_3_2 = run_experiment_3_2_s_curve()
    results_3_3 = run_experiment_3_3_concentric_rings()
    
    print("\n" + "=" * 70)
    print("All Non-Convex Geometry experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

