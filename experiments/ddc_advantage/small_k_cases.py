"""
Categoria 4: Casos com k Pequeno

Experimentos demonstrando vantagem do DDC quando k é pequeno:
- k muito pequeno (50, 100, 200)
- k proporcional ao número de clusters
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from experiments.ddc_advantage.utils import (
    compute_all_metrics, fit_random_coreset, fit_ddc_coreset_optimized,
    plot_spatial_coverage_2d, compute_spatial_coverage,
    save_results, RANDOM_STATE
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_experiment_4_1_small_k():
    """Experiment 4.1: Very small k values."""
    print("=" * 70)
    print("Experiment 4.1: Small k Values (50, 100, 200)")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = 20_000
    n_features = 10
    n_clusters = 4
    
    all_results = []
    
    for k_reps in [50, 100, 200]:
        print(f"\nTesting with k={k_reps}...")
        
        # Generate Gaussian Mixture
        X, labels = make_blobs(
            n_samples=n_samples, n_features=n_features, centers=n_clusters,
            cluster_std=0.8, random_state=RANDOM_STATE
        )
        
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
        
        # Count empty clusters
        random_empty = sum(1 for v in coverage_random['coverage_per_cluster'].values() if v == 0)
        ddc_empty = sum(1 for v in coverage_ddc['coverage_per_cluster'].values() if v == 0)
        
        # Save results
        exp_name = f"small_k_{k_reps}"
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
            title=f"Spatial Coverage: k={k_reps}",
            output_path=output_dir / f"{exp_name}_spatial.png"
        )
        
        result = {
            'k': k_reps,
            'random_empty_clusters': random_empty,
            'ddc_empty_clusters': ddc_empty,
            'random_cov_err': metrics_random['cov_err_fro'],
            'ddc_cov_err': metrics_ddc['cov_err_fro'],
            'ddc_improvement_cov': (metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100,
        }
        all_results.append(result)
        
        print(f"  Empty clusters - Random: {random_empty}, DDC: {ddc_empty}")
        print(f"  DDC Cov Error: {metrics_ddc['cov_err_fro']:.4f} "
              f"({result['ddc_improvement_cov']:+.1f}% vs Random)")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary: Small k Values")
    print("=" * 70)
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(results_dir / "small_k_summary.csv", index=False)
    
    return all_results


def run_experiment_4_2_proportional_k():
    """Experiment 4.2: k proportional to number of clusters."""
    print("\n" + "=" * 70)
    print("Experiment 4.2: k Proportional to Number of Clusters")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    results_dir = Path(__file__).parent / "results"
    
    n_samples = 20_000
    n_features = 10
    n_clusters = 8
    
    all_results = []
    
    for multiplier in [2, 3, 4]:
        k_reps = n_clusters * multiplier
        print(f"\nTesting with k={k_reps} ({multiplier}x {n_clusters} clusters)...")
        
        # Generate Gaussian Mixture
        X, labels = make_blobs(
            n_samples=n_samples, n_features=n_features, centers=n_clusters,
            cluster_std=0.8, random_state=RANDOM_STATE
        )
        
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
        
        # Count clusters with at least 1 point
        random_covered = sum(1 for v in coverage_random['coverage_per_cluster'].values() if v > 0)
        ddc_covered = sum(1 for v in coverage_ddc['coverage_per_cluster'].values() if v > 0)
        
        # Save results
        exp_name = f"proportional_k_{multiplier}x"
        save_results(metrics_random, metrics_ddc, exp_name, results_dir)
        
        result = {
            'k': k_reps,
            'multiplier': multiplier,
            'random_clusters_covered': random_covered,
            'ddc_clusters_covered': ddc_covered,
            'random_cov_err': metrics_random['cov_err_fro'],
            'ddc_cov_err': metrics_ddc['cov_err_fro'],
            'ddc_improvement_cov': (metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100,
        }
        all_results.append(result)
        
        print(f"  Clusters covered - Random: {random_covered}/{n_clusters}, DDC: {ddc_covered}/{n_clusters}")
        print(f"  DDC Cov Error: {metrics_ddc['cov_err_fro']:.4f} "
              f"({result['ddc_improvement_cov']:+.1f}% vs Random)")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary: Proportional k")
    print("=" * 70)
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(results_dir / "proportional_k_summary.csv", index=False)
    
    return all_results


def run_experiment_4_3_two_moons_small_k():
    """Experiment 4.3: Two Moons with small k."""
    print("\n" + "=" * 70)
    print("Experiment 4.3: Two Moons with Small k")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    results_dir = Path(__file__).parent / "results"
    
    n_samples = 5_000
    
    all_results = []
    
    for k_reps in [50, 100, 200]:
        print(f"\nTesting Two Moons with k={k_reps}...")
        
        # Generate Two Moons
        X, labels = make_moons(n_samples=n_samples, noise=0.1, random_state=RANDOM_STATE)
        
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
        exp_name = f"two_moons_k_{k_reps}"
        save_results(metrics_random, metrics_ddc, exp_name, results_dir)
        
        # Visualizations (already 2D)
        plot_spatial_coverage_2d(
            X_scaled, S_random, w_random, S_ddc, w_ddc,
            labels_full=labels,
            title=f"Two Moons: k={k_reps}",
            output_path=output_dir / f"{exp_name}_spatial.png"
        )
        
        result = {
            'k': k_reps,
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
    print("Summary: Two Moons Small k")
    print("=" * 70)
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(results_dir / "two_moons_small_k_summary.csv", index=False)
    
    return all_results


def main():
    """Run all small k experiments."""
    print("Running all Small k experiments...")
    print("=" * 70)
    
    results_4_1 = run_experiment_4_1_small_k()
    results_4_2 = run_experiment_4_2_proportional_k()
    results_4_3 = run_experiment_4_3_two_moons_small_k()
    
    print("\n" + "=" * 70)
    print("All Small k experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

