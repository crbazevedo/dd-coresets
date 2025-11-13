"""
Categoria 5: Datasets Reais com Estrutura Clara

Experimentos com datasets reais que têm estrutura de clusters bem definida:
- MNIST (digitos)
- Iris/Wine (UCI)
- Fashion-MNIST
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml, load_iris, load_wine
from experiments.ddc_advantage.utils import (
    compute_all_metrics, fit_random_coreset, fit_ddc_coreset_optimized,
    plot_spatial_coverage_2d, compute_spatial_coverage,
    save_results, RANDOM_STATE
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_mnist(n_samples=10000, n_components=50, random_state=None):
    """Load MNIST dataset and reduce dimensionality."""
    if random_state is None:
        random_state = RANDOM_STATE
    
    print("Loading MNIST...")
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X_full = mnist.data.astype(np.float32)
        y_full = mnist.target.astype(int)
        
        # Sample subset
        indices = np.random.RandomState(random_state).choice(len(X_full), size=n_samples, replace=False)
        X = X_full[indices]
        y = y_full[indices]
        
        # Reduce dimensionality with PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        X_reduced = pca.fit_transform(X)
        
        print(f"  Loaded {len(X)} samples, reduced to {n_components} dimensions")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        return X_reduced, y
    except Exception as e:
        print(f"  Error loading MNIST: {e}")
        return None, None


def load_fashion_mnist(n_samples=10000, n_components=50, random_state=None):
    """Load Fashion-MNIST dataset and reduce dimensionality."""
    if random_state is None:
        random_state = RANDOM_STATE
    
    print("Loading Fashion-MNIST...")
    try:
        fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
        X_full = fashion_mnist.data.astype(np.float32)
        y_full = fashion_mnist.target.astype(int)
        
        # Sample subset
        indices = np.random.RandomState(random_state).choice(len(X_full), size=n_samples, replace=False)
        X = X_full[indices]
        y = y_full[indices]
        
        # Reduce dimensionality with PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        X_reduced = pca.fit_transform(X)
        
        print(f"  Loaded {len(X)} samples, reduced to {n_components} dimensions")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        return X_reduced, y
    except Exception as e:
        print(f"  Error loading Fashion-MNIST: {e}")
        return None, None


def run_experiment_5_1_mnist():
    """Experiment 5.1: MNIST digits."""
    print("=" * 70)
    print("Experiment 5.1: MNIST Digits")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = 10_000
    n_components = 50
    k_reps = 1000
    
    # Load data
    X, y = load_mnist(n_samples=n_samples, n_components=n_components, random_state=RANDOM_STATE)
    
    if X is None:
        print("Skipping MNIST experiment (dataset not available)")
        return None
    
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
    
    # Check coverage per digit
    coverage_random = compute_spatial_coverage(X_scaled, S_random, labels_full=y)
    coverage_ddc = compute_spatial_coverage(X_scaled, S_ddc, labels_full=y)
    
    # Save results
    exp_name = "real_mnist"
    save_results(metrics_random, metrics_ddc, exp_name, results_dir)
    
    # Visualizations (2D projection)
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=RANDOM_STATE)
        X_2d = reducer.fit_transform(X_scaled)
        S_random_2d = reducer.transform(S_random)
        S_ddc_2d = reducer.transform(S_ddc)
    except ImportError:
        reducer = PCA(n_components=2, random_state=RANDOM_STATE)
        X_2d = reducer.fit_transform(X_scaled)
        S_random_2d = reducer.transform(S_random)
        S_ddc_2d = reducer.transform(S_ddc)
    
    plot_spatial_coverage_2d(
        X_2d, S_random_2d, w_random, S_ddc_2d, w_ddc,
        labels_full=y,
        title="Spatial Coverage: MNIST Digits",
        output_path=output_dir / f"{exp_name}_spatial.png"
    )
    
    print(f"\nCoverage per digit (Random): {len([v for v in coverage_random['coverage_per_cluster'].values() if v > 0])}/10 digits")
    print(f"Coverage per digit (DDC): {len([v for v in coverage_ddc['coverage_per_cluster'].values() if v > 0])}/10 digits")
    print(f"DDC Cov Error: {metrics_ddc['cov_err_fro']:.4f} "
          f"({(metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100:+.1f}% vs Random)")
    
    return metrics_random, metrics_ddc


def run_experiment_5_2_iris_wine():
    """Experiment 5.2: Iris and Wine datasets."""
    print("\n" + "=" * 70)
    print("Experiment 5.2: Iris and Wine (UCI)")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    results_dir = Path(__file__).parent / "results"
    
    all_results = []
    
    for dataset_name, loader in [('iris', load_iris), ('wine', load_wine)]:
        print(f"\nTesting {dataset_name}...")
        
        # Load data
        data = loader()
        X = data.data
        y = data.target
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use k proportional to dataset size
        k_reps = min(100, len(X) // 2)
        
        # Fit coresets
        S_random, w_random = fit_random_coreset(X_scaled, k_reps, random_state=RANDOM_STATE)
        S_ddc, w_ddc, _ = fit_ddc_coreset_optimized(X_scaled, k_reps, n0=None, 
                                                     random_state=RANDOM_STATE)
        
        # Compute metrics
        metrics_random = compute_all_metrics(X_scaled, S_random, w_random, 'Random')
        metrics_ddc = compute_all_metrics(X_scaled, S_ddc, w_ddc, 'DDC')
        
        # Check coverage per class
        coverage_random = compute_spatial_coverage(X_scaled, S_random, labels_full=y)
        coverage_ddc = compute_spatial_coverage(X_scaled, S_ddc, labels_full=y)
        
        # Save results
        exp_name = f"real_{dataset_name}"
        save_results(metrics_random, metrics_ddc, exp_name, results_dir)
        
        # Visualizations (2D if >2D)
        if X_scaled.shape[1] > 2:
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
            labels_full=y,
            title=f"Spatial Coverage: {dataset_name.capitalize()}",
            output_path=output_dir / f"{exp_name}_spatial.png"
        )
        
        result = {
            'dataset': dataset_name,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'k': k_reps,
            'random_cov_err': metrics_random['cov_err_fro'],
            'ddc_cov_err': metrics_ddc['cov_err_fro'],
            'ddc_improvement_cov': (metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100,
        }
        all_results.append(result)
        
        print(f"  Coverage per class - Random: {len([v for v in coverage_random['coverage_per_cluster'].values() if v > 0])}/{len(np.unique(y))}")
        print(f"  Coverage per class - DDC: {len([v for v in coverage_ddc['coverage_per_cluster'].values() if v > 0])}/{len(np.unique(y))}")
        print(f"  DDC Cov Error: {metrics_ddc['cov_err_fro']:.4f} "
              f"({result['ddc_improvement_cov']:+.1f}% vs Random)")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary: Iris and Wine")
    print("=" * 70)
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(results_dir / "iris_wine_summary.csv", index=False)
    
    return all_results


def run_experiment_5_3_fashion_mnist():
    """Experiment 5.3: Fashion-MNIST."""
    print("\n" + "=" * 70)
    print("Experiment 5.3: Fashion-MNIST")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    results_dir = Path(__file__).parent / "results"
    
    n_samples = 10_000
    n_components = 50
    k_reps = 1000
    
    # Load data
    X, y = load_fashion_mnist(n_samples=n_samples, n_components=n_components, random_state=RANDOM_STATE)
    
    if X is None:
        print("Skipping Fashion-MNIST experiment (dataset not available)")
        return None
    
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
    exp_name = "real_fashion_mnist"
    save_results(metrics_random, metrics_ddc, exp_name, results_dir)
    
    print(f"DDC Cov Error: {metrics_ddc['cov_err_fro']:.4f} "
          f"({(metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100:+.1f}% vs Random)")
    
    return metrics_random, metrics_ddc


def main():
    """Run all real dataset experiments."""
    print("Running all Real Dataset experiments...")
    print("=" * 70)
    
    results_5_1 = run_experiment_5_1_mnist()
    results_5_2 = run_experiment_5_2_iris_wine()
    results_5_3 = run_experiment_5_3_fashion_mnist()
    
    print("\n" + "=" * 70)
    print("All Real Dataset experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

