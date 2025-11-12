#!/usr/bin/env python3
"""
Investigate why Random sampling outperforms Global DDC on some datasets.

Test multiple datasets to understand when Random is better and why.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_blobs, fetch_openml
from dd_coresets import fit_ddc_coreset

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def weighted_mean(S, w):
    """Compute weighted mean."""
    S = np.asarray(S, dtype=float)
    w = np.asarray(w, dtype=float)
    return (S * w[:, None]).sum(axis=0)


def weighted_cov(S, w):
    """Compute weighted covariance matrix."""
    S = np.asarray(S, dtype=float)
    w = np.asarray(w, dtype=float)
    mu = weighted_mean(S, w)
    Xc = S - mu
    cov = (Xc * w[:, None]).T @ Xc
    return cov


def corr_from_cov(cov):
    """Compute correlation matrix from covariance matrix."""
    cov = np.asarray(cov, dtype=float)
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    inv_std = 1.0 / std
    C = cov * inv_std[:, None] * inv_std[None, :]
    return C


def compute_metrics(X_train, S, w):
    """Compute joint distribution metrics."""
    mu_full = X_train.mean(axis=0)
    cov_full = np.cov(X_train, rowvar=False)
    corr_full = corr_from_cov(cov_full)
    
    mu_coreset = weighted_mean(S, w)
    cov_coreset = weighted_cov(S, w)
    corr_coreset = corr_from_cov(cov_coreset)
    
    mean_err = np.linalg.norm(mu_full - mu_coreset)
    cov_err = np.linalg.norm(cov_full - cov_coreset, ord='fro')
    corr_err = np.linalg.norm(corr_full - corr_coreset, ord='fro')
    
    return {
        'mean_err': mean_err,
        'cov_err': cov_err,
        'corr_err': corr_err,
    }


def generate_datasets():
    """Generate multiple test datasets."""
    datasets = {}
    
    # 1. Simple Gaussian (isotropic)
    print("Generating dataset 1: Simple Gaussian (isotropic)...")
    X1, _ = make_blobs(n_samples=20_000, n_features=10, centers=1, 
                       cluster_std=1.0, random_state=RANDOM_STATE)
    datasets['Gaussian (isotropic)'] = X1
    
    # 2. Gaussian Mixture (4 components)
    print("Generating dataset 2: Gaussian Mixture (4 components)...")
    X2, _ = make_blobs(n_samples=20_000, n_features=10, centers=4,
                       cluster_std=0.8, random_state=RANDOM_STATE)
    datasets['Gaussian Mixture (4)'] = X2
    
    # 3. Anisotropic Gaussian (strong correlation)
    print("Generating dataset 3: Anisotropic Gaussian...")
    n_samples = 20_000
    n_features = 10
    # Create correlated features
    X3_base = np.random.randn(n_samples, n_features)
    # Add strong correlation between first 5 features
    X3_corr = X3_base.copy()
    X3_corr[:, 1:5] = 0.7 * X3_base[:, 0:1] + 0.3 * X3_base[:, 1:5]
    datasets['Anisotropic Gaussian'] = X3_corr
    
    # 4. High-dimensional sparse
    print("Generating dataset 4: High-dimensional sparse...")
    X4, _ = make_classification(
        n_samples=20_000, n_features=20, n_informative=5, n_redundant=2,
        n_clusters_per_class=2, random_state=RANDOM_STATE, class_sep=1.0
    )
    datasets['High-dim Sparse'] = X4
    
    # 5. Adult Census Income (real dataset)
    print("Generating dataset 5: Adult Census Income...")
    try:
        adult = fetch_openml("adult", version=2, as_frame=True, parser="pandas")
        df = adult.frame.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'class' in df.columns:
            numeric_cols = [c for c in numeric_cols if c != 'class']
        X5 = df[numeric_cols].values
        scaler = StandardScaler()
        X5 = scaler.fit_transform(X5)
        datasets['Adult Census'] = X5
    except Exception as e:
        print(f"  Error loading Adult Census: {e}")
    
    # 6. Large dataset (more samples)
    print("Generating dataset 6: Large Gaussian Mixture...")
    X6, _ = make_blobs(n_samples=100_000, n_features=10, centers=4,
                       cluster_std=0.8, random_state=RANDOM_STATE)
    datasets['Large Gaussian Mixture'] = X6
    
    return datasets


def test_dataset(X, dataset_name, k_reps=1000):
    """Test Random vs Global DDC on a single dataset."""
    print(f"\n{'='*70}")
    print(f"Testing: {dataset_name}")
    print(f"  Shape: {X.shape}")
    print(f"  k_reps: {k_reps}")
    print(f"{'='*70}")
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Random baseline
    np.random.seed(RANDOM_STATE)
    random_indices = np.random.choice(len(X_scaled), size=k_reps, replace=False)
    X_random = X_scaled[random_indices]
    w_random = np.ones(k_reps) / k_reps
    metrics_random = compute_metrics(X_scaled, X_random, w_random)
    
    # Global DDC default
    S_ddc_default, w_ddc_default, _ = fit_ddc_coreset(
        X_scaled, k=k_reps, n0=None,
        alpha=0.3, gamma=1.0, m_neighbors=32, refine_iters=1,
        reweight_full=True, random_state=RANDOM_STATE,
    )
    metrics_ddc_default = compute_metrics(X_scaled, S_ddc_default, w_ddc_default)
    
    # Global DDC optimized
    S_ddc_opt, w_ddc_opt, _ = fit_ddc_coreset(
        X_scaled, k=k_reps, n0=None,
        alpha=0.1, gamma=2.0, m_neighbors=16, refine_iters=2,
        reweight_full=True, random_state=RANDOM_STATE,
    )
    metrics_ddc_opt = compute_metrics(X_scaled, S_ddc_opt, w_ddc_opt)
    
    # Compare
    print(f"\nResults:")
    print(f"  Random:")
    print(f"    Cov error: {metrics_random['cov_err']:.4f}")
    print(f"    Corr error: {metrics_random['corr_err']:.4f}")
    
    print(f"  Global DDC (default):")
    print(f"    Cov error: {metrics_ddc_default['cov_err']:.4f} "
          f"({(metrics_ddc_default['cov_err']/metrics_random['cov_err'] - 1)*100:+.1f}% vs Random)")
    print(f"    Corr error: {metrics_ddc_default['corr_err']:.4f} "
          f"({(metrics_ddc_default['corr_err']/metrics_random['corr_err'] - 1)*100:+.1f}% vs Random)")
    
    print(f"  Global DDC (optimized):")
    print(f"    Cov error: {metrics_ddc_opt['cov_err']:.4f} "
          f"({(metrics_ddc_opt['cov_err']/metrics_random['cov_err'] - 1)*100:+.1f}% vs Random)")
    print(f"    Corr error: {metrics_ddc_opt['corr_err']:.4f} "
          f"({(metrics_ddc_opt['corr_err']/metrics_random['corr_err'] - 1)*100:+.1f}% vs Random)")
    
    # Determine winner
    random_score = metrics_random['cov_err'] + 0.5 * metrics_random['corr_err']
    ddc_default_score = metrics_ddc_default['cov_err'] + 0.5 * metrics_ddc_default['corr_err']
    ddc_opt_score = metrics_ddc_opt['cov_err'] + 0.5 * metrics_ddc_opt['corr_err']
    
    winner = "Random" if random_score < min(ddc_default_score, ddc_opt_score) else "DDC"
    print(f"\n  Winner: {winner} (composite score)")
    
    return {
        'dataset': dataset_name,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'k_reps': k_reps,
        'random_cov_err': metrics_random['cov_err'],
        'random_corr_err': metrics_random['corr_err'],
        'ddc_default_cov_err': metrics_ddc_default['cov_err'],
        'ddc_default_corr_err': metrics_ddc_default['corr_err'],
        'ddc_opt_cov_err': metrics_ddc_opt['cov_err'],
        'ddc_opt_corr_err': metrics_ddc_opt['corr_err'],
        'random_score': random_score,
        'ddc_default_score': ddc_default_score,
        'ddc_opt_score': ddc_opt_score,
        'winner': winner,
    }


def main():
    print("=" * 70)
    print("INVESTIGATION: Why Random outperforms Global DDC?")
    print("=" * 70)
    print()
    
    # Generate datasets
    datasets = generate_datasets()
    
    # Test each dataset
    results = []
    k_reps = 1000
    
    for name, X in datasets.items():
        try:
            result = test_dataset(X, name, k_reps=k_reps)
            results.append(result)
        except Exception as e:
            print(f"\nERROR testing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Random vs Global DDC")
    print("=" * 70)
    
    df_results = pd.DataFrame(results)
    
    print("\nFull Results:")
    print(df_results.to_string(index=False))
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    n_random_wins = (df_results['winner'] == 'Random').sum()
    n_ddc_wins = (df_results['winner'] == 'DDC').sum()
    
    print(f"\nRandom wins: {n_random_wins}/{len(df_results)}")
    print(f"DDC wins: {n_ddc_wins}/{len(df_results)}")
    
    print("\nDatasets where Random is better:")
    random_wins = df_results[df_results['winner'] == 'Random']
    if len(random_wins) > 0:
        for _, row in random_wins.iterrows():
            print(f"  - {row['dataset']}: "
                  f"Random score={row['random_score']:.4f}, "
                  f"DDC opt score={row['ddc_opt_score']:.4f}")
    
    print("\nDatasets where DDC is better:")
    ddc_wins = df_results[df_results['winner'] == 'DDC']
    if len(ddc_wins) > 0:
        for _, row in ddc_wins.iterrows():
            print(f"  - {row['dataset']}: "
                  f"Random score={row['random_score']:.4f}, "
                  f"DDC opt score={row['ddc_opt_score']:.4f}")
    
    # Save results
    output_dir = Path(__file__).parent
    df_results.to_csv(output_dir / "random_vs_ddc_investigation.csv", index=False)
    print(f"\nResults saved to: {output_dir / 'random_vs_ddc_investigation.csv'}")


if __name__ == "__main__":
    main()

