#!/usr/bin/env python3
"""
Extended investigation: Random vs Global DDC with all metrics.

Includes:
- Joint distribution metrics (mean, cov, corr errors, MMD)
- Marginal distribution metrics (Wasserstein-1, KS statistic)
- Per-feature analysis
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


def wasserstein_1d_approx(X_dim, S_dim, w, n_samples=5000, random_state=None):
    """Approximate Wasserstein-1 distance for 1D marginal distributions."""
    if random_state is not None:
        np.random.seed(random_state)
    
    # Sample from weighted coreset
    indices = np.random.choice(len(S_dim), size=n_samples, p=w, replace=True)
    S_sample = S_dim[indices]
    
    # Sample from full data
    X_sample = np.random.choice(X_dim, size=n_samples, replace=True)
    
    # Sort both samples
    X_sorted = np.sort(X_sample)
    S_sorted = np.sort(S_sample)
    
    # Wasserstein-1 is the mean absolute difference of sorted samples
    w1 = np.mean(np.abs(X_sorted - S_sorted))
    
    return w1


def ks_1d_approx(X_dim, S_dim, w, n_grid=512):
    """Approximate KS statistic for 1D marginal distributions."""
    # Create common grid
    x_min = min(X_dim.min(), S_dim.min())
    x_max = max(X_dim.max(), S_dim.max())
    grid = np.linspace(x_min, x_max, n_grid)
    
    # Compute empirical CDFs
    F_X = np.array([np.mean(X_dim <= x) for x in grid])
    
    # Weighted CDF for coreset
    F_S = np.array([np.sum(w[S_dim <= x]) for x in grid])
    
    # KS statistic is max absolute difference
    ks = float(np.max(np.abs(F_X - F_S)))
    
    return ks


def compute_mmd(X, Y, w_Y=None, kernel='rbf', gamma=None, n_samples=1000):
    """Compute Maximum Mean Discrepancy (MMD) between X and weighted Y."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    
    if w_Y is None:
        w_Y = np.ones(len(Y)) / len(Y)
    else:
        w_Y = np.asarray(w_Y, dtype=float)
        w_Y = w_Y / w_Y.sum()
    
    n_sample = min(n_samples, len(X))
    idx_x = np.random.choice(len(X), size=n_sample, replace=False)
    X_sample = X[idx_x]
    
    idx_y = np.random.choice(len(Y), size=n_sample, p=w_Y, replace=True)
    Y_sample = Y[idx_y]
    
    if gamma is None:
        all_data = np.vstack([X_sample, Y_sample])
        pairwise_dists = np.sqrt(((all_data[:, None, :] - all_data[None, :, :]) ** 2).sum(axis=2))
        gamma = 1.0 / np.median(pairwise_dists[pairwise_dists > 0])
    
    def rbf_kernel(X1, X2):
        dists_sq = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-gamma * dists_sq)
    
    K_XX = rbf_kernel(X_sample, X_sample)
    K_YY = rbf_kernel(Y_sample, Y_sample)
    K_XY = rbf_kernel(X_sample, Y_sample)
    
    mmd_sq = K_XX.mean() - 2 * K_XY.mean() + K_YY.mean()
    return np.sqrt(max(0, mmd_sq))


def compute_all_metrics(X_train, S, w, method_name):
    """Compute all metrics: joint and marginal."""
    # Joint distribution metrics
    mu_full = X_train.mean(axis=0)
    cov_full = np.cov(X_train, rowvar=False)
    corr_full = corr_from_cov(cov_full)
    
    mu_coreset = weighted_mean(S, w)
    cov_coreset = weighted_cov(S, w)
    corr_coreset = corr_from_cov(cov_coreset)
    
    mean_err = np.linalg.norm(mu_full - mu_coreset)
    cov_err = np.linalg.norm(cov_full - cov_coreset, ord='fro')
    corr_err = np.linalg.norm(corr_full - corr_coreset, ord='fro')
    mmd = compute_mmd(X_train, S, w_Y=w)
    
    # Marginal distribution metrics
    d = X_train.shape[1]
    W1_dims = []
    KS_dims = []
    
    for dim in range(d):
        seed = RANDOM_STATE + dim
        W1 = wasserstein_1d_approx(X_train[:, dim], S[:, dim], w, 
                                   n_samples=5000, random_state=seed)
        KS = ks_1d_approx(X_train[:, dim], S[:, dim], w, n_grid=512)
        W1_dims.append(W1)
        KS_dims.append(KS)
    
    return {
        'method': method_name,
        # Joint metrics
        'mean_err_l2': mean_err,
        'cov_err_fro': cov_err,
        'corr_err_fro': corr_err,
        'mmd': mmd,
        # Marginal metrics
        'W1_mean': np.mean(W1_dims),
        'W1_max': np.max(W1_dims),
        'KS_mean': np.mean(KS_dims),
        'KS_max': np.max(KS_dims),
        # Per-feature (for detailed analysis)
        'W1_dims': W1_dims,
        'KS_dims': KS_dims,
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
    X3_base = np.random.randn(n_samples, n_features)
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
    
    # 6. Large dataset
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
    metrics_random = compute_all_metrics(X_scaled, X_random, w_random, 'Random')
    
    # Global DDC optimized
    S_ddc_opt, w_ddc_opt, _ = fit_ddc_coreset(
        X_scaled, k=k_reps, n0=None,
        alpha=0.1, gamma=2.0, m_neighbors=16, refine_iters=2,
        reweight_full=True, random_state=RANDOM_STATE,
    )
    metrics_ddc_opt = compute_all_metrics(X_scaled, S_ddc_opt, w_ddc_opt, 'Global DDC (optimized)')
    
    # Print comparison
    print(f"\nJoint Distribution Metrics:")
    print(f"  Mean Error (L2):")
    print(f"    Random: {metrics_random['mean_err_l2']:.4f}")
    print(f"    DDC:    {metrics_ddc_opt['mean_err_l2']:.4f} "
          f"({(metrics_ddc_opt['mean_err_l2']/metrics_random['mean_err_l2'] - 1)*100:+.1f}%)")
    
    print(f"  Covariance Error (Frobenius):")
    print(f"    Random: {metrics_random['cov_err_fro']:.4f}")
    print(f"    DDC:    {metrics_ddc_opt['cov_err_fro']:.4f} "
          f"({(metrics_ddc_opt['cov_err_fro']/metrics_random['cov_err_fro'] - 1)*100:+.1f}%)")
    
    print(f"  Correlation Error (Frobenius):")
    print(f"    Random: {metrics_random['corr_err_fro']:.4f}")
    print(f"    DDC:    {metrics_ddc_opt['corr_err_fro']:.4f} "
          f"({(metrics_ddc_opt['corr_err_fro']/metrics_random['corr_err_fro'] - 1)*100:+.1f}%)")
    
    print(f"  MMD (RBF kernel):")
    print(f"    Random: {metrics_random['mmd']:.4f}")
    print(f"    DDC:    {metrics_ddc_opt['mmd']:.4f} "
          f"({(metrics_ddc_opt['mmd']/metrics_random['mmd'] - 1)*100:+.1f}%)")
    
    print(f"\nMarginal Distribution Metrics:")
    print(f"  Wasserstein-1 (mean):")
    print(f"    Random: {metrics_random['W1_mean']:.4f}")
    print(f"    DDC:    {metrics_ddc_opt['W1_mean']:.4f} "
          f"({(metrics_ddc_opt['W1_mean']/metrics_random['W1_mean'] - 1)*100:+.1f}%)")
    
    print(f"  Wasserstein-1 (max):")
    print(f"    Random: {metrics_random['W1_max']:.4f}")
    print(f"    DDC:    {metrics_ddc_opt['W1_max']:.4f} "
          f"({(metrics_ddc_opt['W1_max']/metrics_random['W1_max'] - 1)*100:+.1f}%)")
    
    print(f"  KS Statistic (mean):")
    print(f"    Random: {metrics_random['KS_mean']:.4f}")
    print(f"    DDC:    {metrics_ddc_opt['KS_mean']:.4f} "
          f"({(metrics_ddc_opt['KS_mean']/metrics_random['KS_mean'] - 1)*100:+.1f}%)")
    
    print(f"  KS Statistic (max):")
    print(f"    Random: {metrics_random['KS_max']:.4f}")
    print(f"    DDC:    {metrics_ddc_opt['KS_max']:.4f} "
          f"({(metrics_ddc_opt['KS_max']/metrics_random['KS_max'] - 1)*100:+.1f}%)")
    
    # Composite scores
    random_joint_score = metrics_random['cov_err_fro'] + 0.5 * metrics_random['corr_err_fro']
    ddc_joint_score = metrics_ddc_opt['cov_err_fro'] + 0.5 * metrics_ddc_opt['corr_err_fro']
    
    random_marginal_score = metrics_random['W1_mean'] + metrics_random['KS_mean']
    ddc_marginal_score = metrics_ddc_opt['W1_mean'] + metrics_ddc_opt['KS_mean']
    
    print(f"\nComposite Scores:")
    print(f"  Joint (cov + 0.5*corr):")
    print(f"    Random: {random_joint_score:.4f}")
    print(f"    DDC:    {ddc_joint_score:.4f}")
    
    print(f"  Marginal (W1_mean + KS_mean):")
    print(f"    Random: {random_marginal_score:.4f}")
    print(f"    DDC:    {ddc_marginal_score:.4f}")
    
    # Determine winner
    winner_joint = "Random" if random_joint_score < ddc_joint_score else "DDC"
    winner_marginal = "Random" if random_marginal_score < ddc_marginal_score else "DDC"
    
    print(f"\n  Winner (Joint): {winner_joint}")
    print(f"  Winner (Marginal): {winner_marginal}")
    
    return {
        'dataset': dataset_name,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'k_reps': k_reps,
        # Random metrics
        'random_mean_err': metrics_random['mean_err_l2'],
        'random_cov_err': metrics_random['cov_err_fro'],
        'random_corr_err': metrics_random['corr_err_fro'],
        'random_mmd': metrics_random['mmd'],
        'random_W1_mean': metrics_random['W1_mean'],
        'random_W1_max': metrics_random['W1_max'],
        'random_KS_mean': metrics_random['KS_mean'],
        'random_KS_max': metrics_random['KS_max'],
        # DDC metrics
        'ddc_mean_err': metrics_ddc_opt['mean_err_l2'],
        'ddc_cov_err': metrics_ddc_opt['cov_err_fro'],
        'ddc_corr_err': metrics_ddc_opt['corr_err_fro'],
        'ddc_mmd': metrics_ddc_opt['mmd'],
        'ddc_W1_mean': metrics_ddc_opt['W1_mean'],
        'ddc_W1_max': metrics_ddc_opt['W1_max'],
        'ddc_KS_mean': metrics_ddc_opt['KS_mean'],
        'ddc_KS_max': metrics_ddc_opt['KS_max'],
        # Scores
        'random_joint_score': random_joint_score,
        'ddc_joint_score': ddc_joint_score,
        'random_marginal_score': random_marginal_score,
        'ddc_marginal_score': ddc_marginal_score,
        'winner_joint': winner_joint,
        'winner_marginal': winner_marginal,
    }


def main():
    print("=" * 70)
    print("EXTENDED INVESTIGATION: Random vs Global DDC")
    print("All Metrics: Joint (mean, cov, corr, MMD) + Marginal (W1, KS)")
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
    print("SUMMARY: Complete Metrics Comparison")
    print("=" * 70)
    
    df_results = pd.DataFrame(results)
    
    # Save detailed results
    output_dir = Path(__file__).parent
    df_results.to_csv(output_dir / "random_vs_ddc_extended.csv", index=False)
    
    # Print summary
    print("\nJoint Distribution Metrics Summary:")
    print(f"{'Dataset':<30} {'Random Cov':<12} {'DDC Cov':<12} {'Random Corr':<12} {'DDC Corr':<12} {'Winner':<10}")
    print("-" * 100)
    for _, row in df_results.iterrows():
        print(f"{row['dataset']:<30} {row['random_cov_err']:>10.4f}  {row['ddc_cov_err']:>10.4f}  "
              f"{row['random_corr_err']:>10.4f}  {row['ddc_corr_err']:>10.4f}  {row['winner_joint']:<10}")
    
    print("\nMarginal Distribution Metrics Summary:")
    print(f"{'Dataset':<30} {'Random W1':<12} {'DDC W1':<12} {'Random KS':<12} {'DDC KS':<12} {'Winner':<10}")
    print("-" * 100)
    for _, row in df_results.iterrows():
        print(f"{row['dataset']:<30} {row['random_W1_mean']:>10.4f}  {row['ddc_W1_mean']:>10.4f}  "
              f"{row['random_KS_mean']:>10.4f}  {row['ddc_KS_mean']:>10.4f}  {row['winner_marginal']:<10}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    n_random_wins_joint = (df_results['winner_joint'] == 'Random').sum()
    n_ddc_wins_joint = (df_results['winner_joint'] == 'DDC').sum()
    
    n_random_wins_marginal = (df_results['winner_marginal'] == 'Random').sum()
    n_ddc_wins_marginal = (df_results['winner_marginal'] == 'DDC').sum()
    
    print(f"\nJoint Distribution:")
    print(f"  Random wins: {n_random_wins_joint}/{len(df_results)}")
    print(f"  DDC wins: {n_ddc_wins_joint}/{len(df_results)}")
    
    print(f"\nMarginal Distribution:")
    print(f"  Random wins: {n_random_wins_marginal}/{len(df_results)}")
    print(f"  DDC wins: {n_ddc_wins_marginal}/{len(df_results)}")
    
    print(f"\nResults saved to: {output_dir / 'random_vs_ddc_extended.csv'}")


if __name__ == "__main__":
    main()

