#!/usr/bin/env python3
"""
Compare optimized Global DDC with baselines (Random, Stratified).

Computes joint distribution metrics for all methods.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, fetch_openml
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


def compute_joint_metrics(X_train, S, w, method_name):
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
    mmd = compute_mmd(X_train, S, w_Y=w)
    
    return {
        'method': method_name,
        'mean_err_l2': mean_err,
        'cov_err_fro': cov_err,
        'corr_err_fro': corr_err,
        'mmd': mmd,
    }


def main():
    print("=" * 70)
    print("COMPARAÇÃO: GLOBAL DDC OTIMIZADO vs BASELINES")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading data...")
    try:
        adult = fetch_openml("adult", version=2, as_frame=True, parser="pandas")
        df = adult.frame.copy()
        if 'class' in df.columns:
            df['target'] = (df['class'] == '>50K').astype(int)
            df = df.drop(columns=['class'])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using synthetic fallback...")
        X, y = make_classification(
            n_samples=30_000, n_features=10, n_informative=5, n_redundant=2,
            n_clusters_per_class=2, weights=[0.75, 0.25], random_state=RANDOM_STATE, class_sep=0.8,
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y
    
    # Preprocess
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'target']
    X_raw = df[numeric_cols].values
    y_raw = df['target'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_raw, test_size=0.3, stratify=y_raw, random_state=RANDOM_STATE
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print()
    
    k_reps = 1000
    results = []
    
    # 1. Random baseline
    print("Computing Random baseline...")
    np.random.seed(RANDOM_STATE)
    random_indices = np.random.choice(len(X_train), size=k_reps, replace=False)
    X_random = X_train[random_indices]
    w_random = np.ones(k_reps) / k_reps
    metrics_random = compute_joint_metrics(X_train, X_random, w_random, 'Random')
    results.append(metrics_random)
    print(f"  Done: cov_err={metrics_random['cov_err_fro']:.4f}, corr_err={metrics_random['corr_err_fro']:.4f}")
    
    # 2. Stratified baseline
    print("Computing Stratified baseline...")
    strat_indices = []
    for class_label in np.unique(y_train):
        class_mask = (y_train == class_label)
        class_indices = np.where(class_mask)[0]
        n_class = int(k_reps * np.sum(class_mask) / len(y_train))
        selected = np.random.choice(class_indices, size=n_class, replace=False)
        strat_indices.extend(selected)
    if len(strat_indices) < k_reps:
        remaining = k_reps - len(strat_indices)
        remaining_indices = np.setdiff1d(np.arange(len(X_train)), strat_indices)
        strat_indices.extend(np.random.choice(remaining_indices, size=remaining, replace=False))
    elif len(strat_indices) > k_reps:
        strat_indices = np.random.choice(strat_indices, size=k_reps, replace=False)
    X_strat = X_train[strat_indices]
    w_strat = np.ones(len(X_strat)) / len(X_strat)
    metrics_strat = compute_joint_metrics(X_train, X_strat, w_strat, 'Stratified')
    results.append(metrics_strat)
    print(f"  Done: cov_err={metrics_strat['cov_err_fro']:.4f}, corr_err={metrics_strat['corr_err_fro']:.4f}")
    
    # 3. Global DDC with default parameters
    print("Computing Global DDC (default: alpha=0.3, gamma=1.0, m_neighbors=32, refine_iters=1, n0=None)...")
    S_global_default, w_global_default, _ = fit_ddc_coreset(
        X_train, k=k_reps, n0=None,
        alpha=0.3, gamma=1.0, m_neighbors=32, refine_iters=1,
        reweight_full=True, random_state=RANDOM_STATE,
    )
    metrics_global_default = compute_joint_metrics(X_train, S_global_default, w_global_default, 'Global DDC (default)')
    results.append(metrics_global_default)
    print(f"  Done: cov_err={metrics_global_default['cov_err_fro']:.4f}, corr_err={metrics_global_default['corr_err_fro']:.4f}")
    
    # 4. Global DDC with optimized parameters
    print("Computing Global DDC (optimized: alpha=0.1, gamma=2.0, m_neighbors=16, refine_iters=2, n0=None)...")
    S_global_opt, w_global_opt, _ = fit_ddc_coreset(
        X_train, k=k_reps, n0=None,
        alpha=0.1, gamma=2.0, m_neighbors=16, refine_iters=2,
        reweight_full=True, random_state=RANDOM_STATE,
    )
    metrics_global_opt = compute_joint_metrics(X_train, S_global_opt, w_global_opt, 'Global DDC (optimized)')
    results.append(metrics_global_opt)
    print(f"  Done: cov_err={metrics_global_opt['cov_err_fro']:.4f}, corr_err={metrics_global_opt['corr_err_fro']:.4f}")
    
    # Create comparison table
    comparison_df = pd.DataFrame(results)
    comparison_df['composite_score'] = comparison_df['cov_err_fro'] + 0.5 * comparison_df['corr_err_fro']
    
    print("\n" + "=" * 70)
    print("COMPARAÇÃO COMPLETA")
    print("=" * 70)
    print("\n" + comparison_df.to_string(index=False))
    
    # Relative comparison
    print("\n" + "=" * 70)
    print("COMPARAÇÃO RELATIVA (vs Random)")
    print("=" * 70)
    random_cov = metrics_random['cov_err_fro']
    random_corr = metrics_random['corr_err_fro']
    random_composite = metrics_random['cov_err_fro'] + 0.5 * metrics_random['corr_err_fro']
    
    print(f"\nBaseline (Random):")
    print(f"  Cov error: {random_cov:.4f}")
    print(f"  Corr error: {random_corr:.4f}")
    print(f"  Composite: {random_composite:.4f}")
    
    print(f"\nComparação:")
    for _, row in comparison_df.iterrows():
        if row['method'] != 'Random':
            cov_ratio = row['cov_err_fro'] / random_cov
            corr_ratio = row['corr_err_fro'] / random_corr
            comp_ratio = row['composite_score'] / random_composite
            
            if cov_ratio < 1.0:
                cov_str = f"{((1 - cov_ratio) * 100):.1f}% melhor"
            else:
                cov_str = f"{(cov_ratio - 1) * 100:.1f}% pior"
            
            if corr_ratio < 1.0:
                corr_str = f"{((1 - corr_ratio) * 100):.1f}% melhor"
            else:
                corr_str = f"{(corr_ratio - 1) * 100:.1f}% pior"
            
            if comp_ratio < 1.0:
                comp_str = f"{((1 - comp_ratio) * 100):.1f}% melhor"
            else:
                comp_str = f"{(comp_ratio - 1) * 100:.1f}% pior"
            
            print(f"\n{row['method']}:")
            print(f"  Cov error: {row['cov_err_fro']:.4f} ({cov_str})")
            print(f"  Corr error: {row['corr_err_fro']:.4f} ({corr_str})")
            print(f"  Composite: {row['composite_score']:.4f} ({comp_str})")
    
    # Ranking
    print("\n" + "=" * 70)
    print("RANKING POR COMPOSITE SCORE (menor é melhor)")
    print("=" * 70)
    ranking = comparison_df.sort_values('composite_score')[['method', 'cov_err_fro', 'corr_err_fro', 'mmd', 'composite_score']]
    for i, (_, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {row['method']:30s}: score={row['composite_score']:.4f}, "
              f"cov={row['cov_err_fro']:.4f}, corr={row['corr_err_fro']:.4f}, mmd={row['mmd']:.4f}")
    
    # Save results
    output_dir = Path(__file__).parent
    comparison_df.to_csv(output_dir / "global_ddc_vs_baselines_comparison.csv", index=False)
    print(f"\nResults saved to: {output_dir / 'global_ddc_vs_baselines_comparison.csv'}")


if __name__ == "__main__":
    main()

