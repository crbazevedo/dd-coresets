#!/usr/bin/env python3
"""
Analyze Global DDC parameters and their effect on distribution metrics.

Computes joint distribution metrics (mean/cov/corr errors) for Global DDC
without running downstream tasks, and performs hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import sys
import os
import json
from pathlib import Path
from itertools import product
import time

# Add parent directory to path
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
    """
    Compute Maximum Mean Discrepancy (MMD) between X and weighted Y.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    
    if w_Y is None:
        w_Y = np.ones(len(Y)) / len(Y)
    else:
        w_Y = np.asarray(w_Y, dtype=float)
        w_Y = w_Y / w_Y.sum()
    
    # Sample from X (uniform) and Y (weighted)
    n_sample = min(n_samples, len(X))
    idx_x = np.random.choice(len(X), size=n_sample, replace=False)
    X_sample = X[idx_x]
    
    idx_y = np.random.choice(len(Y), size=n_sample, p=w_Y, replace=True)
    Y_sample = Y[idx_y]
    
    # Compute pairwise distances for gamma estimation
    if gamma is None:
        all_data = np.vstack([X_sample, Y_sample])
        pairwise_dists = np.sqrt(((all_data[:, None, :] - all_data[None, :, :]) ** 2).sum(axis=2))
        gamma = 1.0 / np.median(pairwise_dists[pairwise_dists > 0])
    
    # RBF kernel
    def rbf_kernel(X1, X2):
        dists_sq = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-gamma * dists_sq)
    
    # MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    K_XX = rbf_kernel(X_sample, X_sample)
    K_YY = rbf_kernel(Y_sample, Y_sample)
    K_XY = rbf_kernel(X_sample, Y_sample)
    
    mmd_sq = K_XX.mean() - 2 * K_XY.mean() + K_YY.mean()
    return np.sqrt(max(0, mmd_sq))


def evaluate_global_ddc_params(X_train, k_reps, alpha, gamma, m_neighbors, refine_iters, n0=None):
    """
    Evaluate Global DDC with given parameters.
    
    Returns dict with distribution metrics only (no downstream task).
    """
    try:
        if n0 is None:
            n0 = min(50_000, X_train.shape[0])
        
        S_global, w_global, info_global = fit_ddc_coreset(
            X_train,
            k=k_reps,
            n0=n0,
            alpha=alpha,
            m_neighbors=m_neighbors,
            gamma=gamma,
            refine_iters=refine_iters,
            reweight_full=True,
            random_state=RANDOM_STATE,
        )
        
        # Compute joint distribution metrics
        mu_full = X_train.mean(axis=0)
        cov_full = np.cov(X_train, rowvar=False)
        corr_full = corr_from_cov(cov_full)
        
        mu_coreset = weighted_mean(S_global, w_global)
        cov_coreset = weighted_cov(S_global, w_global)
        corr_coreset = corr_from_cov(cov_coreset)
        
        mean_err = np.linalg.norm(mu_full - mu_coreset)
        cov_err = np.linalg.norm(cov_full - cov_coreset, ord='fro')
        corr_err = np.linalg.norm(corr_full - corr_coreset, ord='fro')
        mmd = compute_mmd(X_train, S_global, w_Y=w_global)
        
        return {
            'mean_err_l2': mean_err,
            'cov_err_fro': cov_err,
            'corr_err_fro': corr_err,
            'mmd': mmd,
            'success': True,
        }
    except Exception as e:
        return {
            'mean_err_l2': np.inf,
            'cov_err_fro': np.inf,
            'corr_err_fro': np.inf,
            'mmd': np.inf,
            'success': False,
            'error': str(e),
        }


def main():
    print("=" * 70)
    print("GLOBAL DDC PARAMETER ANALYSIS")
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
    
    # Grid search parameters
    param_grid = {
        'alpha': [0.1, 0.2, 0.3, 0.4, 0.5],
        'gamma': [0.5, 1.0, 1.5, 2.0],
        'm_neighbors': [16, 32, 64],
        'refine_iters': [1, 2, 3],
    }
    
    print("Grid search parameters:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    print()
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Total combinations: {total_combinations}")
    print("Starting grid search...")
    print()
    
    results = []
    start_time = time.time()
    
    for i, (alpha, gamma, m_neighbors, refine_iters) in enumerate(
        product(param_grid['alpha'], param_grid['gamma'], 
                param_grid['m_neighbors'], param_grid['refine_iters'])
    ):
        print(f"[{i+1}/{total_combinations}] Testing: alpha={alpha:.1f}, gamma={gamma:.1f}, "
              f"m_neighbors={m_neighbors}, refine_iters={refine_iters}...", end=' ')
        
        metrics = evaluate_global_ddc_params(
            X_train, k_reps,
            alpha=alpha, gamma=gamma,
            m_neighbors=m_neighbors, refine_iters=refine_iters
        )
        
        if metrics['success']:
            composite_score = metrics['cov_err_fro'] + 0.5 * metrics['corr_err_fro']
            result = {
                'alpha': alpha,
                'gamma': gamma,
                'm_neighbors': m_neighbors,
                'refine_iters': refine_iters,
                'mean_err_l2': metrics['mean_err_l2'],
                'cov_err_fro': metrics['cov_err_fro'],
                'corr_err_fro': metrics['corr_err_fro'],
                'mmd': metrics['mmd'],
                'composite_score': composite_score,
            }
            results.append(result)
            print(f"cov_err={metrics['cov_err_fro']:.4f}, corr_err={metrics['corr_err_fro']:.4f}, "
                  f"mmd={metrics['mmd']:.4f}")
        else:
            print(f"ERROR: {metrics.get('error', 'Unknown')}")
            results.append({
                'alpha': alpha,
                'gamma': gamma,
                'm_neighbors': m_neighbors,
                'refine_iters': refine_iters,
                'mean_err_l2': np.inf,
                'cov_err_fro': np.inf,
                'corr_err_fro': np.inf,
                'mmd': np.inf,
                'composite_score': np.inf,
            })
    
    elapsed_time = time.time() - start_time
    print(f"\nGrid search completed in {elapsed_time:.1f} seconds")
    print()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('composite_score')
    
    # Display top 10
    print("=" * 70)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("=" * 70)
    print("\n" + results_df.head(10).to_string(index=False))
    
    # Best parameters
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    print(f"alpha: {best['alpha']:.1f}")
    print(f"gamma: {best['gamma']:.1f}")
    print(f"m_neighbors: {int(best['m_neighbors'])}")
    print(f"refine_iters: {int(best['refine_iters'])}")
    print(f"\nMetrics:")
    print(f"  Mean error (L2): {best['mean_err_l2']:.4f}")
    print(f"  Covariance error (Frobenius): {best['cov_err_fro']:.4f}")
    print(f"  Correlation error (Frobenius): {best['corr_err_fro']:.4f}")
    print(f"  MMD: {best['mmd']:.4f}")
    print(f"  Composite score: {best['composite_score']:.4f}")
    
    # Parameter effect analysis
    print("\n" + "=" * 70)
    print("PARAMETER EFFECT ANALYSIS")
    print("=" * 70)
    
    # Analyze effect of each parameter
    for param_name in ['alpha', 'gamma', 'm_neighbors', 'refine_iters']:
        print(f"\n{param_name.upper()} effect:")
        param_analysis = results_df.groupby(param_name).agg({
            'mean_err_l2': 'mean',
            'cov_err_fro': 'mean',
            'corr_err_fro': 'mean',
            'mmd': 'mean',
            'composite_score': 'mean',
        }).sort_values('composite_score')
        
        print(param_analysis.to_string())
        
        # Best value for this parameter
        best_param_value = param_analysis.index[0]
        print(f"  Best {param_name}: {best_param_value}")
    
    # Interaction analysis: alpha vs gamma
    print("\n" + "=" * 70)
    print("ALPHA vs GAMMA INTERACTION")
    print("=" * 70)
    interaction = results_df.groupby(['alpha', 'gamma']).agg({
        'cov_err_fro': 'mean',
        'corr_err_fro': 'mean',
        'composite_score': 'mean',
    }).sort_values('composite_score')
    print("\n" + interaction.head(10).to_string())
    
    # Save results
    output_dir = Path(__file__).parent
    results_df.to_csv(output_dir / "global_ddc_optimization_results.csv", index=False)
    
    best_params = {
        'alpha': float(best['alpha']),
        'gamma': float(best['gamma']),
        'm_neighbors': int(best['m_neighbors']),
        'refine_iters': int(best['refine_iters']),
        'metrics': {
            'mean_err_l2': float(best['mean_err_l2']),
            'cov_err_fro': float(best['cov_err_fro']),
            'corr_err_fro': float(best['corr_err_fro']),
            'mmd': float(best['mmd']),
        }
    }
    
    with open(output_dir / "global_ddc_best_parameters.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - {output_dir / 'global_ddc_optimization_results.csv'}")
    print(f"  - {output_dir / 'global_ddc_best_parameters.json'}")


if __name__ == "__main__":
    main()

