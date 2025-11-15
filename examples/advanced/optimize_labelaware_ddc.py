#!/usr/bin/env python3
"""
Grid search optimization for Label-Aware DDC parameters.

Optimizes parameters to minimize joint distribution errors (covariance, correlation).
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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


def evaluate_labelaware_ddc_params(
    X_train, y_train, k_reps,
    alpha, gamma, m_neighbors, refine_iters,
    compute_downstream=False, X_val=None, y_val=None
):
    """
    Evaluate label-aware DDC with given parameters.
    
    Returns dict with metrics.
    """
    # Fit label-aware DDC
    S_labelaware_list = []
    w_labelaware_list = []
    y_labelaware_list = []
    
    for class_label in np.unique(y_train):
        class_mask = (y_train == class_label)
        X_class = X_train[class_mask]
        p_class = np.sum(class_mask) / len(y_train)
        k_class = max(1, int(round(k_reps * p_class)))
        n0_class = min(20_000, len(X_class))
        
        # Adjust m_neighbors for small classes
        m_neighbors_class = min(m_neighbors, len(X_class) // 10)
        m_neighbors_class = max(5, m_neighbors_class)  # Minimum 5
        
        try:
            S_class, w_class, _ = fit_ddc_coreset(
                X_class, k=k_class, n0=n0_class,
                alpha=alpha,
                m_neighbors=m_neighbors_class,
                gamma=gamma,
                refine_iters=refine_iters,
                reweight_full=True,
                random_state=RANDOM_STATE + class_label,
            )
            
            w_class_scaled = w_class * p_class
            S_labelaware_list.append(S_class)
            w_labelaware_list.append(w_class_scaled)
            y_labelaware_list.append(np.full(len(S_class), class_label))
        except Exception as e:
            # If fitting fails, return high error
            return {
                'mean_err_l2': np.inf,
                'cov_err_fro': np.inf,
                'corr_err_fro': np.inf,
                'downstream_auc': 0.0,
                'error': str(e),
            }
    
    S_labelaware = np.vstack(S_labelaware_list)
    w_labelaware = np.concatenate(w_labelaware_list)
    w_labelaware = w_labelaware / w_labelaware.sum()
    y_labelaware = np.concatenate(y_labelaware_list)
    
    # Compute joint distribution metrics
    mu_full = X_train.mean(axis=0)
    cov_full = np.cov(X_train, rowvar=False)
    corr_full = corr_from_cov(cov_full)
    
    mu_coreset = weighted_mean(S_labelaware, w_labelaware)
    cov_coreset = weighted_cov(S_labelaware, w_labelaware)
    corr_coreset = corr_from_cov(cov_coreset)
    
    mean_err = np.linalg.norm(mu_full - mu_coreset)
    cov_err = np.linalg.norm(cov_full - cov_coreset, ord='fro')
    corr_err = np.linalg.norm(corr_full - corr_coreset, ord='fro')
    
    results = {
        'mean_err_l2': mean_err,
        'cov_err_fro': cov_err,
        'corr_err_fro': corr_err,
        'downstream_auc': 0.0,
    }
    
    # Compute downstream AUC if requested
    if compute_downstream and X_val is not None and y_val is not None:
        try:
            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight=None)
            sample_weights = w_labelaware * len(X_train)
            lr.fit(S_labelaware, y_labelaware, sample_weight=sample_weights)
            y_pred_proba = lr.predict_proba(X_val)[:, 1]
            results['downstream_auc'] = roc_auc_score(y_val, y_pred_proba)
        except Exception as e:
            results['downstream_auc'] = 0.0
            results['downstream_error'] = str(e)
    
    return results


def main():
    print("=" * 70)
    print("LABEL-AWARE DDC PARAMETER OPTIMIZATION")
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
    
    # Split: train for coreset, validation for downstream evaluation
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled, y_raw, test_size=0.3, stratify=y_raw, random_state=RANDOM_STATE
    )
    
    # Further split train into coreset training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RANDOM_STATE + 1
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Validation set: {X_val.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    print()
    
    k_reps = 1000
    
    # Grid search parameters (reduced for faster execution)
    param_grid = {
        'alpha': [0.1, 0.2, 0.3, 0.4, 0.5],
        'gamma': [0.5, 1.0, 1.5, 2.0],
        'm_neighbors': [16, 32, 64],
        'refine_iters': [1, 2],  # Reduced from [1,2,3] for speed
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
        
        try:
            metrics = evaluate_labelaware_ddc_params(
                X_train, y_train, k_reps,
                alpha=alpha, gamma=gamma,
                m_neighbors=m_neighbors, refine_iters=refine_iters,
                compute_downstream=True, X_val=X_val, y_val=y_val
            )
            
            # Composite score: prioritize cov_err, then corr_err
            # Lower is better, so we use negative for sorting
            composite_score = metrics['cov_err_fro'] + 0.5 * metrics['corr_err_fro']
            
            result = {
                'alpha': alpha,
                'gamma': gamma,
                'm_neighbors': m_neighbors,
                'refine_iters': refine_iters,
                'mean_err_l2': metrics['mean_err_l2'],
                'cov_err_fro': metrics['cov_err_fro'],
                'corr_err_fro': metrics['corr_err_fro'],
                'downstream_auc': metrics['downstream_auc'],
                'composite_score': composite_score,
            }
            
            results.append(result)
            print(f"cov_err={metrics['cov_err_fro']:.4f}, corr_err={metrics['corr_err_fro']:.4f}, "
                  f"AUC={metrics['downstream_auc']:.4f}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'alpha': alpha,
                'gamma': gamma,
                'm_neighbors': m_neighbors,
                'refine_iters': refine_iters,
                'mean_err_l2': np.inf,
                'cov_err_fro': np.inf,
                'corr_err_fro': np.inf,
                'downstream_auc': 0.0,
                'composite_score': np.inf,
                'error': str(e),
            })
    
    elapsed_time = time.time() - start_time
    print(f"\nGrid search completed in {elapsed_time:.1f} seconds")
    print()
    
    # Convert to DataFrame and sort
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
    print(f"  Downstream AUC: {best['downstream_auc']:.4f}")
    print(f"  Composite score: {best['composite_score']:.4f}")
    
    # Save results
    output_dir = Path(__file__).parent
    results_df.to_csv(output_dir / "optimization_results.csv", index=False)
    
    best_params = {
        'alpha': float(best['alpha']),
        'gamma': float(best['gamma']),
        'm_neighbors': int(best['m_neighbors']),
        'refine_iters': int(best['refine_iters']),
        'metrics': {
            'mean_err_l2': float(best['mean_err_l2']),
            'cov_err_fro': float(best['cov_err_fro']),
            'corr_err_fro': float(best['corr_err_fro']),
            'downstream_auc': float(best['downstream_auc']),
        }
    }
    
    with open(output_dir / "best_parameters.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - {output_dir / 'optimization_results.csv'}")
    print(f"  - {output_dir / 'best_parameters.json'}")


if __name__ == "__main__":
    main()

