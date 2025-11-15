#!/usr/bin/env python3
"""
Diagnostic script for Label-Aware DDC coreset.

Analyzes:
1. Weight distribution and entropy
2. Spatial coverage
3. Parameter adequacy
4. Comparison with baselines
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, fetch_openml
from sklearn.decomposition import PCA
from dd_coresets import fit_ddc_coreset

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

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


def compute_entropy(w):
    """Compute entropy of weights (diversity measure)."""
    w = np.asarray(w, dtype=float)
    w = w[w > 0]  # Remove zeros
    return -np.sum(w * np.log(w))


def analyze_weights(S, w, y, class_labels):
    """Analyze weight distribution per class."""
    results = {}
    
    for class_label in class_labels:
        mask = (y == class_label)
        w_class = w[mask]
        S_class = S[mask]
        
        results[class_label] = {
            'n_reps': len(w_class),
            'weight_sum': w_class.sum(),
            'weight_mean': w_class.mean(),
            'weight_std': w_class.std(),
            'weight_min': w_class.min(),
            'weight_max': w_class.max(),
            'entropy': compute_entropy(w_class),
            'max_weight_ratio': w_class.max() / w_class.mean() if w_class.mean() > 0 else np.inf,
            'concentration': np.sum(w_class ** 2),  # Gini coefficient proxy
        }
    
    return results


def analyze_spatial_coverage(X_train, S, w, y_train, y_coreset, n_samples=1000):
    """Analyze spatial coverage of coreset."""
    # Sample from training data
    n_sample = min(n_samples, len(X_train))
    idx_sample = np.random.choice(len(X_train), size=n_sample, replace=False)
    X_sample = X_train[idx_sample]
    
    # Compute distances from sampled points to coreset
    distances = []
    for x in X_sample:
        dists_to_coreset = np.sqrt(((S - x) ** 2).sum(axis=1))
        min_dist = dists_to_coreset.min()
        # Weighted distance (closer points with higher weights)
        weighted_dist = np.sum(dists_to_coreset * w)
        distances.append({
            'min_distance': min_dist,
            'weighted_distance': weighted_dist,
        })
    
    distances_df = pd.DataFrame(distances)
    
    return {
        'mean_min_distance': distances_df['min_distance'].mean(),
        'median_min_distance': distances_df['min_distance'].median(),
        'max_min_distance': distances_df['min_distance'].max(),
        'mean_weighted_distance': distances_df['weighted_distance'].mean(),
        'coverage_radius_95': np.percentile(distances_df['min_distance'], 95),
    }


def analyze_parameters(X_train, y_train, k_reps, current_params):
    """Analyze if current parameters are adequate."""
    results = {}
    
    for class_label in np.unique(y_train):
        class_mask = (y_train == class_label)
        X_class = X_train[class_mask]
        n_class = len(X_class)
        p_class = n_class / len(y_train)
        k_class = max(1, int(round(k_reps * p_class)))
        
        # Current parameters
        n0_class = min(20_000, n_class)
        alpha_class = current_params.get('alpha', 0.3)
        m_neighbors_class = min(32, n_class // 10)
        
        results[class_label] = {
            'n_samples': n_class,
            'proportion': p_class,
            'k_allocated': k_class,
            'k_ratio': k_class / n_class,
            'n0': n0_class,
            'n0_ratio': n0_class / n_class,
            'alpha': alpha_class,
            'm_neighbors': m_neighbors_class,
            'm_neighbors_ratio': m_neighbors_class / n_class,
            'adequate_n0': n0_class >= min(1000, n_class * 0.5),
            'adequate_k': k_class >= max(10, int(n_class * 0.01)),
        }
    
    return results


def compute_joint_metrics(X_train, S, w, method_name):
    """Compute joint distribution metrics."""
    mu_full = X_train.mean(axis=0)
    cov_full = np.cov(X_train, rowvar=False)
    corr_full = corr_from_cov(cov_full)
    
    mu_coreset = weighted_mean(S, w)
    cov_coreset = weighted_cov(S, w)
    corr_coreset = corr_from_cov(cov_coreset)
    
    return {
        'method': method_name,
        'mean_err_l2': np.linalg.norm(mu_full - mu_coreset),
        'cov_err_fro': np.linalg.norm(cov_full - cov_coreset, ord='fro'),
        'corr_err_fro': np.linalg.norm(corr_full - corr_coreset, ord='fro'),
    }


def main():
    print("=" * 70)
    print("LABEL-AWARE DDC DIAGNOSTIC ANALYSIS")
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
    print(f"Class distribution: {np.bincount(y_train) / len(y_train)}")
    print()
    
    # Current label-aware DDC parameters
    k_reps = 1000
    current_params = {
        'alpha': 0.3,
        'gamma': 1.0,
        'm_neighbors': 32,
        'refine_iters': 2,
    }
    
    print("=" * 70)
    print("1. WEIGHT ANALYSIS")
    print("=" * 70)
    
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
        alpha_class = 0.25 if len(X_class) < 5000 else 0.3
        m_neighbors_class = min(32, len(X_class) // 10)
        
        S_class, w_class, _ = fit_ddc_coreset(
            X_class, k=k_class, n0=n0_class,
            alpha=alpha_class, m_neighbors=m_neighbors_class,
            gamma=current_params['gamma'],
            refine_iters=current_params['refine_iters'],
            reweight_full=True,
            random_state=RANDOM_STATE + class_label,
        )
        
        w_class_scaled = w_class * p_class
        S_labelaware_list.append(S_class)
        w_labelaware_list.append(w_class_scaled)
        y_labelaware_list.append(np.full(len(S_class), class_label))
    
    S_labelaware = np.vstack(S_labelaware_list)
    w_labelaware = np.concatenate(w_labelaware_list)
    w_labelaware = w_labelaware / w_labelaware.sum()
    y_labelaware = np.concatenate(y_labelaware_list)
    
    # Analyze weights
    weight_analysis = analyze_weights(S_labelaware, w_labelaware, y_labelaware, np.unique(y_train))
    
    for class_label, metrics in weight_analysis.items():
        print(f"\nClass {class_label}:")
        print(f"  Representatives: {metrics['n_reps']}")
        print(f"  Weight sum: {metrics['weight_sum']:.6f}")
        print(f"  Weight mean: {metrics['weight_mean']:.6f}")
        print(f"  Weight std: {metrics['weight_std']:.6f}")
        print(f"  Weight range: [{metrics['weight_min']:.6f}, {metrics['weight_max']:.6f}]")
        print(f"  Entropy: {metrics['entropy']:.4f} (higher = more diverse)")
        print(f"  Max/Mean ratio: {metrics['max_weight_ratio']:.2f} (lower = more uniform)")
        print(f"  Concentration: {metrics['concentration']:.4f} (lower = less concentrated)")
    
    print("\n" + "=" * 70)
    print("2. SPATIAL COVERAGE ANALYSIS")
    print("=" * 70)
    
    coverage = analyze_spatial_coverage(X_train, S_labelaware, w_labelaware, y_train, y_labelaware)
    
    print(f"\nMean minimum distance to coreset: {coverage['mean_min_distance']:.4f}")
    print(f"Median minimum distance: {coverage['median_min_distance']:.4f}")
    print(f"95th percentile distance: {coverage['coverage_radius_95']:.4f}")
    print(f"Mean weighted distance: {coverage['mean_weighted_distance']:.4f}")
    
    print("\n" + "=" * 70)
    print("3. PARAMETER ANALYSIS")
    print("=" * 70)
    
    param_analysis = analyze_parameters(X_train, y_train, k_reps, current_params)
    
    for class_label, params in param_analysis.items():
        print(f"\nClass {class_label}:")
        print(f"  Samples: {params['n_samples']:,} ({params['proportion']:.2%})")
        print(f"  k allocated: {params['k_allocated']} ({params['k_ratio']:.4%} of class)")
        print(f"  n0: {params['n0']:,} ({params['n0_ratio']:.2%} of class)")
        print(f"  alpha: {params['alpha']:.2f}")
        print(f"  m_neighbors: {params['m_neighbors']} ({params['m_neighbors_ratio']:.4%} of class)")
        print(f"  Adequate n0: {params['adequate_n0']}")
        print(f"  Adequate k: {params['adequate_k']}")
    
    print("\n" + "=" * 70)
    print("4. JOINT DISTRIBUTION METRICS COMPARISON")
    print("=" * 70)
    
    # Random baseline
    random_indices = np.random.choice(len(X_train), size=k_reps, replace=False)
    X_random = X_train[random_indices]
    w_random = np.ones(k_reps) / k_reps
    
    # Stratified baseline
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
    
    # Compute metrics
    metrics_random = compute_joint_metrics(X_train, X_random, w_random, 'Random')
    metrics_strat = compute_joint_metrics(X_train, X_strat, w_strat, 'Stratified')
    metrics_labelaware = compute_joint_metrics(X_train, S_labelaware, w_labelaware, 'Label-aware DDC')
    
    comparison_df = pd.DataFrame([metrics_random, metrics_strat, metrics_labelaware])
    print("\n" + comparison_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("5. DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    print("\nKey Observations:")
    print(f"1. Weight distribution: Check entropy and concentration metrics above")
    print(f"2. Spatial coverage: 95% of points within {coverage['coverage_radius_95']:.4f} distance")
    print(f"3. Parameter adequacy: Check 'Adequate n0' and 'Adequate k' flags above")
    print(f"4. Joint metrics: Label-aware DDC has:")
    print(f"   - Mean error: {metrics_labelaware['mean_err_l2']:.4f} (Random: {metrics_random['mean_err_l2']:.4f})")
    print(f"   - Cov error: {metrics_labelaware['cov_err_fro']:.4f} (Random: {metrics_random['cov_err_fro']:.4f})")
    print(f"   - Corr error: {metrics_labelaware['corr_err_fro']:.4f} (Random: {metrics_random['corr_err_fro']:.4f})")
    
    # Save results
    output_dir = Path(__file__).parent
    comparison_df.to_csv(output_dir / "diagnostic_joint_metrics.csv", index=False)
    print(f"\nResults saved to: {output_dir / 'diagnostic_joint_metrics.csv'}")


if __name__ == "__main__":
    main()

