"""
Shared utilities for DDC advantage experiments.

Provides standardized functions for:
- Metrics computation (joint and marginal)
- Visualization
- Dataset generation helpers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dd_coresets import fit_ddc_coreset

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ========== Metrics Computation ==========

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


def compute_all_metrics(X_full, S, w, method_name: str) -> Dict:
    """
    Compute all metrics: joint and marginal.
    
    Returns dictionary with all metrics.
    """
    # Joint distribution metrics
    mu_full = X_full.mean(axis=0)
    cov_full = np.cov(X_full, rowvar=False)
    corr_full = corr_from_cov(cov_full)
    
    mu_coreset = weighted_mean(S, w)
    cov_coreset = weighted_cov(S, w)
    corr_coreset = corr_from_cov(cov_coreset)
    
    mean_err = np.linalg.norm(mu_full - mu_coreset)
    cov_err = np.linalg.norm(cov_full - cov_coreset, ord='fro')
    corr_err = np.linalg.norm(corr_full - corr_coreset, ord='fro')
    mmd = compute_mmd(X_full, S, w_Y=w)
    
    # Marginal distribution metrics
    d = X_full.shape[1]
    W1_dims = []
    KS_dims = []
    
    for dim in range(d):
        seed = RANDOM_STATE + dim
        W1 = wasserstein_1d_approx(X_full[:, dim], S[:, dim], w, 
                                   n_samples=5000, random_state=seed)
        KS = ks_1d_approx(X_full[:, dim], S[:, dim], w, n_grid=512)
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


def compute_spatial_coverage(X_full, S, labels_full=None, labels_coreset=None):
    """
    Compute spatial coverage metrics.
    
    Returns:
    - coverage_per_cluster: dict mapping cluster_id to coverage ratio
    - min_distances: minimum distance from each coreset point to full data
    - mean_distances: mean distance from each coreset point to full data
    """
    if labels_full is None:
        # Single cluster case
        return {
            'coverage_per_cluster': {0: 1.0},
            'min_distances': np.zeros(len(S)),
            'mean_distances': np.zeros(len(S)),
        }
    
    # Compute distances
    try:
        from scipy.spatial.distance import cdist
        distances = cdist(S, X_full)
    except ImportError:
        # Fallback: manual computation
        distances = np.sqrt(((S[:, None, :] - X_full[None, :, :]) ** 2).sum(axis=2))
    
    min_distances = distances.min(axis=1)
    mean_distances = distances.mean(axis=1)
    
    # Coverage per cluster
    coverage_per_cluster = {}
    unique_labels = np.unique(labels_full)
    
    for label in unique_labels:
        cluster_mask = (labels_full == label)
        cluster_size = np.sum(cluster_mask)
        
        if labels_coreset is not None:
            coreset_mask = (labels_coreset == label)
            coreset_size = np.sum(coreset_mask)
        else:
            # Approximate: count coreset points closest to this cluster
            cluster_distances = distances[:, cluster_mask]
            if cluster_distances.shape[1] > 0:
                min_dist_to_cluster = cluster_distances.min(axis=1)
                # Count points where this cluster is the closest
                all_min_dists = []
                for other_label in unique_labels:
                    if other_label != label:
                        other_mask = (labels_full == other_label)
                        if other_mask.any():
                            other_distances = distances[:, other_mask]
                            all_min_dists.append(other_distances.min(axis=1))
                if all_min_dists:
                    other_min_dists = np.array(all_min_dists).min(axis=0)
                    coreset_size = np.sum(min_dist_to_cluster < other_min_dists)
                else:
                    coreset_size = len(S)
            else:
                coreset_size = 0
        
        coverage_per_cluster[label] = coreset_size / max(cluster_size, 1)
    
    return {
        'coverage_per_cluster': coverage_per_cluster,
        'min_distances': min_distances,
        'mean_distances': mean_distances,
    }


# ========== Visualization ==========

def plot_spatial_coverage_2d(X_full, S_random, w_random, S_ddc, w_ddc, 
                             labels_full=None, title="Spatial Coverage Comparison",
                             output_path=None):
    """Plot 2D spatial coverage comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    methods = [
        ('Random', S_random, w_random, 'blue'),
        ('DDC', S_ddc, w_ddc, 'orange'),
    ]
    
    for ax, (method_name, S, w, color) in zip(axes, methods):
        # Background: full data
        if labels_full is not None:
            scatter = ax.scatter(X_full[:, 0], X_full[:, 1], 
                               c=labels_full, cmap='tab10', alpha=0.1, s=1)
        else:
            ax.scatter(X_full[:, 0], X_full[:, 1], 
                      c='gray', alpha=0.1, s=1, label='Full Data')
        
        # Overlay: representatives
        sizes = 200 * (w / w.max()) if w.max() > 0 else 50
        ax.scatter(S[:, 0], S[:, 1], 
                  c=color, s=sizes, edgecolors='black', linewidth=0.5, 
                  alpha=0.8, label=method_name)
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'{method_name} (n={len(S)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_marginal_distributions(X_full, S_random, w_random, S_ddc, w_ddc,
                                n_features=4, title="Marginal Distribution Comparison",
                                output_path=None):
    """Plot marginal distributions comparison."""
    n_features = min(n_features, X_full.shape[1])
    feature_indices = list(range(n_features))
    
    n_cols = 2
    n_rows = (n_features + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for plot_idx, feat_idx in enumerate(feature_indices):
        ax = axes[plot_idx]
        
        # Full training data (reference)
        ax.hist(X_full[:, feat_idx], bins=50, density=True, alpha=0.3, 
                label='Full Data', color='gray', edgecolor='black')
        
        # Random subset
        ax.hist(S_random[:, feat_idx], bins=30, density=True, alpha=0.6, 
                label='Random', color='blue', histtype='step', linewidth=2)
        
        # DDC (weighted)
        ax.hist(S_ddc[:, feat_idx], bins=30, weights=w_ddc, 
                density=True, label='DDC', color='orange', 
                histtype='step', linewidth=2)
        
        ax.set_xlabel(f'Feature {feat_idx}')
        ax.set_ylabel('Density')
        ax.set_title(f'Marginal Distribution: Feature {feat_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metrics_comparison(metrics_random, metrics_ddc, output_path=None):
    """Plot bar chart comparing metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Joint metrics
    joint_metrics = ['mean_err_l2', 'cov_err_fro', 'corr_err_fro', 'mmd']
    joint_labels = ['Mean Error (L2)', 'Cov Error (Fro)', 'Corr Error (Fro)', 'MMD']
    
    for ax, metric, label in zip(axes[0, :], joint_metrics[:2], joint_labels[:2]):
        values = [metrics_random[metric], metrics_ddc[metric]]
        colors = ['blue', 'orange']
        bars = ax.bar(['Random', 'DDC'], values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    for ax, metric, label in zip(axes[1, :], joint_metrics[2:], joint_labels[2:]):
        values = [metrics_random[metric], metrics_ddc[metric]]
        colors = ['blue', 'orange']
        bars = ax.bar(['Random', 'DDC'], values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ========== Coreset Fitting ==========

def fit_random_coreset(X, k, random_state=None):
    """Fit random coreset."""
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.random.choice(len(X), size=k, replace=False)
    S = X[indices]
    w = np.ones(k) / k
    
    return S, w


def fit_ddc_coreset_optimized(X, k, n0=None, random_state=None):
    """Fit DDC coreset with optimized parameters."""
    if random_state is None:
        random_state = RANDOM_STATE
    
    S, w, info = fit_ddc_coreset(
        X, k=k, n0=n0,
        alpha=0.1, gamma=2.0, m_neighbors=16, refine_iters=2,
        reweight_full=True, random_state=random_state,
    )
    
    return S, w, info


# ========== Results Saving ==========

def save_results(metrics_random, metrics_ddc, experiment_name, output_dir):
    """Save results to CSV and text table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame([metrics_random, metrics_ddc])
    
    # Save CSV
    csv_path = output_dir / f"{experiment_name}_metrics.csv"
    comparison_df.to_csv(csv_path, index=False)
    
    # Save text table
    txt_path = output_dir / f"{experiment_name}_comparison_table.txt"
    with open(txt_path, 'w') as f:
        f.write(f"Comparison: {experiment_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # Add relative comparison
        f.write("Relative Comparison (DDC vs Random):\n")
        f.write("-" * 70 + "\n")
        for metric in ['mean_err_l2', 'cov_err_fro', 'corr_err_fro', 'mmd', 
                       'W1_mean', 'W1_max', 'KS_mean', 'KS_max']:
            random_val = metrics_random[metric]
            ddc_val = metrics_ddc[metric]
            if random_val > 0:
                pct_change = (ddc_val / random_val - 1) * 100
                f.write(f"{metric:20s}: {pct_change:+.1f}%\n")
    
    return csv_path, txt_path

