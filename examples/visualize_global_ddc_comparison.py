#!/usr/bin/env python3
"""
Visual comparison of Global DDC (default and optimized) vs baselines.

Creates visualizations including:
- 2D projections (UMAP if >2D, otherwise direct)
- Marginal distributions
- Covariance/correlation heatmaps
- Spatial coverage plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, fetch_openml
from dd_coresets import fit_ddc_coreset

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. Install with: pip install umap-learn")
    print("Will use PCA for dimensionality reduction instead.")

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


def project_to_2d(X, method='umap', random_state=RANDOM_STATE):
    """Project high-dimensional data to 2D."""
    if X.shape[1] <= 2:
        return X[:, :2]
    
    if method == 'umap' and HAS_UMAP:
        reducer = UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(X)
    else:
        # Fallback to PCA
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(X)


def main():
    print("=" * 70)
    print("VISUAL COMPARISON: GLOBAL DDC vs BASELINES")
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
    
    # Create all subsets/coresets
    print("Creating subsets/coresets...")
    
    # Random
    np.random.seed(RANDOM_STATE)
    random_indices = np.random.choice(len(X_train), size=k_reps, replace=False)
    X_random = X_train[random_indices]
    w_random = np.ones(k_reps) / k_reps
    
    # Stratified
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
    
    # Global DDC default
    print("  Fitting Global DDC (default, n0=None)...")
    S_global_default, w_global_default, _ = fit_ddc_coreset(
        X_train, k=k_reps, n0=None,
        alpha=0.3, gamma=1.0, m_neighbors=32, refine_iters=1,
        reweight_full=True, random_state=RANDOM_STATE,
    )
    
    # Global DDC optimized
    print("  Fitting Global DDC (optimized, n0=None)...")
    S_global_opt, w_global_opt, _ = fit_ddc_coreset(
        X_train, k=k_reps, n0=None,
        alpha=0.1, gamma=2.0, m_neighbors=16, refine_iters=2,
        reweight_full=True, random_state=RANDOM_STATE,
    )
    
    print("Done!")
    print()
    
    # Project to 2D
    print("Projecting to 2D...")
    if X_train.shape[1] > 2:
        if HAS_UMAP:
            print("  Using UMAP for projection")
            method_name = "UMAP"
        else:
            print("  Using PCA for projection (UMAP not available)")
            method_name = "PCA"
    else:
        print("  Using first 2 dimensions")
        method_name = "Direct"
    
    X_train_2d = project_to_2d(X_train, method='umap' if HAS_UMAP else 'pca', random_state=RANDOM_STATE)
    X_random_2d = project_to_2d(X_random, method='umap' if HAS_UMAP else 'pca', random_state=RANDOM_STATE)
    X_strat_2d = project_to_2d(X_strat, method='umap' if HAS_UMAP else 'pca', random_state=RANDOM_STATE)
    S_global_default_2d = project_to_2d(S_global_default, method='umap' if HAS_UMAP else 'pca', random_state=RANDOM_STATE)
    S_global_opt_2d = project_to_2d(S_global_opt, method='umap' if HAS_UMAP else 'pca', random_state=RANDOM_STATE)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "docs" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
    sns.set_palette("husl")
    
    # ========== FIGURE 1: 2D Spatial Coverage ==========
    print("Creating Figure 1: 2D Spatial Coverage...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    methods_2d = [
        ('Random', X_random_2d, w_random, 'blue'),
        ('Stratified', X_strat_2d, w_strat, 'green'),
        ('Global DDC (default)', S_global_default_2d, w_global_default, 'red'),
        ('Global DDC (optimized)', S_global_opt_2d, w_global_opt, 'orange'),
    ]
    
    for ax, (method_name, subset_2d, subset_w, color) in zip(axes, methods_2d):
        # Background: full data (low alpha)
        ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], 
                  c='gray', alpha=0.05, s=1, label='Full Data')
        
        # Overlay: representatives
        sizes = 200 * (subset_w / subset_w.max()) if subset_w.max() > 0 else 50
        ax.scatter(subset_2d[:, 0], subset_2d[:, 1], 
                  c=color, s=sizes, edgecolors='black', linewidth=0.5, 
                  alpha=0.8, label=method_name)
        
        ax.set_xlabel(f'{method_name} Dimension 1')
        ax.set_ylabel(f'{method_name} Dimension 2')
        ax.set_title(f'{method_name} (n={len(subset_2d)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'2D Spatial Coverage Comparison ({method_name} projection)', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'global_ddc_vs_baselines_spatial_coverage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: global_ddc_vs_baselines_spatial_coverage.png")
    
    # ========== FIGURE 2: Marginal Distributions ==========
    print("Creating Figure 2: Marginal Distributions...")
    n_features_to_plot = min(4, X_train.shape[1])
    feature_indices = list(range(n_features_to_plot))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for plot_idx, feat_idx in enumerate(feature_indices):
        ax = axes[plot_idx]
        
        # Full training data (reference)
        ax.hist(X_train[:, feat_idx], bins=50, density=True, alpha=0.3, 
                label='Full Data', color='gray', edgecolor='black')
        
        # Random subset
        ax.hist(X_random[:, feat_idx], bins=30, density=True, alpha=0.6, 
                label='Random', color='blue', histtype='step', linewidth=2)
        
        # Stratified subset
        ax.hist(X_strat[:, feat_idx], bins=30, density=True, alpha=0.6, 
                label='Stratified', color='green', histtype='step', linewidth=2)
        
        # Global DDC default (weighted)
        ax.hist(S_global_default[:, feat_idx], bins=30, weights=w_global_default, 
                density=True, label='Global DDC (default)', color='red', 
                histtype='step', linewidth=2)
        
        # Global DDC optimized (weighted)
        ax.hist(S_global_opt[:, feat_idx], bins=30, weights=w_global_opt, 
                density=True, label='Global DDC (optimized)', color='orange', 
                histtype='step', linewidth=2)
        
        ax.set_xlabel(f'Feature {feat_idx}')
        ax.set_ylabel('Density')
        ax.set_title(f'Marginal Distribution: Feature {feat_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Marginal Distribution Comparison', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'global_ddc_vs_baselines_marginals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: global_ddc_vs_baselines_marginals.png")
    
    # ========== FIGURE 3: Covariance Heatmaps ==========
    print("Creating Figure 3: Covariance Heatmaps...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Full data covariance
    cov_full = np.cov(X_train, rowvar=False)
    im = axes[0, 0].imshow(cov_full, cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_title('Full Data Covariance')
    axes[0, 0].set_xlabel('Feature')
    axes[0, 0].set_ylabel('Feature')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Random
    cov_random = np.cov(X_random, rowvar=False)
    im = axes[0, 1].imshow(cov_random, cmap='RdBu_r', aspect='auto', vmin=cov_full.min(), vmax=cov_full.max())
    axes[0, 1].set_title('Random Covariance')
    axes[0, 1].set_xlabel('Feature')
    axes[0, 1].set_ylabel('Feature')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Stratified
    cov_strat = np.cov(X_strat, rowvar=False)
    im = axes[0, 2].imshow(cov_strat, cmap='RdBu_r', aspect='auto', vmin=cov_full.min(), vmax=cov_full.max())
    axes[0, 2].set_title('Stratified Covariance')
    axes[0, 2].set_xlabel('Feature')
    axes[0, 2].set_ylabel('Feature')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Global DDC default
    cov_global_default = weighted_cov(S_global_default, w_global_default)
    im = axes[1, 0].imshow(cov_global_default, cmap='RdBu_r', aspect='auto', vmin=cov_full.min(), vmax=cov_full.max())
    axes[1, 0].set_title('Global DDC (default) Covariance')
    axes[1, 0].set_xlabel('Feature')
    axes[1, 0].set_ylabel('Feature')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Global DDC optimized
    cov_global_opt = weighted_cov(S_global_opt, w_global_opt)
    im = axes[1, 1].imshow(cov_global_opt, cmap='RdBu_r', aspect='auto', vmin=cov_full.min(), vmax=cov_full.max())
    axes[1, 1].set_title('Global DDC (optimized) Covariance')
    axes[1, 1].set_xlabel('Feature')
    axes[1, 1].set_ylabel('Feature')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Error heatmap (difference from full)
    cov_error_opt = np.abs(cov_full - cov_global_opt)
    im = axes[1, 2].imshow(cov_error_opt, cmap='Reds', aspect='auto')
    axes[1, 2].set_title('Covariance Error (Optimized - Full)')
    axes[1, 2].set_xlabel('Feature')
    axes[1, 2].set_ylabel('Feature')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'global_ddc_vs_baselines_covariance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: global_ddc_vs_baselines_covariance.png")
    
    # ========== FIGURE 4: Correlation Heatmaps ==========
    print("Creating Figure 4: Correlation Heatmaps...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Full data correlation
    corr_full = corr_from_cov(cov_full)
    im = axes[0, 0].imshow(corr_full, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[0, 0].set_title('Full Data Correlation')
    axes[0, 0].set_xlabel('Feature')
    axes[0, 0].set_ylabel('Feature')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Random
    corr_random = corr_from_cov(cov_random)
    im = axes[0, 1].imshow(corr_random, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[0, 1].set_title('Random Correlation')
    axes[0, 1].set_xlabel('Feature')
    axes[0, 1].set_ylabel('Feature')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Stratified
    corr_strat = corr_from_cov(cov_strat)
    im = axes[0, 2].imshow(corr_strat, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[0, 2].set_title('Stratified Correlation')
    axes[0, 2].set_xlabel('Feature')
    axes[0, 2].set_ylabel('Feature')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Global DDC default
    corr_global_default = corr_from_cov(cov_global_default)
    im = axes[1, 0].imshow(corr_global_default, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[1, 0].set_title('Global DDC (default) Correlation')
    axes[1, 0].set_xlabel('Feature')
    axes[1, 0].set_ylabel('Feature')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Global DDC optimized
    corr_global_opt = corr_from_cov(cov_global_opt)
    im = axes[1, 1].imshow(corr_global_opt, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_title('Global DDC (optimized) Correlation')
    axes[1, 1].set_xlabel('Feature')
    axes[1, 1].set_ylabel('Feature')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Error heatmap
    corr_error_opt = np.abs(corr_full - corr_global_opt)
    im = axes[1, 2].imshow(corr_error_opt, cmap='Reds', aspect='auto')
    axes[1, 2].set_title('Correlation Error (Optimized - Full)')
    axes[1, 2].set_xlabel('Feature')
    axes[1, 2].set_ylabel('Feature')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'global_ddc_vs_baselines_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: global_ddc_vs_baselines_correlation.png")
    
    # ========== FIGURE 5: Metrics Bar Chart ==========
    print("Creating Figure 5: Metrics Bar Chart...")
    
    # Compute metrics
    metrics = {
        'Random': {
            'cov_err': np.linalg.norm(cov_full - cov_random, ord='fro'),
            'corr_err': np.linalg.norm(corr_full - corr_random, ord='fro'),
        },
        'Stratified': {
            'cov_err': np.linalg.norm(cov_full - cov_strat, ord='fro'),
            'corr_err': np.linalg.norm(corr_full - corr_strat, ord='fro'),
        },
        'Global DDC\n(default)': {
            'cov_err': np.linalg.norm(cov_full - cov_global_default, ord='fro'),
            'corr_err': np.linalg.norm(corr_full - corr_global_default, ord='fro'),
        },
        'Global DDC\n(optimized)': {
            'cov_err': np.linalg.norm(cov_full - cov_global_opt, ord='fro'),
            'corr_err': np.linalg.norm(corr_full - corr_global_opt, ord='fro'),
        },
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = list(metrics.keys())
    cov_errors = [metrics[m]['cov_err'] for m in methods]
    corr_errors = [metrics[m]['corr_err'] for m in methods]
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # Covariance error
    bars1 = ax1.bar(methods, cov_errors, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Covariance Error (Frobenius)')
    ax1.set_title('Covariance Error Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # Add value labels
    for bar, val in zip(bars1, cov_errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Correlation error
    bars2 = ax2.bar(methods, corr_errors, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Correlation Error (Frobenius)')
    ax2.set_title('Correlation Error Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, corr_errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'global_ddc_vs_baselines_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: global_ddc_vs_baselines_metrics.png")
    
    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    print("  1. global_ddc_vs_baselines_spatial_coverage.png")
    print("  2. global_ddc_vs_baselines_marginals.png")
    print("  3. global_ddc_vs_baselines_covariance.png")
    print("  4. global_ddc_vs_baselines_correlation.png")
    print("  5. global_ddc_vs_baselines_metrics.png")


if __name__ == "__main__":
    main()

