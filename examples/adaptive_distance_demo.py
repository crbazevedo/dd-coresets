#!/usr/bin/env python3
"""
Quick 2D demo of adaptive distances for density estimation.

Demonstrates how Mahalanobis adaptive distances help in density estimation
compared to Euclidean distances, especially for elliptical clusters.

Uses the new DDC API with mode="euclidean" vs mode="adaptive".
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from dd_coresets.ddc import fit_ddc_coreset

# Set random seed for reproducibility
np.random.seed(42)


def ecdf(x):
    """Empirical CDF."""
    sorted_x = np.sort(x)
    n = len(x)
    y = np.arange(1, n + 1) / n
    return sorted_x, y


def ks_1d_approx(x1, x2, w2=None):
    """KS statistic (approximate for weighted)."""
    if w2 is not None:
        n_samples = min(5000, len(x2))
        probs = w2 / w2.sum()
        idx = np.random.choice(len(x2), size=n_samples, p=probs)
        x2_sampled = x2[idx]
    else:
        x2_sampled = x2

    x1_sorted, y1 = ecdf(x1)
    x2_sorted, y2 = ecdf(x2_sampled)

    all_x = np.unique(np.concatenate([x1_sorted, x2_sorted]))
    y1_interp = np.interp(all_x, x1_sorted, y1)
    y2_interp = np.interp(all_x, x2_sorted, y2)
    return np.abs(y1_interp - y2_interp).max()


def wasserstein_1d_approx(x1, x2, w2=None):
    """Wasserstein-1 distance (1D, approximate for weighted)."""
    if w2 is not None:
        n_samples = min(5000, len(x2))
        probs = w2 / w2.sum()
        idx = np.random.choice(len(x2), size=n_samples, p=probs)
        x2_sampled = x2[idx]
    else:
        x2_sampled = x2

    x1_sorted = np.sort(x1)
    x2_sorted = np.sort(x2_sampled)

    n1, n2 = len(x1_sorted), len(x2_sorted)
    quantiles = np.linspace(0, 1, min(n1, n2))
    q1 = np.quantile(x1_sorted, quantiles)
    q2 = np.quantile(x2_sorted, quantiles)
    return np.abs(q1 - q2).mean()


def main():
    """Run adaptive distance demo."""
    print("=" * 70)
    print("Adaptive Distance Demo: Euclidean vs Mahalanobis")
    print("=" * 70)
    
    # Generate elliptical cluster
    n_samples = 2000
    X, labels = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=1,
        cluster_std=[2.0, 0.5],  # Elliptical
        random_state=42
    )
    
    # Rotate to make it more elliptical
    angle = np.pi / 4
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    X = X @ rotation.T
    
    print(f"\nDataset: {n_samples} points, 2D elliptical cluster")
    print(f"Cluster std: [2.0, 0.5] (rotated 45°)")
    
    k = 100
    print(f"\nComputing DDC coresets (k={k})...")
    
    # Euclidean
    print("  Euclidean mode...")
    S_eucl, w_eucl, info_eucl = fit_ddc_coreset(
        X, k=k, mode="euclidean", preset="balanced", random_state=42
    )
    
    # Adaptive
    print("  Adaptive mode...")
    S_adapt, w_adapt, info_adapt = fit_ddc_coreset(
        X, k=k, mode="adaptive", preset="balanced", random_state=42
    )
    
    # Compute metrics
    print("\nComputing distribution metrics...")
    ks_eucl = [ks_1d_approx(X[:, i], S_eucl[:, i], w_eucl) for i in range(2)]
    ks_adapt = [ks_1d_approx(X[:, i], S_adapt[:, i], w_adapt) for i in range(2)]
    
    w1_eucl = [wasserstein_1d_approx(X[:, i], S_eucl[:, i], w_eucl) for i in range(2)]
    w1_adapt = [wasserstein_1d_approx(X[:, i], S_adapt[:, i], w_adapt) for i in range(2)]
    
    print(f"\nKS Statistic (mean):")
    print(f"  Euclidean: {np.mean(ks_eucl):.4f}")
    print(f"  Adaptive:  {np.mean(ks_adapt):.4f}")
    print(f"  Improvement: {(1 - np.mean(ks_adapt)/np.mean(ks_eucl))*100:.1f}%")
    
    print(f"\nWasserstein-1 (mean):")
    print(f"  Euclidean: {np.mean(w1_eucl):.4f}")
    print(f"  Adaptive:  {np.mean(w1_adapt):.4f}")
    print(f"  Improvement: {(1 - np.mean(w1_adapt)/np.mean(w1_eucl))*100:.1f}%")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    axes[0].scatter(X[:, 0], X[:, 1], c='gray', alpha=0.3, s=10)
    axes[0].set_title('Original Data (Elliptical Cluster)')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].grid(True, alpha=0.3)
    
    # Euclidean coreset
    scatter1 = axes[1].scatter(
        S_eucl[:, 0], S_eucl[:, 1], c=w_eucl, cmap='viridis', s=w_eucl*1000, alpha=0.7, edgecolors='black', linewidth=0.5
    )
    axes[1].scatter(X[:, 0], X[:, 1], c='gray', alpha=0.1, s=2)
    axes[1].set_title(f'DDC Coreset (Euclidean, k={k})')
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[1], label='Weight')
    
    # Adaptive coreset
    scatter2 = axes[2].scatter(
        S_adapt[:, 0], S_adapt[:, 1], c=w_adapt, cmap='viridis', s=w_adapt*1000, alpha=0.7, edgecolors='black', linewidth=0.5
    )
    axes[2].scatter(X[:, 0], X[:, 1], c='gray', alpha=0.1, s=2)
    axes[2].set_title(f'DDC Coreset (Adaptive, k={k})')
    axes[2].set_xlabel('X1')
    axes[2].set_ylabel('X2')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[2], label='Weight')
    
    plt.tight_layout()
    
    # Save or show
    output_path = 'adaptive_distance_demo.png'
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
    except:
        print("\n⚠ Could not save figure (displaying instead)")
        plt.show()
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nKey insight: Adaptive distances better capture elliptical cluster shape,")
    print("leading to better distribution preservation in the coreset.")
    print(f"\nPipeline info (Euclidean): {info_eucl['pipeline']}")
    print(f"Pipeline info (Adaptive): {info_adapt['pipeline']}")


if __name__ == "__main__":
    main()

