#!/usr/bin/env python3
"""
Quick 2D demo of adaptive distances for density estimation.

Demonstrates how Mahalanobis adaptive distances help in density estimation
compared to Euclidean distances, especially for elliptical clusters.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs

# Set random seed for reproducibility
np.random.seed(42)


def estimate_density_euclidean(X, k=32):
    """Estimate density using Euclidean k-NN."""
    n, d = X.shape
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    kth_distances = distances[:, k]
    densities = k / (kth_distances ** d + 1e-10)
    densities /= densities.sum()
    return densities


def estimate_density_adaptive(X, k=32, n_iter=3):
    """Estimate density using adaptive Mahalanobis distances."""
    n, d = X.shape
    densities = np.ones(n) / n
    
    # Initialize with Euclidean
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    kth_distances = distances[:, k]
    densities = k / (kth_distances ** d + 1e-10)
    
    # Iterative refinement
    for iteration in range(n_iter):
        new_densities = np.zeros(n)
        
        for i in range(n):
            neighbor_indices = indices[i, 1:k+1]
            neighbors = X[neighbor_indices]
            
            if len(neighbors) > d:
                # Local covariance
                local_mean = neighbors.mean(axis=0)
                local_cov = np.cov(neighbors.T)
                local_cov += np.eye(d) * 1e-6
                
                try:
                    # Mahalanobis distances
                    inv_cov = np.linalg.inv(local_cov)
                    diff = X - local_mean
                    mahal_distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                    
                    # k-th Mahalanobis neighbor
                    sorted_idx = np.argsort(mahal_distances)
                    rk_mahal = mahal_distances[sorted_idx[k]]
                    
                    # Adaptive density
                    det_cov = np.linalg.det(local_cov)
                    new_densities[i] = k / (rk_mahal ** d * np.sqrt(det_cov) + 1e-10)
                except:
                    new_densities[i] = densities[i]
            else:
                new_densities[i] = densities[i]
        
        densities = new_densities
    
    densities /= densities.sum()
    return densities


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
    
    # Estimate densities
    print("\nEstimating densities...")
    densities_euclidean = estimate_density_euclidean(X, k=32)
    densities_adaptive = estimate_density_adaptive(X, k=32, n_iter=3)
    
    # Compute statistics
    cv_euclidean = np.std(densities_euclidean) / np.mean(densities_euclidean)
    cv_adaptive = np.std(densities_adaptive) / np.mean(densities_adaptive)
    
    print(f"\nCoefficient of Variation:")
    print(f"  Euclidean: {cv_euclidean:.4f}")
    print(f"  Adaptive:  {cv_adaptive:.4f}")
    print(f"  Improvement: {cv_adaptive / cv_euclidean:.2f}x")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    axes[0].scatter(X[:, 0], X[:, 1], c='gray', alpha=0.3, s=10)
    axes[0].set_title('Original Data (Elliptical Cluster)')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].grid(True, alpha=0.3)
    
    # Euclidean densities
    scatter1 = axes[1].scatter(X[:, 0], X[:, 1], c=densities_euclidean, 
                               cmap='viridis', s=20, alpha=0.6)
    axes[1].set_title('Density (Euclidean)')
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[1])
    
    # Adaptive densities
    scatter2 = axes[2].scatter(X[:, 0], X[:, 1], c=densities_adaptive,
                               cmap='viridis', s=20, alpha=0.6)
    axes[2].set_title('Density (Adaptive Mahalanobis)')
    axes[2].set_xlabel('X1')
    axes[2].set_ylabel('X2')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[2])
    
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
    print("leading to more accurate density estimates.")


if __name__ == "__main__":
    main()

