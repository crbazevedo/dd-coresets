# Density Estimation

## What It Is

Density estimation is the process of estimating how "crowded" or "dense" a region of space is, given a set of data points. In DDC, we use density estimates to identify important regions (modes, clusters) that should be represented in the coreset.

## k-NN Density Estimation

DDC uses **k-nearest neighbors (k-NN)** to estimate local density. Here's the intuition:

### Geometric Intuition

In high dimensions, the volume of a sphere grows exponentially with dimension. This means:
- Points in **dense regions** have many close neighbors → small distance to k-th neighbor
- Points in **sparse regions** have distant neighbors → large distance to k-th neighbor

### Mathematical Intuition

The density estimate for a point `x` is:

```
p(x) ∝ 1 / r_k(x)^d
```

Where:
- `r_k(x)` is the distance to the k-th nearest neighbor
- `d` is the dimensionality
- The `^d` term accounts for the volume of a d-dimensional sphere

**Why this works**: In a dense region, `r_k` is small, so `1/r_k^d` is large (high density). In a sparse region, `r_k` is large, so `1/r_k^d` is small (low density).

## Why This Matters for DDC

DDC uses density estimates to:
1. **Prioritize important regions**: Points with high density (modes, clusters) are more likely to be selected
2. **Preserve the distribution**: By selecting from dense regions, we preserve where most of the data "lives"
3. **Balance with diversity**: We don't select *only* from the densest region—we also ensure spatial coverage

## The Curse of Dimensionality

In high dimensions (d ≥ 20-30), k-NN density estimation becomes unreliable because:
- All points become roughly equidistant
- The volume concentrates in the "shell" of high-dimensional spheres
- Distance differences become less meaningful

**Solution**: DDC uses adaptive distances (Mahalanobis) or PCA reduction for high-dimensional data. See [Adaptive Distances](adaptive_distances.md) for details.

## Example

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Generate data with two clusters
X = np.vstack([
    np.random.randn(1000, 2) + [0, 0],  # Dense cluster at origin
    np.random.randn(100, 2) + [5, 5]    # Sparse cluster
])

# Estimate density using k-NN
k = 10
nn = NearestNeighbors(n_neighbors=k+1)
nn.fit(X)
dists, _ = nn.kneighbors(X)

# Density is inversely proportional to k-th neighbor distance
rk = dists[:, -1]  # k-th neighbor distance
density = 1.0 / (rk ** 2)  # d=2 dimensions
density = density / density.sum()  # Normalize

# Points in dense cluster have higher density
print(f"Dense cluster density: {density[:1000].mean():.4f}")
print(f"Sparse cluster density: {density[1000:].mean():.4f}")
```

## Further Reading

- [Algorithm Overview](algorithm.md) - How density is used in DDC
- [Adaptive Distances](adaptive_distances.md) - Handling high dimensions
- [Understanding Metrics](../guides/understanding_metrics.md) - How to evaluate density preservation

