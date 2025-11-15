# Adaptive Distances

## The Problem: Curse of Dimensionality

In high dimensions (d ≥ 20-30), Euclidean distance becomes less meaningful:
- All points become roughly equidistant
- Volume concentrates in the "shell" of high-dimensional spheres
- k-NN density estimates become unreliable

**Result**: DDC's density-based selection becomes ineffective.

## The Solution: Mahalanobis Distance

Adaptive distances use **local Mahalanobis distance** instead of Euclidean distance. This accounts for the **shape** of the data.

### Intuition

Instead of measuring distance in the original space, we:
1. Estimate the **local covariance** around each point
2. **Stretch** the space along principal directions
3. Make clusters more "spherical" in the transformed space

**Result**: Density estimation becomes more accurate, even in high dimensions.

### Mathematical Intuition

**Euclidean distance**: `||x - y||² = (x-y)ᵀ (x-y)`

**Mahalanobis distance**: `||x - y||²_M = (x-y)ᵀ C⁻¹ (x-y)`

Where `C` is the local covariance matrix. This "stretches" the space:
- Directions with high variance → shorter distances
- Directions with low variance → longer distances

**Effect**: Clusters become more "spherical" in the transformed space, making density estimation more reliable.

## How It Works in DDC

1. **Local covariance estimation**: For each point, estimate covariance from its k nearest neighbors
2. **Shrinkage**: Apply OAS (Oracle Approximating Shrinkage) for robust estimation
3. **Cholesky decomposition**: Compute Mahalanobis distances efficiently (no matrix inversion)
4. **Density estimation**: Use Mahalanobis distances for k-NN density estimation

## When to Use Adaptive Distances

**Use adaptive when**:
- d ≥ 20 (medium to high dimensions)
- Data has **elliptical clusters** (not spherical)
- You want better **marginal distribution preservation**

**Use Euclidean when**:
- d < 20 (low dimensions)
- Clusters are roughly spherical
- You want fastest computation

**Auto mode**: DDC automatically chooses based on dimensionality:
- d < 20: Euclidean
- 20 ≤ d < 30: Adaptive (if feasible)
- d ≥ 30: PCA reduction → Adaptive

## Example

```python
from dd_coresets import fit_ddc_coreset
import numpy as np

# Generate elliptical clusters
X = np.vstack([
    np.random.randn(1000, 15) @ np.diag([3, 1, 1, ...]),  # Elongated
    np.random.randn(1000, 15) @ np.diag([1, 3, 1, ...])   # Different orientation
])

# Euclidean (may miss elliptical structure)
S_euc, w_euc, _ = fit_ddc_coreset(X, k=200, mode="euclidean")

# Adaptive (better for elliptical)
S_adapt, w_adapt, _ = fit_ddc_coreset(X, k=200, mode="adaptive")

# Compare: Adaptive typically has 20-30% better W1
```

## Further Reading

- [Density Estimation](density_estimation.md) - How adaptive distances improve density estimation
- [Algorithm Overview](algorithm.md) - How adaptive distances fit into DDC
- [ADAPTIVE_DISTANCES_EXPLAINED.md](../../ADAPTIVE_DISTANCES_EXPLAINED.md) - Technical deep-dive

