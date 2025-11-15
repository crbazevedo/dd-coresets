# High-Dimensional Data

## Use Case: Dimensionality Reduction and Density Estimation

### The Problem

You have high-dimensional data (d ≥ 30)—perhaps features from a neural network, gene expression data, or text embeddings. You want to create a coreset, but:

- k-NN density estimation becomes unreliable in high dimensions (curse of dimensionality)
- Euclidean distance becomes less meaningful
- Computation becomes expensive

### The Solution: Automatic PCA + Adaptive Distances

DDC automatically handles high-dimensional data by applying PCA reduction and using adaptive distances when appropriate.

## Example: High-Dimensional Feature Space

```python
from dd_coresets import fit_ddc_coreset
import numpy as np

# High-dimensional dataset (e.g., neural network features)
# X = extract_features(model, images)  # 100k images, 512 features

# For demonstration, generate synthetic high-dimensional data
X = np.random.randn(50000, 60)  # 50k points, 60 features

# DDC automatically applies PCA when d ≥ 30
S, w, info = fit_ddc_coreset(
    X,
    k=500,
    mode="auto",  # Automatically chooses PCA + adaptive
    preset="balanced"
)

print(f"Original dimensions: {X.shape[1]}")
print(f"PCA applied: {info['pipeline']['pca_used']}")
if info['pipeline']['pca_used']:
    print(f"Reduced dimensions: {info['pipeline']['d_effective']}")
    print(f"Explained variance: {info['pipeline']['explained_variance_ratio']:.2%}")
print(f"Representatives shape: {S.shape}")  # Still (500, 60) - original space!
```

**Key point**: Representatives `S` are always returned in the **original feature space** (60D), not the reduced space. DDC uses PCA internally for density estimation but maps representatives back to original dimensions.

## Why PCA is Necessary

**The Curse of Dimensionality**: In high dimensions (d ≥ 30), several problems occur:

1. **Distance concentration**: All points become roughly equidistant
2. **Volume concentration**: Volume concentrates in the "shell" of high-dimensional spheres
3. **Density estimation failure**: k-NN density estimates become uniform (no discrimination)

**PCA Solution**: By reducing to 20-50 dimensions while retaining 95% variance, we:
- Make density estimation feasible
- Preserve the essential structure of the data
- Speed up computation

**Mathematical intuition**: PCA finds the directions of maximum variance. By working in this reduced space, we can estimate density accurately, then project representatives back to original space.

## Performance Comparison

For a dataset with 50k points and 60 features:

- **Euclidean (no PCA)**: Slow (8-10s), unreliable density estimates
- **Auto (with PCA)**: Fast (3-5s), accurate density estimates, 1.5-2× speedup

## Understanding the Pipeline

When `mode="auto"` and `d ≥ 30`, DDC:

1. **Applies PCA**: Reduces to 20-50 dimensions (retaining 95% variance)
2. **Estimates density**: Uses k-NN in reduced space
3. **Selects representatives**: Greedy selection in reduced space
4. **Maps back**: Returns representatives in original space

**Why this works**: The representatives selected in reduced space are still valid in original space because PCA preserves the essential structure. The mapping back ensures you get real data points in the original feature space.

## Configuration

You can control PCA behavior via `pipeline_cfg`:

```python
S, w, info = fit_ddc_coreset(
    X,
    k=500,
    mode="auto",
    preset="manual",
    pipeline_cfg={
        "dim_threshold_adaptive": 25,  # Trigger PCA at d ≥ 25
        "retain_variance": 0.95,  # Retain 95% variance
        "cap_components": 50  # Maximum 50 components
    }
)
```

## Best Practices

1. **Use `mode="auto"`**: Let DDC decide when to apply PCA
2. **Check explained variance**: Ensure PCA retains ≥90% variance
3. **Verify in original space**: Always check that representatives are in original feature space
4. **Compare with/without PCA**: For d ≈ 30, compare both to see if PCA helps

## Conceptual Note: Curse of Dimensionality

The curse of dimensionality is a fundamental problem in high-dimensional statistics. As dimensions increase:

- **Volume grows exponentially**: A sphere's volume grows as r^d
- **Distance differences vanish**: All pairwise distances become similar
- **Density estimation fails**: k-NN can't distinguish dense from sparse regions

**DDC's solution**: PCA reduces dimensions to a manageable size (typically 20-50) where density estimation is reliable, while adaptive distances (Mahalanobis) account for the shape of the data in the reduced space.

See [Density Estimation](../concepts/density_estimation.md) and [Adaptive Distances](../concepts/adaptive_distances.md) for detailed explanations.

## Further Reading

- [High-Dimensional Tutorial](../tutorials/high_dimensional.md) - Complete example
- [Adaptive Distances](../concepts/adaptive_distances.md) - Mahalanobis distance explanation
- [Density Estimation](../concepts/density_estimation.md) - Curse of dimensionality

