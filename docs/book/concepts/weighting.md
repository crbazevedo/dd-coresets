# Weight Assignment

## Why Weights Matter

Unlike simple sampling, DDC assigns **weights** to selected representatives. This allows a small coreset to accurately represent the full distribution:

- A point with weight 0.1 "stands for" 10% of the original data in that region
- Weights are computed to preserve distributional properties (mean, covariance, marginals)
- Think of it like creating a "summary map" where each landmark (representative) has a size (weight) proportional to how much territory it represents

## Soft Assignments

DDC uses **soft assignments** instead of hard assignments:

- **Hard assignment**: Each original point belongs to exactly one representative
- **Soft assignment**: Each original point contributes to multiple representatives (with different weights)

**Why soft?**: Soft assignments are more robust and better approximate the distribution. They create a smooth, weighted approximation.

## Kernel-Based Weighting

Weights are computed using a **kernel function** (typically RBF):

```
w_j = Σ_i kernel(x_i, s_j) / Σ_j Σ_i kernel(x_i, s_j)
```

Where:
- `kernel(x_i, s_j)` is large if `x_i` is close to representative `s_j`
- Weights are normalized so they sum to 1

**Intuition**: 
- Points close to a representative get high weight for that representative
- Points far from all representatives get distributed weights across nearby representatives
- This creates a smooth, continuous approximation

## Weight Interpretation

**Probability interpretation**: Weights can be interpreted as probabilities. The coreset `(S, w)` defines a probability distribution:

```
P(x) = Σ_j w_j × kernel(x, s_j)
```

This distribution approximates the original empirical distribution.

**Mass interpretation**: A representative with weight 0.1 "represents" 10% of the original data mass in its region.

## Normalization

Weights are always normalized so `Σ w_j = 1`. This ensures:
- The coreset defines a valid probability distribution
- Weighted statistics (mean, covariance) are properly scaled
- The coreset can be used directly in downstream tasks

## Reweighting on Full Dataset

By default, DDC computes weights on a working sample (`n0` points) for efficiency. The optional `reweight_full=True` parameter recomputes weights on the full dataset:

**When to use**:
- You want maximum accuracy
- The working sample may not be fully representative
- You have time for the extra computation

**When to skip**:
- Working sample is large enough (n0 ≈ n)
- You need fastest computation
- The difference is negligible

## Example

```python
from dd_coresets import fit_ddc_coreset
import numpy as np

X = np.random.randn(10000, 10)
S, w, info = fit_ddc_coreset(X, k=200, reweight_full=True)

# Weights sum to 1
print(f"Weights sum: {w.sum():.6f}")

# Weighted mean
weighted_mean = (S * w[:, None]).sum(axis=0)

# Compare to original
original_mean = X.mean(axis=0)
print(f"Mean error: {np.linalg.norm(weighted_mean - original_mean):.4f}")
```

## Further Reading

- [Algorithm Overview](algorithm.md) - How weights are computed in DDC
- [Understanding Metrics](metrics.md) - How to evaluate weighted coresets
- [Choosing Parameters](../guides/choosing_parameters.md) - When to use `reweight_full`

