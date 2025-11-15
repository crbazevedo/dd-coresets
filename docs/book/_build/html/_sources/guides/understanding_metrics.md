# Understanding Metrics

## Overview

This guide explains how to interpret the metrics used to evaluate DDC coresets and what they mean in practice.

## Quick Reference

| Metric | What It Measures | Lower is Better? | DDC Typically Wins? |
|--------|-----------------|------------------|---------------------|
| Mean Error | Center of distribution | Yes | Yes (50-70% better) |
| Cov Error | Shape and scale | Yes | Sometimes (Random may win) |
| Corr Error | Feature relationships | Yes | Yes (in clustered data) |
| W1 (Wasserstein-1) | Distribution distance | Yes | Yes (20-30% better) |
| KS (Kolmogorov-Smirnov) | Worst-case deviation | Yes | Yes (marginals) |
| MMD | Comprehensive distance | Yes | Yes |

## Distributional Metrics

### Mean Error

**Interpretation**: How well the coreset preserves the "center" of the data.

**Good values**: < 0.1 for normalized data

**When it matters**: Important for preserving the location of the distribution. If mean error is high, the coreset is systematically shifted.

**DDC advantage**: Typically 50-70% better than Random because density-based selection preserves important regions.

### Covariance Error

**Interpretation**: How well the coreset preserves the shape and scale of the distribution.

**Good values**: < 0.5 for normalized data

**When it matters**: Important for preserving correlations and variance. High covariance error means the coreset has different correlations than the original.

**Trade-off**: Random may sometimes win here because it preserves global covariance better. DDC focuses on local structure, which may sacrifice some global covariance.

**When Random wins**: High-dimensional sparse data, uniform distributions, strong global correlations.

### Correlation Error

**Interpretation**: How well the coreset preserves pairwise feature relationships.

**Good values**: < 0.3

**When it matters**: Important for understanding how features relate to each other. DDC often preserves correlations better, especially in clustered data.

### Wasserstein-1 Distance

**Interpretation**: The "cost" of transforming the coreset distribution into the original distribution.

**Intuition**: Imagine moving piles of dirt (coreset) to match another shape (original). W1 is the minimum work needed.

**Good values**: < 0.2 for normalized data

**When it matters**: Especially important for **marginal distributions** (per-feature). Lower W1 means the coreset distribution is closer to the original.

**DDC advantage**: Typically 20-30% better than Random because DDC preserves local structure better.

### Kolmogorov-Smirnov Statistic

**Interpretation**: Maximum difference between cumulative distribution functions.

**Intuition**: The worst-case deviation in the distribution. High KS means there's a region where the coreset deviates significantly.

**Good values**: < 0.1

**When it matters**: Useful for detecting systematic biases. If KS is high, the coreset is missing or over-representing some region.

## Spatial Metrics

### Coverage per Cluster

**Interpretation**: How many coreset points fall in each cluster/region.

**Good values**: All clusters should have at least 1 representative

**When it matters**: Important for ensuring all modes/regions are represented. DDC guarantees coverage of all clusters, even small ones.

**DDC advantage**: Guarantees coverage, while Random may miss small clusters entirely.

## Interpreting Results

### DDC vs Random: What to Expect

**DDC typically wins in**:
- Mean error (50-70% better)
- Wasserstein-1 (20-30% better)
- Spatial coverage (guarantees all clusters)
- Correlation error (in clustered data)

**Random may win in**:
- Covariance error (in high-dimensional sparse data)
- Global structure preservation (in uniform distributions)

**Trade-off**: DDC preserves **local structure** (clusters, modes) better, while Random may preserve **global structure** (covariance) better.

### Practical Guidelines

**Use DDC when**:
- Mean error or W1 are important
- You have clustered data
- You need spatial coverage
- You have small k

**Use Random when**:
- Preserving exact global covariance is critical
- Data is high-dimensional and sparse
- You have uniform distributions

## Example: Interpreting Results

```python
from dd_coresets import fit_ddc_coreset, fit_random_coreset
import numpy as np

X = np.random.randn(10000, 10)
k = 200

# DDC
S_ddc, w_ddc, _ = fit_ddc_coreset(X, k=k)
# Compute metrics...

# Random
S_rnd, w_rnd, _ = fit_random_coreset(X, k=k)
# Compute metrics...

# Compare
print("Mean Error:")
print(f"  DDC: {mean_err_ddc:.4f}")
print(f"  Random: {mean_err_rnd:.4f}")
print(f"  DDC is {((mean_err_rnd - mean_err_ddc) / mean_err_rnd * 100):.1f}% better")

print("\nWasserstein-1:")
print(f"  DDC: {w1_ddc:.4f}")
print(f"  Random: {w1_rnd:.4f}")
print(f"  DDC is {((w1_rnd - w1_ddc) / w1_rnd * 100):.1f}% better")
```

## Further Reading

- [Metrics Concepts](../concepts/metrics.md) - Detailed explanation of each metric
- [When to Use DDC](../../DDC_ADVANTAGE_CASES.md) - Analysis of DDC advantages
- [Choosing Parameters](choosing_parameters.md) - How parameters affect metrics

