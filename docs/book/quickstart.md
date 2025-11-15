# Quick Start

Get started with DDC in 5 minutes. This guide walks you through creating your first coreset and understanding what it does.

## A Simple Example

Let's start with a concrete example. Suppose you have a dataset of customer transactions with 10,000 records and 8 features (purchase amount, age, location, etc.). You want to create a small summary for analysis:

```python
from dd_coresets import fit_ddc_coreset
import numpy as np

# Your dataset (10k points, 8 features)
# In practice, this would be: X = your_dataframe.values
X = np.random.randn(10000, 8)

# Create a coreset of 200 representatives
S, w, info = fit_ddc_coreset(
    X, 
    k=200,  # Number of representatives (2% of original)
    mode="auto",  # Automatic pipeline selection
    preset="balanced"  # Balanced speed/quality trade-off
)

print(f"Selected {len(S)} representatives")
print(f"Weights sum to: {w.sum():.6f}")  # Should be 1.0
print(f"Compression ratio: {len(X) / len(S):.1f}x smaller")
```

**What happened?** DDC analyzed your 10,000 points and selected 200 real data points that, when weighted, approximate the full distribution. The coreset is 50× smaller but preserves the essential structure.

## Understanding the Output

The `fit_ddc_coreset` function returns three values:

- **`S`**: Array of shape `(k, d)` - the selected representatives. These are **real data points** from your original dataset `X`, not synthetic values. Each row in `S` corresponds to a row in `X`.

- **`w`**: Array of shape `(k,)` - weights that sum to 1. Each weight `w[i]` tells you how much representative `S[i]` "stands for" in the original distribution. A weight of 0.1 means that representative accounts for 10% of the data mass.

- **`info`**: Dictionary with metadata about the coreset, including which pipeline was used, whether PCA was applied, and other configuration details.

**Key insight**: Unlike simple sampling where each selected point represents 1/n of the data, DDC uses weights to allow a small coreset to accurately represent the full distribution. This is similar to how a histogram uses bin counts to represent a distribution, but DDC uses actual data points with weights.

## Why Weights Matter

Unlike simple sampling where each point represents 1/n of the data, DDC assigns **weights** to selected points. This is crucial for accurate distributional approximation.

**The Problem with Uniform Weights**: If you randomly sample 200 points from 10,000, each point represents 1/200 = 0.5% of the data. But if your data has clusters of different sizes, this uniform representation fails—a small cluster might have only 1 point, but it should represent more than 0.5% of the data.

**The Solution: Weighted Representatives**: DDC computes weights so that:
- A point from a dense region (large cluster) gets a higher weight
- A point from a sparse region (small cluster) gets a lower weight
- The weighted coreset preserves the mean, covariance, and marginal distributions

**Analogy**: Think of creating a "summary map" of a country. Instead of placing markers uniformly (which would over-represent small cities), you place markers at important locations and give each marker a "size" (weight) proportional to the population it represents. A marker in a large city might have weight 0.3 (30% of population), while a marker in a small town might have weight 0.01 (1% of population).

This weighted representation allows 200 points to accurately represent 10,000 points, something impossible with uniform sampling.

## Verify Distribution Preservation

Let's verify that the coreset preserves the distribution. We'll compute weighted statistics and compare them to the original:

```python
# Compute weighted statistics from coreset
weighted_mean = (S * w[:, None]).sum(axis=0)
weighted_cov = np.cov(S, rowvar=False, aweights=w)

# Compare to original statistics
original_mean = X.mean(axis=0)
original_cov = np.cov(X, rowvar=False)

mean_error = np.linalg.norm(weighted_mean - original_mean)
cov_error = np.linalg.norm(weighted_cov - original_cov, ord='fro')

print(f"Mean error: {mean_error:.6f}")  # Typically < 0.01
print(f"Covariance error: {cov_error:.4f}")  # Depends on data structure
```

**What to expect**: For well-structured data (clustered, multimodal), mean error is typically less than 0.01. Covariance error depends on the data—DDC preserves local structure better, while random sampling may preserve global covariance better in some cases.

**Why this matters**: These small errors mean you can use the coreset for downstream tasks (visualization, analysis, prototyping) with confidence that it represents the original distribution.

## Next Steps

1. **[Basic Tabular Tutorial](tutorials/basic_tabular.md)** - Learn with a complete example
2. **[Understanding Metrics](guides/understanding_metrics.md)** - Learn how to evaluate coresets
3. **[Choosing Parameters](guides/choosing_parameters.md)** - Learn how to tune DDC

## The Density–Diversity Trade-off

DDC balances two competing objectives:

**Density**: Select points from important regions (modes, clusters) where most of the data "lives". This preserves the distribution by focusing on high-probability regions.

**Diversity**: Ensure spatial coverage so we don't select all points from one cluster. This guarantees that all regions are represented, even sparse ones.

**Why both matter**: If we only optimize for density, we'd select all points from the largest cluster, missing important but smaller clusters. If we only optimize for diversity, we'd spread points uniformly, missing the distribution's structure. DDC balances both.

**The `alpha` parameter** (default 0.3) controls this trade-off:
- **Higher `alpha` (e.g., 0.5)**: Favors density → better distribution preservation, but may under-sample small clusters
- **Lower `alpha` (e.g., 0.1)**: Favors diversity → better spatial coverage, but may sacrifice some distribution fidelity

**When to adjust**: 
- Use higher `alpha` when you have a dominant mode and want to preserve it accurately
- Use lower `alpha` when you have many small clusters and need to ensure all are represented
- The default `alpha=0.3` works well for most cases

This trade-off is fundamental to coreset construction and appears in related work on facility location and diversity sampling (see [Theoretical Foundations](intro.md#theoretical-foundations)).

