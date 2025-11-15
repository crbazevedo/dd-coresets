# Quick Start

Get started with DDC in 5 minutes.

## Basic Usage

```python
from dd_coresets import fit_ddc_coreset
import numpy as np

# Generate or load your dataset
X = np.random.randn(10000, 8)  # 10k points, 8 features

# Create a coreset
S, w, info = fit_ddc_coreset(
    X, 
    k=200,  # Number of representatives
    mode="auto",  # Automatic pipeline selection
    preset="balanced"  # Balanced speed/quality trade-off
)

print(f"Selected {len(S)} representatives")
print(f"Weights sum to: {w.sum():.6f}")
```

## Understanding the Output

- **`S`**: Array of shape `(k, d)` - the selected representatives (real data points)
- **`w`**: Array of shape `(k,)` - weights that sum to 1
- **`info`**: Dictionary with metadata about the coreset

## Conceptual Note: Why Weights?

Unlike simple sampling, DDC assigns **weights** to selected points. This allows a small coreset to accurately represent the full distribution:

- A point with weight 0.1 "stands for" 10% of the original data in that region
- Weights are computed to preserve distributional properties (mean, covariance, marginals)
- Think of it like creating a "summary map" where each landmark (representative) has a size (weight) proportional to how much territory it represents

## Use the Coreset

```python
# Compute weighted statistics
weighted_mean = (S * w[:, None]).sum(axis=0)
weighted_cov = np.cov(S, rowvar=False, aweights=w)

# Compare to original
original_mean = X.mean(axis=0)
original_cov = np.cov(X, rowvar=False)

print("Mean error:", np.linalg.norm(weighted_mean - original_mean))
print("Cov error:", np.linalg.norm(weighted_cov - original_cov, ord='fro'))
```

## Next Steps

1. **[Basic Tabular Tutorial](tutorials/basic_tabular.md)** - Learn with a complete example
2. **[Understanding Metrics](guides/understanding_metrics.md)** - Learn how to evaluate coresets
3. **[Choosing Parameters](guides/choosing_parameters.md)** - Learn how to tune DDC

## Conceptual Note: Density–Diversity Trade-off

DDC balances two objectives:

- **Density**: Select points from important regions (modes, clusters) to preserve the distribution
- **Diversity**: Ensure spatial coverage so we don't select all points from one cluster

The `alpha` parameter (default 0.3) controls this trade-off:
- Higher `alpha` (e.g., 0.5): Favors density → better distribution preservation, but may miss small clusters
- Lower `alpha` (e.g., 0.1): Favors diversity → better spatial coverage, but may sacrifice distribution fidelity

The default `alpha=0.3` provides a good balance for most use cases.

