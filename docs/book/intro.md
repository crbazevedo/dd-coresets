# Welcome to dd-coresets

**Density–Diversity Coresets (DDC)**: a small weighted set of *real* data points that approximates the empirical distribution of a large dataset.

## What is DDC?

DDC selects real data points from your dataset and assigns weights to them, creating a small "summary" that preserves the distributional properties of the original data. Unlike random sampling, DDC:

- **Guarantees spatial coverage** of all clusters and modes
- **Preserves marginal distributions** better than random sampling
- **Selects real data points** (not synthetic centroids) for interpretability
- **Works unsupervised** (no domain knowledge required)

## Why Use DDC?

Large datasets are ubiquitous in data science, but many workflows require small, interpretable subsets:

- **Exploratory plots and dashboards** need small, representative samples
- **Scenario analysis and simulations** need few representative points with weights
- **Prototyping models** is faster on coresets than on full data
- **Distributional stress testing** requires faithful approximation of the empirical distribution

## Quick Example

```python
from dd_coresets import fit_ddc_coreset
import numpy as np

# Your large dataset
X = np.random.randn(100000, 10)  # 100k points, 10 features

# Create a coreset of 500 weighted points
S, w, info = fit_ddc_coreset(X, k=500, mode="auto", preset="balanced")

# S: 500 representatives (real data points)
# w: weights (sum to 1)
# Use S and w to approximate the full distribution
```

## Documentation Structure

This documentation is organized into:

- **Tutorials**: Step-by-step guides with examples
- **Concepts**: Explanations of algorithms, metrics, and theory
- **Guides**: Practical advice on parameters, metrics, troubleshooting
- **API Reference**: Complete function documentation
- **Use Cases**: Real-world applications

## Getting Started

1. **[Installation](installation.md)** - Install dd-coresets
2. **[Quick Start](quickstart.md)** - Your first coreset in 5 minutes
3. **[Tutorials](tutorials/basic_tabular.md)** - Learn with examples

## Key Concepts

### Density–Diversity Trade-off

DDC balances two objectives:
- **Density**: Select points from important regions (modes, clusters)
- **Diversity**: Ensure spatial coverage (don't select all from one cluster)

The `alpha` parameter controls this trade-off: higher `alpha` favors density, lower `alpha` favors diversity.

### Why Weights Matter

Unlike simple sampling, DDC assigns weights to selected points. A point with weight 0.1 "stands for" 10% of the original data in that region. This allows a small coreset to accurately represent the full distribution.

### When DDC Excels

DDC is superior to random sampling when:
- Your data has **well-defined clusters** or modes
- You need **spatial coverage** of all regions
- You have **small k** (limited representatives)
- Your data has **complex marginal distributions**

See [When to Use DDC](../DDC_ADVANTAGE_CASES.md) for detailed analysis.

## Learn More

- **GitHub**: [crbazevedo/dd-coresets](https://github.com/crbazevedo/dd-coresets)
- **PyPI**: [dd-coresets](https://pypi.org/project/dd-coresets/)
- **Issues**: [Report bugs or request features](https://github.com/crbazevedo/dd-coresets/issues)

