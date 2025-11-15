# Welcome to dd-coresets

**Density–Diversity Coresets (DDC)**: a small weighted set of *real* data points that approximates the empirical distribution of a large dataset.

## The Problem We Solve

Imagine you have a dataset with millions of rows—perhaps customer transactions, sensor readings, or scientific measurements. You need to:

- Create exploratory visualizations, but plotting millions of points is slow and cluttered
- Run scenario analyses, but you need representative scenarios, not just random samples
- Prototype machine learning models, but training on the full dataset takes hours
- Perform distributional stress tests, but you need a faithful approximation of the data distribution

**Random sampling** is simple, but it can miss important clusters, under-sample rare modes, and fail to preserve the distribution's shape. **Stratified sampling** works well when you know the right strata, but requires domain knowledge and may not align with your analysis goals.

DDC solves this by automatically selecting a small set of real data points with weights that preserve the distributional properties of your data—no domain knowledge required.

## What is DDC?

DDC selects real data points from your dataset and assigns weights to them, creating a small "summary" that preserves the distributional properties of the original data. The algorithm balances two objectives:

- **Density**: Select points from important regions (modes, clusters) to preserve the distribution
- **Diversity**: Ensure spatial coverage so we don't select all points from one cluster

Unlike random sampling, DDC:

- **Guarantees spatial coverage** of all clusters and modes, even small ones
- **Preserves marginal distributions** better than random sampling (typically 20-30% improvement in Wasserstein-1 distance)
- **Selects real data points** (not synthetic centroids like k-means) for interpretability
- **Works unsupervised** (no domain knowledge required)

## Why Use DDC?

Large datasets are ubiquitous in data science, but many workflows require small, interpretable subsets. Here are common scenarios where DDC helps:

**Scenario 1: Exploratory Data Analysis**
You have 10 million customer records and want to create interactive dashboards. Plotting all points is impossible, and random sampling might miss important customer segments. DDC ensures all segments are represented.

**Scenario 2: Scenario Analysis**
You need to run "what-if" simulations on a financial dataset. Instead of using random samples (which may not represent important edge cases), DDC provides weighted scenarios that preserve the distribution.

**Scenario 3: Model Prototyping**
You're developing a machine learning model on a 50GB dataset. Training on the full data takes 8 hours per iteration. DDC creates a 1GB coreset that preserves the distribution, allowing you to prototype in minutes instead of hours.

**Scenario 4: Distributional Stress Testing**
You need to test how your system behaves under different data distributions. DDC provides a small, weighted coreset that accurately represents the original distribution, enabling efficient stress testing.

## Quick Example

Let's see DDC in action. We'll create a coreset from a large dataset and verify it preserves the distribution:

```python
from dd_coresets import fit_ddc_coreset
import numpy as np

# Your large dataset (100k points, 10 features)
X = np.random.randn(100000, 10)

# Create a coreset of 500 weighted points
S, w, info = fit_ddc_coreset(X, k=500, mode="auto", preset="balanced")

# S: 500 representatives (real data points from X)
# w: weights (sum to 1)
# info: metadata about the coreset

# Verify distribution preservation
weighted_mean = (S * w[:, None]).sum(axis=0)
original_mean = X.mean(axis=0)
mean_error = np.linalg.norm(weighted_mean - original_mean)

print(f"Mean error: {mean_error:.6f}")  # Typically < 0.01
print(f"Coreset size: {len(S)} points (0.5% of original)")
```

In this example, we compressed 100,000 points into 500 weighted representatives while preserving the mean with error typically less than 0.01. The coreset is 200× smaller but maintains distributional fidelity.

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

## Theoretical Foundations

DDC builds on several well-established concepts in computer science and statistics:

**Coresets**: The theoretical foundation of coresets was established by Feldman & Langberg (2011) in their unified framework for coreset construction. A coreset is a small weighted subset that approximates a function on the full dataset. DDC applies this concept to distributional approximation.

**Facility Location and k-Medoids**: The greedy selection strategy in DDC is related to facility location problems and k-medoids clustering (Kaufman & Rousseeuw, 1990). Like k-medoids, DDC selects real data points (medoids) rather than synthetic centroids, ensuring interpretability.

**Diversity Sampling**: The diversity component of DDC is inspired by determinantal point processes (DPPs) and diversity sampling methods (Kulesza & Taskar, 2012). DPPs provide a principled way to sample diverse subsets, and DDC adapts this intuition to balance diversity with density.

**Optimal Transport and Wasserstein Distance**: The evaluation of DDC coresets using Wasserstein distance connects to optimal transport theory (Villani, 2009). Wasserstein distance measures the "cost" of transforming one distribution into another, providing a natural metric for distributional approximation.

**Kernel Methods**: The weight assignment in DDC uses kernel-based soft assignments, a technique common in kernel methods and non-parametric statistics (Schölkopf & Smola, 2002). Kernels allow smooth, continuous approximations of discrete distributions.

## References

- Feldman, D., & Langberg, M. (2011). A unified framework for approximating and clustering data. *Proceedings of the 43rd annual ACM symposium on Theory of computing*.

- Kaufman, L., & Rousseeuw, P. J. (1990). *Finding groups in data: an introduction to cluster analysis*. John Wiley & Sons.

- Kulesza, A., & Taskar, B. (2012). Determinantal point processes for machine learning. *Foundations and Trends in Machine Learning*, 5(2-3), 123-286.

- Villani, C. (2009). *Optimal transport: old and new* (Vol. 338). Springer.

- Schölkopf, B., & Smola, A. J. (2002). *Learning with kernels: support vector machines, regularization, optimization, and beyond*. MIT press.

## Learn More

- **GitHub**: [crbazevedo/dd-coresets](https://github.com/crbazevedo/dd-coresets)
- **PyPI**: [dd-coresets](https://pypi.org/project/dd-coresets/)
- **Issues**: [Report bugs or request features](https://github.com/crbazevedo/dd-coresets/issues)

