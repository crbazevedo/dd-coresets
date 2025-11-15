# Exploratory Data Analysis

## Use Case: Interactive Dashboards from Large Datasets

### The Problem

You have a dataset with millions of rows—customer transactions, sensor readings, or scientific measurements. You want to create interactive dashboards for exploratory analysis, but:

- Plotting millions of points is slow and cluttered
- Random sampling might miss important segments or modes
- You need a representative sample that preserves the distribution

### The Solution: DDC Coreset

DDC creates a small, weighted coreset that preserves the distributional properties of your data, making it ideal for EDA dashboards.

## Example: Customer Transaction Data

```python
from dd_coresets import fit_ddc_coreset
import numpy as np
import pandas as pd

# Load your large dataset
# df = pd.read_csv('transactions.csv')  # 10M rows
# X = df[['amount', 'age', 'location_x', 'location_y', ...]].values

# For demonstration, generate synthetic data
X = np.random.randn(100000, 8)  # 100k transactions, 8 features

# Create a coreset for dashboard
S, w, info = fit_ddc_coreset(
    X, 
    k=500,  # 500 points for interactive visualization
    mode="auto",
    preset="balanced"
)

print(f"Compressed {len(X):,} points to {len(S):,} representatives")
print(f"Compression ratio: {len(X) / len(S):.1f}x")
```

## Why DDC Works for EDA

**Spatial Coverage**: DDC guarantees all regions are represented. If your data has customer segments (clusters), DDC ensures all segments appear in the dashboard, even small ones.

**Distribution Preservation**: The weighted coreset preserves marginal distributions, so histograms and density plots look similar to the full data.

**Real Data Points**: Unlike k-means centroids, DDC selects real data points, making the dashboard interpretable—you can see actual customer transactions, not synthetic averages.

## Creating Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Use weighted coreset for visualization
# Histogram with weights
plt.figure(figsize=(10, 6))
plt.hist(S[:, 0], weights=w, bins=50, alpha=0.7, label='DDC Coreset')
plt.hist(X[:, 0], bins=50, alpha=0.3, label='Full Data')
plt.xlabel('Feature 0')
plt.ylabel('Density')
plt.legend()
plt.title('Marginal Distribution: DDC vs Full Data')
plt.show()

# Scatter plot (use weights for point sizes)
plt.figure(figsize=(10, 6))
plt.scatter(S[:, 0], S[:, 1], s=w*1000, alpha=0.6, c='red', label='DDC Coreset')
plt.scatter(X[::100, 0], X[::100, 1], s=1, alpha=0.1, c='gray', label='Full Data (sample)')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend()
plt.title('Spatial Coverage: DDC Coreset')
plt.show()
```

## Performance Comparison

For a dataset with 10M rows:

- **Full data**: Plotting takes 30+ seconds, cluttered visualization
- **Random sample (10k)**: Fast but may miss segments, distribution may be distorted
- **DDC coreset (500)**: Fast plotting, preserves distribution, guarantees coverage

## Best Practices

1. **Choose k based on visualization needs**: 200-1000 points work well for most dashboards
2. **Use `mode="auto"`**: Let DDC choose the best pipeline for your data
3. **Verify preservation**: Check that weighted statistics match original (mean, variance)
4. **Compare with Random**: Always compare to understand when DDC helps

## Further Reading

- [Basic Tabular Tutorial](../tutorials/basic_tabular.md) - Complete EDA example
- [Understanding Metrics](../guides/understanding_metrics.md) - How to verify coreset quality
- [Choosing Parameters](../guides/choosing_parameters.md) - Tuning for your use case

