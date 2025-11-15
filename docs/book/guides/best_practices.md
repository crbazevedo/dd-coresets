# Best Practices

## General Guidelines

### 1. Start with Defaults

DDC's default parameters are tuned for most use cases:
- `mode="euclidean"` (or `"auto"` for high dimensions)
- `preset="balanced"`
- `alpha=0.3`
- `k` = 1-10% of data size

Start here and tune only if needed.

### 2. Understand Your Data

Before using DDC, understand your data:
- **Clustered?** → DDC excels
- **High-dimensional?** → Use `mode="auto"` for PCA
- **Uniform?** → Random may be better
- **Small k needed?** → DDC guarantees coverage

### 3. Choose k Appropriately

**Rule of thumb**: `k = 0.01 × n` to `0.1 × n`

**For clustered data**: `k ≥ 2 × number_of_clusters` to ensure all clusters are represented.

**For small k**: Decrease `alpha` to favor diversity and ensure coverage.

### 4. Use Auto Mode for High Dimensions

For `d ≥ 20`, use `mode="auto"` to let DDC:
- Apply PCA reduction automatically
- Choose appropriate distance metric
- Optimize for your data

### 5. Set Random State for Reproducibility

Always set `random_state` for reproducible results:

```python
S, w, _ = fit_ddc_coreset(X, k=200, random_state=42)
```

## Use Case Specific

### Exploratory Data Analysis

- Use `k=200-500` for quick exploration
- Use `preset="fast"` for speed
- Focus on spatial coverage (low `alpha`)

### Distributional Approximation

- Use `k=500-1000` for high fidelity
- Use `preset="robust"` for quality
- Set `reweight_full=True`
- Focus on distribution metrics (W1, KS)

### Model Prototyping

- Use `k=1000+` for large datasets
- Use `preset="balanced"` for speed/quality trade-off
- Ensure class proportions if supervised (use `fit_ddc_coreset_by_label`)

### Scenario Analysis

- Use `k=200-500` for manageable scenarios
- Preserve important modes (medium `alpha`)
- Use weights in your analysis

## Performance Optimization

### For Large Datasets (n > 1M)

- Use `n0=20000` (default working sample)
- Set `reweight_full=False` if speed is critical
- Use `preset="fast"` for quick runs

### For High Dimensions (d > 50)

- Use `mode="auto"` (triggers PCA)
- Let DDC reduce to 20-50 dimensions automatically
- Representatives are returned in original space

### For Small k (k < 100)

- Decrease `alpha` (e.g., `alpha=0.2`) to ensure coverage
- Ensure `k >= 2 × number_of_clusters`
- Use `preset="robust"` for better quality

## Evaluation

### Always Compute Multiple Metrics

Don't rely on a single metric:
- **Mean error**: Center preservation
- **W1**: Marginal distribution preservation
- **Covariance error**: Global structure
- **Spatial coverage**: Cluster representation

### Compare with Baselines

Always compare with Random (and Stratified if applicable):
- Understand when DDC helps
- Understand when Random is better
- Make informed decisions

### Visualize Results

Visualizations help understand coreset quality:
- 2D/3D scatter plots (with UMAP if needed)
- Marginal distribution histograms
- Coverage heatmaps

## Common Mistakes

### ❌ Don't: Use DDC blindly

**Do**: Understand when DDC helps (clustered data, small k, spatial coverage)

### ❌ Don't: Ignore Random baseline

**Do**: Always compare with Random to understand trade-offs

### ❌ Don't: Use very small k without tuning

**Do**: Decrease `alpha` for small k to ensure coverage

### ❌ Don't: Forget random_state

**Do**: Always set `random_state` for reproducibility

### ❌ Don't: Use DDC for uniform distributions

**Do**: Use Random if data is uniform or has no clear structure

## Further Reading

- [Choosing Parameters](choosing_parameters.md) - Detailed parameter guide
- [Understanding Metrics](understanding_metrics.md) - How to evaluate coresets
- [When to Use DDC](../../DDC_ADVANTAGE_CASES.md) - Comprehensive analysis
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

