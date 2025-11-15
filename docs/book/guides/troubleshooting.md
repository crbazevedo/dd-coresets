# Troubleshooting

## Common Issues and Solutions

### Issue: DDC is slower than expected

**Possible causes**:
- Large `n0` (working sample size)
- `reweight_full=True` on large datasets
- High dimensionality without PCA

**Solutions**:
- Reduce `n0` (e.g., `n0=10000` instead of default)
- Set `reweight_full=False` for faster computation
- Use `mode="auto"` to trigger PCA for high-dimensional data
- Use `preset="fast"` for quicker runs

### Issue: DDC performs worse than Random

**Possible causes**:
- High-dimensional sparse data
- Uniform distributions
- Strong global correlations
- Small `k` relative to number of clusters

**Solutions**:
- Check if your data has clear cluster structure (DDC excels here)
- For high-dimensional data, ensure `mode="auto"` triggers PCA
- Increase `k` if it's too small relative to number of clusters
- Consider using Random if preserving exact global covariance is critical

**When Random is better**: See [When to Use DDC](../../DDC_ADVANTAGE_CASES.md) for detailed analysis.

### Issue: Small clusters are not represented

**Possible causes**:
- `k` is too small
- `alpha` is too high (favors density over diversity)
- Small clusters are very sparse

**Solutions**:
- Increase `k` (at least `2 × number_of_clusters`)
- Decrease `alpha` (e.g., `alpha=0.2`) to favor diversity
- Ensure `k` is proportional to cluster sizes

### Issue: Weights don't sum to 1

**This should not happen**. If it does:
- Check for numerical errors
- Report as a bug

### Issue: Memory errors on large datasets

**Possible causes**:
- `n0` is too large
- `reweight_full=True` on very large datasets
- High dimensionality

**Solutions**:
- Reduce `n0` (e.g., `n0=20000` max)
- Set `reweight_full=False`
- Use `mode="auto"` to reduce dimensionality with PCA

### Issue: Adaptive distances not working

**Possible causes**:
- `m_neighbors <= d` (dimensionality)
- Not enough samples for covariance estimation

**Solutions**:
- Ensure `m_neighbors > d` (dimensionality)
- Use `mode="auto"` to let DDC decide
- For high dimensions, PCA will be applied automatically

**Conceptual note**: Adaptive distances require `m_neighbors > d` because we need enough neighbors to estimate a d×d covariance matrix reliably.

### Issue: Results are not reproducible

**Possible causes**:
- `random_state` not set
- Different `n0` values
- Different working sample selection

**Solutions**:
- Always set `random_state` for reproducibility:
  ```python
  S, w, _ = fit_ddc_coreset(X, k=200, random_state=42)
  ```
- Use the same `n0` value across runs
- Note: Some randomness is inherent in the algorithm (working sample selection, initialization)

### Issue: Coreset doesn't preserve distribution well

**Possible causes**:
- `k` is too small
- `alpha` is too low (too much diversity, not enough density)
- Data doesn't have clear structure (DDC may not help)

**Solutions**:
- Increase `k`
- Increase `alpha` (e.g., `alpha=0.4-0.5`)
- Check if DDC is appropriate for your data (see [When to Use DDC](../../DDC_ADVANTAGE_CASES.md))

## Performance Tips

### For Speed
- Use `preset="fast"`
- Set `n0` to a smaller value
- Set `reweight_full=False`
- Use `mode="euclidean"` for low dimensions

### For Quality
- Use `preset="robust"`
- Increase `n0` (or use `n0=None`)
- Set `reweight_full=True`
- Use `mode="auto"` for optimal pipeline

### For Small k
- Decrease `alpha` (e.g., `alpha=0.2`) to ensure coverage
- Ensure `k >= 2 × number_of_clusters`

## Getting Help

If you encounter issues not covered here:

1. Check [When to Use DDC](../../DDC_ADVANTAGE_CASES.md) to see if DDC is appropriate
2. Review [Choosing Parameters](choosing_parameters.md) for parameter guidance
3. Open an issue on [GitHub](https://github.com/crbazevedo/dd-coresets/issues)
4. Check the [API Reference](../api/reference.md) for function details

