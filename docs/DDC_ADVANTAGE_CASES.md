# DDC Advantage Cases: When to Use DDC Over Random Sampling

This document summarizes systematic experiments demonstrating when Density-Diversity Coresets (DDC) provide clear advantages over Random sampling.

## Executive Summary

Based on comprehensive experiments across 7 categories, **DDC is superior to Random** when:

1. **Well-defined cluster structures** (Gaussian mixtures with clear separation)
2. **Complex marginal distributions** (skewed, heavy-tailed, multimodal)
3. **Non-convex geometries** (manifolds, rings, moons)
4. **Small k values** (k << n, especially k proportional to number of clusters)
5. **Real datasets with clear structure** (MNIST, Iris, Wine)
6. **Specific use cases** (outlier preservation, low-density region coverage)
7. **Advanced cluster structures** (nested, rare, multi-scale, varying separability)

**Random is superior** when:
- Preserving exact global covariance is critical
- High-dimensional sparse data
- Very large datasets (n >> k) with complex non-Gaussian structure
- Strong correlations between features

## Category 1: Cluster Structures

### 1.1 Varied Number of Clusters

**Experiment**: Gaussian Mixtures with 2, 4, 8, 16 clusters

**Results**:
- DDC shows consistent advantage across all cluster counts
- Improvement increases with number of clusters (better spatial coverage)
- Covariance error: DDC 23-73% better than Random
- Wasserstein-1: DDC 29-30% better than Random

**Key Finding**: DDC maintains advantage especially with well-separated clusters.

### 1.2 Imbalanced Clusters

**Experiment**: Clusters with 1:10 size ratio

**Results**:
- DDC ensures coverage of small clusters
- Random may miss small clusters entirely
- DDC preserves structure better in imbalanced scenarios

**Key Finding**: DDC guarantees spatial coverage even for minority clusters.

### 1.3 Different Shapes

**Experiment**: Mixture of spherical and elliptical clusters

**Results**:
- DDC adapts to different cluster shapes
- Better preservation of elliptical structures

**Key Finding**: DDC handles shape diversity better than Random.

### 1.4 Different Densities

**Experiment**: Clusters with 1:5:10 density ratio

**Results**:
- DDC covers low-density clusters better
- Random may under-sample sparse regions
- Better Wasserstein-1 metrics for low-density clusters

**Key Finding**: DDC ensures coverage of sparse regions that Random might miss.

## Category 2: Complex Marginal Distributions

### 2.1 Skewed/Heavy-tailed Distributions

**Experiment**: Features with log-normal, gamma, pareto distributions

**Results**:
- DDC preserves tail distributions better
- Quantile errors (Q0.05, Q0.95): DDC significantly better
- Wasserstein-1: DDC 29% better on average
- KS statistic: DDC 11% better on average

**Key Finding**: DDC captures outliers and tail behavior better than Random.

### 2.2 Multimodal Per-Feature

**Experiment**: Each feature has 3-modal distribution

**Results**:
- DDC preserves multiple modes per feature
- Better Wasserstein-1 and KS metrics
- Random may miss some modes

**Key Finding**: DDC captures multimodal structure that Random may flatten.

## Category 3: Non-Convex Geometries

### 3.1 Swiss Roll

**Experiment**: 3D Swiss Roll manifold (n=10k)

**Results**:
- DDC covers manifold better
- Better preservation of local geometry
- Covariance error: DDC competitive or better

**Key Finding**: DDC preserves manifold structure better than Random.

### 3.2 S-Curve

**Experiment**: 3D S-Curve manifold (n=10k)

**Results**:
- Similar to Swiss Roll
- DDC maintains better spatial coverage

**Key Finding**: DDC excels on non-linear manifolds.

### 3.3 Concentric Rings

**Experiment**: 2-3 concentric rings (n=10k)

**Results**:
- DDC covers all rings equally
- Random may miss outer rings
- Better coverage per ring

**Key Finding**: DDC ensures coverage of all geometric structures.

## Category 4: Small k Cases

### 4.1 Very Small k (50, 100, 200)

**Experiment**: Gaussian Mixture with k << n

**Results**:
- DDC prevents empty clusters
- Random may leave clusters unrepresented
- Better covariance preservation with small k

**Key Finding**: DDC is more robust with limited coreset size.

### 4.2 k Proportional to Clusters

**Experiment**: k = 2-4x number of clusters

**Results**:
- DDC guarantees at least 1 point per cluster
- Random may miss some clusters
- Better coverage ratio

**Key Finding**: DDC ensures minimum representation per cluster.

### 4.3 Two Moons with Small k

**Experiment**: Two Moons dataset with k=50, 100, 200

**Results**:
- DDC maintains better coverage
- Better Wasserstein-1 metrics
- Preserves non-convex structure even with small k

**Key Finding**: DDC works well even with very small coresets.

## Category 5: Real Datasets

### 5.1 MNIST

**Experiment**: MNIST digits (n=10k, PCA to 50D)

**Results**:
- DDC covers all digits better
- Better preservation of digit-specific structure
- Competitive covariance metrics

**Key Finding**: DDC preserves class structure in real image data.

### 5.2 Iris/Wine

**Experiment**: UCI datasets (Iris: n=150, Wine: n=178)

**Results**:
- DDC ensures coverage of all classes
- Works well even with small n
- Better class balance preservation

**Key Finding**: DDC works effectively even on small datasets.

### 5.3 Fashion-MNIST

**Experiment**: Fashion-MNIST (n=10k, PCA to 50D)

**Results**:
- Similar to MNIST
- Better category coverage

**Key Finding**: DDC preserves categorical structure in real data.

## Category 6: Specific Use Cases

### 6.1 Outlier Preservation

**Experiment**: Gaussian with 5% outliers

**Results**:
- DDC includes more outliers in coreset
- Better tail quantile preservation (Q0.95)
- Random may miss outliers entirely

**Key Finding**: DDC captures important outliers that Random may exclude.

### 6.2 Low-Density Region Coverage

**Experiment**: Mixture with 1:1:10 cluster sizes

**Results**:
- DDC covers small clusters better
- Random may under-sample small clusters
- Better Wasserstein-1 for minority clusters

**Key Finding**: DDC ensures coverage of sparse but important regions.

## Category 7: Advanced Cluster Structures

### 7.1 Nested Clusters

**Experiment**: Hierarchical structure (large clusters containing sub-clusters)

**Results**:
- DDC shows clear advantage in hierarchical structures
- Covariance error: DDC 13.5% better than Random
- Correlation error: DDC 89.8% better than Random
- Wasserstein-1: DDC 37.1% better than Random
- Both methods cover all 6 sub-clusters

**Key Finding**: DDC excels in hierarchical structures, especially in correlation preservation (+89.8%).

### 7.2 Rare Clusters

**Experiment**: 3 common clusters + 1 rare cluster (1.0% of data)

**Results**:
- DDC significantly superior in rare cluster scenarios
- Covariance error: DDC 124.7% better than Random
- Correlation error: DDC 108.0% better than Random
- Wasserstein-1: DDC 32.9% better than Random
- Both methods cover all 4 clusters

**Key Finding**: DDC is much superior for rare but important clusters (+124.7% cov, +108% corr). Critical use case (fraud, anomalies).

### 7.3 Multi-Scale Clusters

**Experiment**: 3 clusters with 1:10:100 size ratios

**Results**:
- Mixed results: DDC better in correlation (+70%), Random better in covariance (-13%)
- Both methods guarantee complete coverage
- DDC ensures representation of smallest cluster (0.9% of data)

**Key Finding**: DDC guarantees coverage even for very small clusters, with trade-off between covariance and correlation preservation.

### 7.4 CIFAR-10

**Experiment**: 10-class dataset (simulated with 10 well-separated clusters, 50D after PCA)

**Results**:
- Random performed better with synthetic data
- May be due to high dimensionality (50D) or specific structure
- Both methods cover all 10 classes

**Key Finding**: DDC may struggle in high-dimensional sparse spaces. Real CIFAR-10 testing needed.

### 7.5 Varying Separability

**Experiment**: 4 clusters with different separation levels (0.5x to 5.0x)

**Results**:
- DDC maintains advantage in covariance across all separation levels (+29% to +124%)
- Mixed results in Wasserstein-1 depending on separation
- Both methods guarantee complete coverage

**Key Finding**: DDC maintains covariance advantage regardless of cluster separation level.

## Summary Table: When DDC Wins

| Category | Scenario | DDC Advantage | Key Metric |
|----------|----------|---------------|------------|
| Clusters | Well-separated mixtures | 23-73% better cov error | Covariance Error |
| Clusters | Imbalanced sizes | Guaranteed coverage | Spatial Coverage |
| Marginals | Skewed/heavy-tailed | 29% better W1 | Wasserstein-1 |
| Marginals | Multimodal | Better mode preservation | KS Statistic |
| Geometry | Non-convex (moons, rings) | Better coverage | Spatial Coverage |
| Geometry | Manifolds (Swiss Roll) | Better local structure | Covariance Error |
| Small k | k << n | No empty clusters | Cluster Coverage |
| Small k | k proportional to clusters | Guaranteed per-cluster | Coverage Ratio |
| Real Data | MNIST/Iris/Wine | Better class coverage | Class Coverage |
| Use Cases | Outliers | Better tail preservation | Quantile Error |
| Use Cases | Low-density regions | Guaranteed coverage | Spatial Coverage |
| Advanced | Nested clusters | +89.8% corr error | Correlation Error |
| Advanced | Rare clusters | +124.7% cov error | Covariance Error |
| Advanced | Multi-scale | Guaranteed coverage | Spatial Coverage |
| Advanced | Varying separability | +29-124% cov error | Covariance Error |

## Decision Guide: DDC vs Random

### Use DDC When:

1. **You have well-defined clusters**
   - Gaussian mixtures with clear separation
   - Multiple distinct groups in your data

2. **You need to preserve complex distributions**
   - Skewed or heavy-tailed features
   - Multimodal distributions
   - Important outliers

3. **You have non-convex structures**
   - Manifolds, rings, moons
   - Non-linear geometries

4. **k is small relative to n**
   - k << n (e.g., k=100, n=20k)
   - k proportional to number of clusters

5. **You need guaranteed spatial coverage**
   - All clusters/regions must be represented
   - Low-density regions are important

6. **You have real data with clear structure**
   - Image datasets (MNIST, Fashion-MNIST)
   - Classification datasets with clear classes

7. **You have advanced cluster structures**
   - Hierarchical/nested clusters
   - Rare but important clusters (fraud, anomalies)
   - Multi-scale clusters (very different sizes)
   - Varying cluster separability

### Use Random When:

1. **Exact covariance preservation is critical**
   - Statistical inference requires unbiased estimates
   - Strong correlations between features

2. **High-dimensional sparse data**
   - Many features, few informative
   - Sparse feature space

3. **Very large datasets (n >> k)**
   - n >> k (e.g., n=100k, k=1k)
   - Random sampling is statistically sufficient

4. **Complex non-Gaussian structure**
   - Real-world data without clear clusters
   - Complex correlations

## Implementation Notes

All experiments use:
- **DDC optimized parameters**: alpha=0.1, gamma=2.0, m_neighbors=16, refine_iters=2
- **Standardized metrics**: Mean Error (L2), Cov Error (Fro), Corr Error (Fro), MMD, W1 (mean/max), KS (mean/max)
- **Consistent evaluation**: All datasets scaled with StandardScaler

## Files and Results

- **Experiment scripts**: `experiments/ddc_advantage/`
- **Results**: `experiments/ddc_advantage/results/`
- **Visualizations**: `docs/images/ddc_advantage/`
- **Unified runner**: `experiments/ddc_advantage/run_all_experiments.py`

## References

- See individual experiment files for detailed results
- Check `experiments/ddc_advantage/results/` for CSV files with all metrics
- Visualizations available in `docs/images/ddc_advantage/`
- For advanced experiments (Category 7), see `docs/ALL_NEW_EXPERIMENTS_CONSOLIDATED.md` for comprehensive analysis

