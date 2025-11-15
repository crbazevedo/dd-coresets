# Understanding DDC Metrics

This page explains the metrics used to evaluate DDC coresets and what they measure.

## Distributional Metrics

### Mean Error (L2)

**What it measures**: How well the coreset preserves the mean (center) of the distribution.

**Definition**: `||μ_original - μ_coreset||₂`

Where `μ_coreset = Σ w_i × s_i` (weighted mean).

**Intuition**: Lower is better. A small mean error means the coreset is centered similarly to the original data.

**When it matters**: Important for preserving the "location" of the distribution. DDC typically achieves 50-70% better mean preservation than Random.

### Covariance Error (Frobenius)

**What it measures**: How well the coreset preserves the covariance matrix (shape and scale).

**Definition**: `||Σ_original - Σ_coreset||_F`

Where `Σ_coreset` is the weighted covariance matrix.

**Intuition**: Lower is better. Covariance captures correlations and variance. Random sampling sometimes preserves global covariance better than DDC (which focuses on local density).

**When Random might win**: If the original distribution has strong global correlations, Random can preserve the covariance structure better. DDC may sacrifice some global covariance for better local structure preservation.

### Correlation Error (Frobenius)

**What it measures**: How well the coreset preserves pairwise correlations (normalized covariance).

**Definition**: `||R_original - R_coreset||_F`

Where `R` is the correlation matrix (normalized covariance).

**Intuition**: Lower is better. Correlation captures relationships between features. DDC often preserves correlations better than Random, especially in clustered data.

### Wasserstein-1 Distance

**What it measures**: The "cost" of transforming one distribution into another.

**Intuition**: Imagine you have piles of dirt (distribution 1) and need to move them to match another shape (distribution 2). W1 is the minimum "work" (distance × mass) needed.

**Mathematical definition**: For 1D distributions, W₁(P, Q) = ∫ |F_P(x) - F_Q(x)| dx, where F is the cumulative distribution function.

**Why it matters for DDC**: Lower W1 means the coreset distribution is closer to the original. This is especially important for **marginal distributions** (per-feature). DDC typically achieves 20-30% better W1 than Random.

**When Random might win**: If the original distribution is uniform or has strong global structure, Random can sometimes preserve it better.

### Kolmogorov-Smirnov (KS) Statistic

**What it measures**: Maximum difference between cumulative distribution functions.

**Definition**: `KS = max_x |F_original(x) - F_coreset(x)|`

**Intuition**: Lower is better. KS captures the worst-case deviation in the distribution. Useful for detecting systematic biases.

**Why it matters**: A high KS means there's a region where the coreset distribution deviates significantly from the original.

### Maximum Mean Discrepancy (MMD)

**What it measures**: Distance between distributions in a reproducing kernel Hilbert space (RKHS).

**Intuition**: MMD measures how different two distributions are using a kernel function. It captures both mean and higher-order moments.

**Why it matters**: MMD is a comprehensive measure of distributional similarity. Lower MMD means better distributional preservation.

## Spatial Metrics

### Coverage per Cluster

**What it measures**: How many coreset points fall in each cluster/region.

**Intuition**: Higher coverage means better representation of that cluster. DDC guarantees coverage of all clusters, even small ones.

**When it matters**: Important for ensuring all modes/regions are represented. Random may miss small clusters entirely.

### Minimum Distance to Full Data

**What it measures**: For each point in the full dataset, the distance to the nearest coreset point.

**Intuition**: Lower minimum distance means better spatial coverage. DDC typically achieves better coverage than Random.

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

- **Use DDC when**: You have clustered data, need spatial coverage, or have small k
- **Use Random when**: Preserving exact global covariance is critical, or data is high-dimensional and sparse

## Further Reading

- [Understanding Metrics Guide](../guides/understanding_metrics.md) - Practical guide to interpreting metrics
- [Algorithm Overview](algorithm.md) - How DDC achieves these metrics
- [When to Use DDC](../../DDC_ADVANTAGE_CASES.md) - Detailed analysis of DDC advantages

