# Choosing Parameters

## Overview

DDC has several parameters that control its behavior. This guide explains what each parameter does and how to choose values.

## Core Parameters

### `k` (Number of Representatives)

**What it does**: Controls the size of the coreset.

**How to choose**:
- **Small k (50-200)**: For quick exploration, small datasets, or when you need a very compact summary
- **Medium k (200-1000)**: Good default for most use cases
- **Large k (1000+)**: For high-fidelity approximation, large datasets, or when you need many representatives

**Rule of thumb**: 
- `k = 0.01 × n` to `0.1 × n` (1-10% of data)
- For clustered data: `k ≥ 2 × number_of_clusters` to ensure all clusters are represented

**Conceptual note**: With very small k, DDC's diversity term becomes crucial—it ensures all clusters are covered even with limited representatives. Random sampling may miss small clusters entirely.

### `alpha` (Density-Diversity Trade-off)

**What it does**: Controls the balance between density (important regions) and diversity (spatial coverage).

**Range**: 0.0 to 1.0 (default: 0.3)

**How to choose**:
- **Low alpha (0.1-0.2)**: Favors diversity → better spatial coverage, may sacrifice distribution fidelity
  - Use when: You need all clusters/modes represented, even small ones
  - Use when: Spatial coverage is more important than exact distribution match
- **Medium alpha (0.3-0.4)**: Balanced (default) → good trade-off for most cases
  - Use when: You want both good distribution preservation and spatial coverage
- **High alpha (0.5-0.8)**: Favors density → better distribution preservation, may miss small clusters
  - Use when: Preserving the distribution exactly is more important than covering all regions
  - Use when: You have a dominant mode/cluster

**Conceptual note**: `alpha` controls the trade-off in the selection score: `score(x) = p(x)^α × distance(x)^(1-α)`. Higher `alpha` means density matters more; lower `alpha` means diversity matters more.

### `gamma` (Weight Concentration)

**What it does**: Controls how concentrated weights are (via kernel bandwidth).

**Range**: 0.5 to 3.0 (default: 1.0)

**How to choose**:
- **Low gamma (0.5-0.8)**: More spread out weights → smoother approximation
  - Use when: You want a smoother distribution approximation
- **Medium gamma (1.0-1.5)**: Balanced (default)
- **High gamma (2.0-3.0)**: More concentrated weights → sharper approximation
  - Use when: You want weights to be more localized to representatives

**Conceptual note**: `gamma` controls the kernel bandwidth. Higher `gamma` means the kernel decays faster with distance, so weights are more concentrated near representatives.

## Pipeline Parameters

### `mode` (Distance Mode)

**Options**: `"euclidean"`, `"adaptive"`, `"auto"` (default: `"euclidean"`)

**How to choose**:
- **`"euclidean"`**: Fastest, use for d < 20
- **`"adaptive"`**: Better for elliptical clusters, use for d ≥ 20
- **`"auto"`**: Let DDC decide based on dimensionality (recommended)

**Conceptual note**: Adaptive distances use local Mahalanobis distance to account for data shape. See [Adaptive Distances](../concepts/adaptive_distances.md) for details.

### `preset` (Configuration Preset)

**Options**: `"fast"`, `"balanced"`, `"robust"` (default: `"balanced"`)

**How to choose**:
- **`"fast"`**: Quick runs, fewer neighbors (24), 1 iteration
  - Use for: Prototyping, large datasets, when speed matters
- **`"balanced"`**: Default, good trade-off (32 neighbors, 1 iteration)
  - Use for: Most use cases
- **`"robust"`**: Better quality, more neighbors (64), 2 iterations
  - Use for: When quality is more important than speed

## Advanced Parameters

### `n0` (Working Sample Size)

**What it does**: Size of the working sample used for density estimation and selection.

**Default**: `None` (uses `min(20000, n)`)

**How to choose**:
- **Small n0 (5000-10000)**: Faster, may miss rare modes
- **Medium n0 (20000)**: Good default
- **Large n0 (50000+)**: More accurate, slower
- **`n0=None`**: Uses all data (slowest, most accurate)

**Conceptual note**: DDC uses a working sample for efficiency. Larger `n0` means better density estimates but slower computation.

### `m_neighbors` (Number of Neighbors)

**What it does**: Number of neighbors used for k-NN density estimation.

**Default**: 32 (from preset)

**How to choose**:
- **Small m (16-24)**: Faster, less accurate density estimates
- **Medium m (32)**: Good default
- **Large m (64-128)**: More accurate, slower

**Conceptual note**: More neighbors → more stable density estimates, but slower computation. For adaptive distances, `m_neighbors` must be > d (dimensionality).

### `reweight_full` (Full Dataset Reweighting)

**What it does**: Whether to recompute weights on the full dataset after selection.

**Default**: `True`

**How to choose**:
- **`True`**: More accurate weights, slower
- **`False`**: Faster, weights computed only on working sample

**Conceptual note**: Reweighting on the full dataset ensures weights accurately represent the full distribution, not just the working sample.

## Quick Reference

| Parameter | Default | When to Increase | When to Decrease |
|-----------|---------|-----------------|------------------|
| `k` | - | Need more representatives | Need smaller coreset |
| `alpha` | 0.3 | Want better distribution match | Want better spatial coverage |
| `gamma` | 1.0 | Want more localized weights | Want smoother approximation |
| `n0` | 20000 | Want better accuracy | Want faster computation |
| `m_neighbors` | 32 | Want better density estimates | Want faster computation |

## Example: Tuning for Your Use Case

```python
from dd_coresets import fit_ddc_coreset

# Scenario 1: Small k, need all clusters
S, w, _ = fit_ddc_coreset(
    X, k=100, 
    alpha=0.2,  # Low alpha for better coverage
    mode="auto"
)

# Scenario 2: High-fidelity approximation
S, w, _ = fit_ddc_coreset(
    X, k=1000,
    alpha=0.4,  # Higher alpha for better distribution
    preset="robust",  # Better quality
    reweight_full=True
)

# Scenario 3: Fast prototyping
S, w, _ = fit_ddc_coreset(
    X, k=200,
    preset="fast",  # Fastest
    n0=10000,  # Smaller working sample
    reweight_full=False  # Skip full reweighting
)
```

## Further Reading

- [Algorithm Overview](../concepts/algorithm.md) - How parameters affect the algorithm
- [Understanding Metrics](understanding_metrics.md) - How to evaluate parameter choices
- [Troubleshooting](troubleshooting.md) - Common parameter-related issues

