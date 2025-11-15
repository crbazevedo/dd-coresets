# The DDC Algorithm: Why It Works

## Overview

The DDC algorithm has three main steps:
1. **Density Estimation**: Estimate local density for each point
2. **Greedy Selection**: Select representatives balancing density and diversity
3. **Weight Assignment**: Assign weights to selected points

## Step 1: Density Estimation

**What**: Estimate `p(x)` for each point using k-NN (see [Density Estimation](density_estimation.md)).

**Why**: We want to prioritize points from dense regions (modes) to preserve the distribution. But we also need diversity (not all from one mode).

**Trade-off**: High density alone would select all points from the largest cluster. We need diversity to cover all modes.

## Step 2: Greedy Selection

**What**: Iteratively select points that maximize a score combining density and diversity.

**The Score**: For each candidate point `x`, we compute:

```
score(x) = p(x)^α × min_distance(x, selected_reps)^(1-α)
```

Where:
- `p(x)` is the density estimate
- `min_distance(x, selected_reps)` is the minimum distance to already-selected representatives
- `α` (alpha) controls the trade-off (default 0.3)

**Why greedy works**: The diversity term (distance) ensures we don't select points too close to already-selected points. This is similar to facility location problems (submodular optimization).

**Intuition**: At each step, we pick the point that:
1. Has high density (important region)
2. Is far from already-selected points (diverse)

This naturally balances coverage of modes with spatial diversity.

### Why Not Optimal?

Greedy selection is not globally optimal, but:
- **Fast**: O(k × n) instead of exponential
- **Effective**: Works well in practice
- **Guaranteed coverage**: Ensures all clusters are represented (with appropriate `α`)

## Step 3: Weight Assignment

**What**: Assign weights to selected representatives using soft assignments.

**Why soft assignments**: Hard assignments (each point belongs to one representative) can be brittle. Soft assignments (points contribute to multiple representatives) are more robust and better approximate the distribution.

**Kernel-based weighting**: We use a kernel (e.g., RBF) to determine how much each original point "belongs" to each representative:

```
w_j = Σ_i kernel(x_i, s_j) / Σ_j Σ_i kernel(x_i, s_j)
```

Where:
- `kernel(x_i, s_j)` is large if `x_i` is close to representative `s_j`
- Weights are normalized so they sum to 1

**Intuition**: Points close to a representative get high weight for that representative. This creates a smooth, weighted approximation of the original distribution.

## Why This Works

The combination of these three steps ensures:

1. **Distributional fidelity**: Density-based selection preserves important regions
2. **Spatial coverage**: Diversity term ensures all regions are represented
3. **Accurate approximation**: Soft weights allow a small coreset to represent the full distribution

## Complexity

- **Density estimation**: O(n log n) for k-NN graph construction
- **Greedy selection**: O(k × n × d) for k representatives
- **Weight assignment**: O(n × k × d) for soft assignments
- **Total**: O(n log n + k × n × d), which is sublinear in n when using a working sample

## Medoid Refinement (Optional)

After selection, DDC optionally refines representatives by:
1. For each representative, find the medoid (most central point) in its local neighborhood
2. Replace the representative with the medoid

**Why**: This improves the coreset by ensuring representatives are truly representative of their regions.

## Further Reading

- [Density Estimation](density_estimation.md) - How density is estimated
- [Weighting](weighting.md) - Details on weight assignment
- [Choosing Parameters](../guides/choosing_parameters.md) - How to tune `α` and other parameters

