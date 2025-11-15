# API Reference

Complete reference for all public functions in `dd-coresets`.

## Core Functions

### `fit_ddc_coreset`

```python
fit_ddc_coreset(
    X: np.ndarray,
    k: int,
    *,
    n0: int | None = None,
    alpha: float = 0.3,
    gamma: float = 1.0,
    refine_iters: int = 1,
    reweight_full: bool = True,
    random_state: int | None = 42,
    mode: str = "euclidean",
    preset: str = "balanced",
    distance_cfg: dict | None = None,
    pipeline_cfg: dict | None = None,
    **legacy_kwargs
) -> tuple[np.ndarray, np.ndarray, dict]
```

**Description**: Computes a Density-Diversity Coreset (DDC) for the input dataset.

**Parameters**:

- **X** (array, shape `(n, d)`): Input dataset. Each row is a data point.
- **k** (int): Number of representatives to select. Must be positive and less than or equal to `n`.
- **n0** (int | None): Size of working sample. If `None`, uses `min(20000, n)`. Larger values improve accuracy but increase computation time.
- **alpha** (float): Density-diversity trade-off parameter. Range [0, 1]. Higher values favor density (distribution preservation), lower values favor diversity (spatial coverage). Default: 0.3.
- **gamma** (float): Weight concentration parameter. Controls kernel bandwidth for weight assignment. Default: 1.0.
- **refine_iters** (int): Number of medoid refinement iterations. Default: 1.
- **reweight_full** (bool): If `True`, recomputes weights on full dataset after selection. More accurate but slower. Default: `True`.
- **random_state** (int | None): Random seed for reproducibility. Default: 42.
- **mode** (str): Distance mode. Options: `"euclidean"`, `"adaptive"`, `"auto"`. Default: `"euclidean"`.
  - `"euclidean"`: Use Euclidean distance (fastest, for d < 20)
  - `"adaptive"`: Use local Mahalanobis distance (for elliptical clusters, d ≥ 20)
  - `"auto"`: Automatically choose based on dimensionality
- **preset** (str): Configuration preset. Options: `"fast"`, `"balanced"`, `"robust"`, `"manual"`. Default: `"balanced"`.
  - `"fast"`: Quick runs (fewer neighbors, 1 iteration)
  - `"balanced"`: Good trade-off (default)
  - `"robust"`: Better quality (more neighbors, 2 iterations)
  - `"manual"`: Use `distance_cfg` and `pipeline_cfg` for full control
- **distance_cfg** (dict | None): Manual distance configuration. Used when `preset="manual"`. See [Choosing Parameters](../guides/choosing_parameters.md) for details.
- **pipeline_cfg** (dict | None): Manual pipeline configuration. Used when `preset="manual"`. See [Choosing Parameters](../guides/choosing_parameters.md) for details.
- **legacy_kwargs**: Deprecated parameters. Emit warnings and map to new API.

**Returns**:

- **S** (array, shape `(k, d)`): Selected representatives (real data points from `X`).
- **w** (array, shape `(k,)`): Weights that sum to 1. Each weight `w[i]` indicates how much representative `S[i]` "stands for" in the original distribution.
- **info** (dict): Metadata dictionary containing:
  - `method`: "ddc"
  - `k`: Number of representatives
  - `n`: Original dataset size
  - `pipeline`: Pipeline information (mode, adaptive, PCA, etc.)
  - `selected_indices`: Indices of selected points in original dataset
  - `config`: Resolved configuration
  - Other metadata

**Example**:

```python
from dd_coresets import fit_ddc_coreset
import numpy as np

X = np.random.randn(10000, 10)
S, w, info = fit_ddc_coreset(X, k=200, mode="auto", preset="balanced")

print(f"Selected {len(S)} representatives")
print(f"Mean error: {compute_mean_error(X, S, w):.6f}")
```

**Conceptual note**: DDC balances density (selecting from important regions) and diversity (ensuring spatial coverage). The `alpha` parameter controls this trade-off. See [Algorithm Overview](../concepts/algorithm.md) for details.

---

### `fit_ddc_coreset_by_label`

```python
fit_ddc_coreset_by_label(
    X: np.ndarray,
    y: np.ndarray,
    k_total: int,
    **ddc_kwargs
) -> tuple[np.ndarray, np.ndarray, dict]
```

**Description**: Computes a DDC coreset that preserves class proportions by applying DDC separately within each class.

**Parameters**:

- **X** (array, shape `(n, d)`): Input features.
- **y** (array, shape `(n,)`): Class labels (integers).
- **k_total** (int): Total number of representatives across all classes.
- **ddc_kwargs**: Keyword arguments passed to `fit_ddc_coreset` for each class.

**Returns**:

- **S** (array, shape `(k_total, d)`): Selected representatives from all classes.
- **w** (array, shape `(k_total,)`): Weights that sum to 1.
- **info** (dict): Metadata including:
  - `method`: "ddc_by_label"
  - `k_total`: Total representatives
  - `classes`: List of unique class labels
  - `k_per_class`: Number of representatives per class
  - `n_per_class`: Original size per class
  - `info_per_class`: DDC info for each class

**Example**:

```python
from dd_coresets import fit_ddc_coreset_by_label
import numpy as np

X = np.random.randn(10000, 10)
y = np.random.randint(0, 3, 10000)  # 3 classes

S, w, info = fit_ddc_coreset_by_label(
    X, y, k_total=500, mode="auto", preset="balanced"
)

print(f"Class proportions preserved: {info['k_per_class']}")
```

**Conceptual note**: Label-aware DDC preserves class proportions by design, making it ideal for supervised learning tasks. It applies DDC separately within each class, preserving both distributional properties and label balance. See [Weighting](../concepts/weighting.md) for details.

---

### `fit_random_coreset`

```python
fit_random_coreset(
    X: np.ndarray,
    k: int,
    *,
    n0: int = 20000,
    gamma: float = 1.0,
    reweight_full: bool = True,
    random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray, dict]
```

**Description**: Creates a coreset by uniform random sampling. Useful as a baseline for comparison.

**Parameters**:

- **X** (array, shape `(n, d)`): Input dataset.
- **k** (int): Number of representatives.
- **n0** (int): Working sample size (for consistency with DDC API). Default: 20000.
- **gamma** (float): Weight concentration (for consistency). Default: 1.0.
- **reweight_full** (bool): If `True`, computes weights on full dataset. Default: `True`.
- **random_state** (int | None): Random seed.

**Returns**:

- **S** (array, shape `(k, d)`): Randomly selected points.
- **w** (array, shape `(k,)`): Uniform weights (all equal to 1/k).
- **info** (dict): Metadata.

**Example**:

```python
from dd_coresets import fit_random_coreset

S_random, w_random, info_random = fit_random_coreset(X, k=200, random_state=42)
```

**When to use**: As a baseline for comparison. Random sampling may preserve global covariance better than DDC in some cases (high-dimensional sparse data, uniform distributions).

---

### `fit_stratified_coreset`

```python
fit_stratified_coreset(
    X: np.ndarray,
    strata: np.ndarray,
    k: int,
    *,
    n0: int = 20000,
    gamma: float = 1.0,
    reweight_full: bool = True,
    random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray, dict]
```

**Description**: Creates a coreset by stratified sampling, preserving stratum proportions.

**Parameters**:

- **X** (array, shape `(n, d)`): Input dataset.
- **strata** (array, shape `(n,)`): Stratum labels for each point.
- **k** (int): Number of representatives.
- **n0** (int): Working sample size. Default: 20000.
- **gamma** (float): Weight concentration. Default: 1.0.
- **reweight_full** (bool): If `True`, computes weights on full dataset. Default: `True`.
- **random_state** (int | None): Random seed.

**Returns**:

- **S** (array, shape `(k, d)`): Stratified sample.
- **w** (array, shape `(k,)`): Weights preserving stratum proportions.
- **info** (dict): Metadata including stratum information.

**Example**:

```python
from dd_coresets import fit_stratified_coreset

# Stratify by cluster labels
S_strat, w_strat, info_strat = fit_stratified_coreset(
    X, strata=cluster_labels, k=200, random_state=42
)
```

**When to use**: When you know the right strata (e.g., classes, segments) and want to preserve their proportions. Stratified sampling is a strong baseline when strata are known.

---

### `fit_kmedoids_coreset`

```python
fit_kmedoids_coreset(
    X: np.ndarray,
    k: int,
    *,
    n0: int = 20000,
    random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray, dict]
```

**Description**: Creates a coreset using k-medoids clustering. Selects real data points (medoids) that minimize within-cluster distances.

**Parameters**:

- **X** (array, shape `(n, d)`): Input dataset.
- **k** (int): Number of medoids.
- **n0** (int): Working sample size. Default: 20000.
- **random_state** (int | None): Random seed.

**Returns**:

- **S** (array, shape `(k, d)`): Selected medoids.
- **w** (array, shape `(k,)`): Weights based on cluster sizes.
- **info** (dict): Metadata including cluster assignments.

**Example**:

```python
from dd_coresets import fit_kmedoids_coreset

S_kmed, w_kmed, info_kmed = fit_kmedoids_coreset(X, k=200, random_state=42)
```

**When to use**: When you want to minimize reconstruction error (like k-means) but need real data points. K-medoids is more robust to outliers than k-means but doesn't explicitly preserve distributional properties like DDC.

**Conceptual note**: K-medoids is related to facility location problems (Kaufman & Rousseeuw, 1990). Like DDC, it selects real data points, but it optimizes for reconstruction error rather than distributional fidelity.

---

## Data Types

### `CoresetInfo`

A dataclass (or dict) containing metadata about a coreset. Available fields:

- `method`: Method name ("ddc", "random", "stratified", "kmedoids")
- `k`: Number of representatives
- `n`: Original dataset size
- `selected_indices`: Indices of selected points
- `pipeline`: Pipeline information (for DDC)
- Other method-specific fields

**Note**: In v0.2.0+, `fit_ddc_coreset` returns a standard Python `dict` instead of `CoresetInfo` dataclass for backward compatibility, but with the same fields.

---

## Configuration

### Presets

Presets provide pre-configured parameter sets:

- **`"fast"`**: Quick runs
  - `m_neighbors=24`, `iterations=1`
  - Use for prototyping or large datasets
- **`"balanced"`**: Default, good trade-off
  - `m_neighbors=32`, `iterations=1`
  - Use for most cases
- **`"robust"`**: Better quality
  - `m_neighbors=64`, `iterations=2`
  - Use when quality is more important than speed

### Manual Configuration

When `preset="manual"`, use `distance_cfg` and `pipeline_cfg`:

```python
S, w, info = fit_ddc_coreset(
    X, k=200,
    preset="manual",
    distance_cfg={
        "m_neighbors": 40,
        "iterations": 2,
        "shrinkage": "oas",
        "reg_eps": 1e-6
    },
    pipeline_cfg={
        "dim_threshold_adaptive": 25,
        "reduce": "auto",
        "retain_variance": 0.95,
        "cap_components": 50
    }
)
```

See [Choosing Parameters](../guides/choosing_parameters.md) for detailed guidance.

---

## Further Reading

- [Quick Start](../quickstart.md) - Get started in 5 minutes
- [Choosing Parameters](../guides/choosing_parameters.md) - Parameter tuning guide
- [Algorithm Overview](../concepts/algorithm.md) - How DDC works
- [Understanding Metrics](../guides/understanding_metrics.md) - How to evaluate coresets

