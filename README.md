# dd-coresets

**Density–Diversity Coresets (DDC)**: a small weighted set of *real* data points that approximates the empirical distribution of a large dataset.

This library exposes a simple API (in the spirit of scikit-learn) to:
- build an **unsupervised** density–diversity coreset (`fit_ddc_coreset`);
- compare against **random** and **stratified** baselines (`fit_random_coreset`, `fit_stratified_coreset`).

The goal is pragmatic: help data scientists work with large datasets using small, distribution-preserving subsets that are easy to simulate, visualise, and explain.

---

## Motivation

In practice, we rarely work on the **full dataset** for everything:

- Exploratory plots and dashboards need **small, interpretable samples**.
- Scenario analysis and simulations need **few representative points** with **weights**.
- Prototyping models and ideas is faster on **coresets** than on full data.

Common approaches:

- **Random sampling**: simple, but can miss important modes or tails.
- **Stratified sampling**: good when you already know the right strata (segments, classes, products), but needs domain knowledge and alignment with stakeholders.
- **Cluster centroids (e.g. k-means)**: minimise reconstruction error, but centroids are not real data points and are not explicitly distributional.

**DDC** sits in between:

- Unsupervised, geometry-aware.
- Selects **real points** (medoids) that cover dense regions and diverse modes.
- Learns **weights** via soft assignments, approximating the empirical distribution.

---

## Installation

```bash
git clone https://github.com/crbazevedo/dd-coresets.git
cd dd-coresets

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Dependencies (minimal):

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib` (for examples/plots)

---

## Quickstart

### 1. Fit a DDC coreset (unsupervised default)

```python
import numpy as np
from dd_coresets.ddc import fit_ddc_coreset

# X: (n, d) preprocessed features (e.g. scaled, encoded, etc.)
X = ...  # load your data here

S, w, info = fit_ddc_coreset(
    X,
    k=200,           # number of representatives
    n0=20000,        # working sample size (None = use all)
    m_neighbors=32,  # kNN for density
    alpha=0.3,       # density–diversity trade-off
    gamma=1.0,       # kernel scale
    refine_iters=1,  # medoid refinement iters
    reweight_full=True,
    random_state=42,
)

# S: (k, d) real data points
# w: (k,) non-negative, sum to 1
# info: metadata (indices, parameters, etc.)
print(S.shape, w.shape)
print(info.method, info.k, info.n, info.n0)
```

You can now use `(S, w)` for:

- simulation / scenario analysis,
- plotting weighted histograms or KDEs,
- approximate distributional comparisons.

### 2. Random and stratified baselines

```python
from dd_coresets.ddc import (
    fit_random_coreset,
    fit_stratified_coreset,
)

# Random coreset (no domain knowledge)
S_rnd, w_rnd, info_rnd = fit_random_coreset(
    X,
    k=200,
    n0=20000,
    gamma=1.0,
    reweight_full=True,
    random_state=0,
)

# Stratified coreset (when you have strata)
# strata: 1D array, same length as X, e.g. segment, class, product line
strata = ...  # e.g. y labels or business segments

S_strat, w_strat, info_strat = fit_stratified_coreset(
    X,
    strata=strata,
    k=200,
    n0=20000,
    gamma=1.0,
    reweight_full=True,
    random_state=0,
)
```

Use these baselines to benchmark DDC on your data (moment errors, Wasserstein distances, etc.).

---

## API Overview

All functions assume `X` is a NumPy array of shape `(n, d)` with **preprocessed** numerical features (e.g. scaled, encoded, etc.).

### `fit_ddc_coreset`

```python
S, w, info = fit_ddc_coreset(
    X,
    k,
    n0=20000,
    m_neighbors=32,
    alpha=0.3,
    gamma=1.0,
    refine_iters=1,
    reweight_full=True,
    random_state=None,
)
```

- **Parameters**
  - `X`: `(n, d)` array-like, preprocessed data.
  - `k`: number of representatives.
  - `n0`: working sample size. If `None` or `>= n`, uses all data.
  - `m_neighbors`: kNN parameter for local density.
  - `alpha`: density–diversity trade-off (`0 ≈ diversity`, `1 ≈ density`).
  - `gamma`: kernel scale multiplier (used in soft assignment).
  - `refine_iters`: medoid refinement iterations (usually 1 is enough).
  - `reweight_full`: if `True`, reweights using the full dataset; else uses only the working sample.
  - `random_state`: RNG seed.

- **Returns**
  - `S`: `(k, d)` representatives (real data points).
  - `w`: `(k,)` weights (`w >= 0`, `sum(w) = 1`).
  - `info`: `CoresetInfo` with metadata (method name, n, n0, indices, params).

**Recommended use:**  
Default choice when you **do not yet know** which strata or labels matter. Good for EDA, exploratory simulation, and early-stage modelling.

---

### `fit_random_coreset`

```python
S, w, info = fit_random_coreset(
    X,
    k,
    n0=20000,
    gamma=1.0,
    reweight_full=True,
    random_state=None,
)
```

- Samples `k` points uniformly from a working sample (size `n0`) and applies the same soft-weighting scheme as DDC.

**Use case:**  
Baseline to compare against DDC and stratified; reflects what many teams do today (simple downsampling).

---

### `fit_stratified_coreset`

```python
S, w, info = fit_stratified_coreset(
    X,
    strata,
    k,
    n0=20000,
    gamma=1.0,
    reweight_full=True,
    random_state=None,
)
```

- **Parameters**
  - `X`: `(n, d)` data.
  - `strata`: 1D array of length `n` with stratum labels (e.g. product, region, class, risk band).
  - Other parameters analogous to `fit_random_coreset`.

- Internally:
  - Computes stratum frequencies on the working sample.
  - Allocates `k_g` reps per stratum ∝ frequency.
  - Samples uniformly inside each stratum.
  - Applies the same soft-weighting scheme as DDC.

**Use case:**  
When you **know** the relevant strata and must preserve their proportions (regulatory reporting, risk/actuarial slices, business segments).

---

## Experiments

The repo includes two example scripts under `experiments/`:

- `synthetic_ddc_vs_baselines.py`  
  5D Gaussian mixture with:
  - DDC vs Random vs Stratified,
  - metrics: mean/cov/corr errors, Wasserstein-1 marginals, KS,
  - basic plots.

- `multimodal_2d_ring_ddc.py`  
  2D example (3 Gaussians + ring) for visual intuition:
  - shows how DDC covers multiple modes and a ring structure with few reps.

Run:

```bash
python experiments/synthetic_ddc_vs_baselines.py
python experiments/multimodal_2d_ring_ddc.py
```

---

## When to use what?

- **DDC** (`fit_ddc_coreset`):  
  Default in **low-knowledge** regimes (no clear strata yet). Better than random sampling for a fixed `k`.

- **Stratified** (`fit_stratified_coreset`):  
  Preferred in **high-knowledge** regimes (well-defined strata aligned with the task, e.g. risk bands, products), especially when `k` is large enough.

- **Random** (`fit_random_coreset`):  
  Baseline and sanity check; still useful when you want the simplest possible comparison.

---