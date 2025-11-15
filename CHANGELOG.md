# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-14

### Added

- **Adaptive distance support**: Local Mahalanobis density estimation for elliptical clusters
  - `mode="adaptive"` for explicit adaptive distances
  - `mode="auto"` for automatic selection based on dimensionality
  - OAS shrinkage for robust covariance estimation
  - Cholesky-based implementation (no explicit matrix inversion)
- **Pipeline presets**: Simple configuration via `preset` parameter
  - `"fast"`: Quick runs (fewer neighbors, 1 iteration)
  - `"balanced"`: Default (good trade-off)
  - `"robust"`: More neighbors, 2 iterations (better quality)
  - `"manual"`: Full control via `distance_cfg` and `pipeline_cfg` dicts
- **Dimensionality-aware pipelines**: Automatic PCA reduction for high-dimensional data (d ≥ 30)
  - Configurable via `pipeline_cfg` (threshold, variance retention, component cap)
  - Representatives always returned in original feature space
- **Label-wise wrapper**: `fit_ddc_coreset_by_label` preserves class proportions
  - Allocates k per class proportionally
  - Runs DDC separately per class for better within-class structure
- **Extended info dict**: Rich metadata including pipeline decisions, fallbacks, configs
- **Tests**: Comprehensive test suite (`tests/test_adaptive_density.py`)
  - Elliptical cluster comparison (Euclidean vs Adaptive)
  - Feasibility guards (m_neighbors <= d_eff)
  - High-dimensional PCA reduction
  - Backward compatibility checks
  - Label-wise wrapper validation
- **Examples**: Updated `adaptive_distance_demo.py` using new API
- **Documentation**: README updated with adaptive distances section and usage guide

### Changed

- **API**: `fit_ddc_coreset` now returns `dict` instead of `CoresetInfo` dataclass
  - Backward compatible: old code still works (info dict has same fields)
  - Extended with pipeline info, configs, PCA details
- **Default behavior**: `mode="euclidean"` (backward compatible)
  - Old API calls work unchanged
  - New features opt-in via `mode` and `preset` parameters
- **n0 default**: Changed from `20000` to `None` (which internally uses 20000)
  - Backward compatible: explicit `n0=20000` still works
  - New default allows `n0=None` for cleaner API

### Deprecated

- Legacy kwargs: `use_adaptive_distance`, `m_neighbors`, `adaptive_iterations`, etc.
  - Emit `DeprecationWarning` when used
  - Mapped to new `mode`/`preset`/`*_cfg` system
  - Will be removed in a future version

### Technical Details

- **No SciPy dependency**: Core library uses only NumPy + scikit-learn
- **Cholesky-based**: Never inverts covariance matrices explicitly
- **OAS shrinkage**: Default for robust covariance estimation
- **Iterative refinement**: Up to 2 iterations for adaptive density (configurable)
- **Fallback handling**: Clean fallbacks when adaptive not feasible (m_neighbors <= d)

## [0.1.3] - 2025-11-12

### Added

- Comprehensive README improvements with badges (PyPI, License, Python versions)
- Concrete motivation section with mini-case example ("30M rows → 500 points + weights")
- "When to Use / When Not to Use" section with clear guidance
- "Complexity and Scale" section with asymptotic analysis and practical scaling examples
- "Relation to Existing Methods" section comparing DDC with alternatives
- "Examples / Notebooks" section with placeholder links
- Citation section with BibTeX format

### Changed

- Reorganized README content for better flow and readability
- Standardized API naming throughout documentation
- Improved visual examples organization

## [0.1.2] - 2025-11-12

### Fixed

- Fix image URLs in README to use absolute GitHub URLs for PyPI compatibility

## [0.1.1] - 2025-11-11

### Added

- K-medoids baseline implementation (`fit_kmedoids_coreset`) with PAM-like optimization
- Two Moons experiment (`experiments/two_moons_ddc.py`) comparing DDC vs Random vs K-medoids
- Memory-efficient distance computation (on-demand, avoids O(n²) allocation)
- Optimized k-means++ initialization for k-medoids

### Changed

- Updated README with Two Moons results and quantitative comparisons
- Enhanced documentation with K-medoids API and use cases

### Notes

- K-medoids uses canonical sklearn `make_moons` dataset generation
- All optimizations maintain backward compatibility

## [0.1.0-alpha] - 2025-11-11

### Added

- First public alpha release of `dd-coresets`.
- Implemented Density–Diversity Coreset (DDC) algorithm.
- Provided NumPy + scikit-learn reference implementation.
- Added CLI and experiment scripts for multimodal ring dataset.
- Published to [PyPI](https://pypi.org/project/dd-coresets/).

### Notes

- Experimental release for community testing and feedback.
- API subject to change before 1.0.

