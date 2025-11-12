# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

