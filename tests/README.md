# Test Suite for dd-coresets

## Overview

Comprehensive test suite with **49 test functions** covering:
- Adaptive distance density estimation
- Pipeline functions and presets
- Core DDC functions
- Edge cases and error handling

## Test Files

### `test_adaptive_density.py` (5 tests)
- Elliptical cluster comparison (Euclidean vs Adaptive)
- Feasibility guards (m_neighbors <= d_eff)
- High-dimensional PCA reduction
- Backward compatibility
- Label-wise wrapper

### `test_pipelines.py` (10 tests)
- Preset validation
- Pipeline choice logic (euclidean/adaptive/auto)
- Dimensionality reduction (PCA)
- Label-wise wrapper (basic, proportions, single class)

### `test_ddc_core.py` (18 tests)
- Core functions: `_density_knn_euclidean`, `_select_reps_greedy`, `_soft_assign_weights`, `_medoid_refinement`
- `fit_ddc_coreset`: all data, n0 variations, presets, manual config, legacy kwargs
- Baselines: `fit_random_coreset`, `fit_stratified_coreset`, `fit_kmedoids_coreset`
- High-dim PCA, adaptive shrinkage methods, iterations

### `test_edge_cases.py` (16 tests)
- Edge cases: k >= n, k=1, very small datasets, single feature
- Invalid parameters (mode, preset)
- Parameter variations: alpha (0, 1), gamma, refine_iters=0
- Special data: constant, duplicates
- Label-wise: imbalanced, single-sample class
- Adaptive: m_neighbors too small, PCA edge cases

## Running Tests

### With pytest (if installed)
```bash
pytest tests/ -v
```

### Without pytest (direct execution)
```bash
# Run all tests
PYTHONPATH=. python3 tests/test_adaptive_density.py
PYTHONPATH=. python3 tests/test_pipelines.py
PYTHONPATH=. python3 tests/test_ddc_core.py
PYTHONPATH=. python3 tests/test_edge_cases.py
```

### Quick check
```bash
PYTHONPATH=. python3 -c "
import sys, os
sys.path.insert(0, '.')
for tf in ['tests/test_adaptive_density.py', 'tests/test_pipelines.py', 
           'tests/test_ddc_core.py', 'tests/test_edge_cases.py']:
    exec(open(tf).read().replace('if __name__', 'if False'))
    print(f'{tf}: OK')
"
```

## Coverage Areas

### ✅ Core Functions
- Density estimation (Euclidean, Adaptive)
- Greedy selection
- Soft assignment and weighting
- Medoid refinement

### ✅ API Functions
- `fit_ddc_coreset` (all modes, presets, configs)
- `fit_random_coreset`
- `fit_stratified_coreset`
- `fit_kmedoids_coreset`
- `fit_ddc_coreset_by_label`

### ✅ Pipeline Functions
- `choose_pipeline`
- `reduce_dimensionality_if_needed`
- Preset handling

### ✅ Edge Cases
- Small datasets (n < m_neighbors)
- k >= n
- k = 1
- Single feature (d=1)
- Constant/duplicate data
- Invalid parameters
- Parameter boundaries (alpha=0, alpha=1, gamma variations)

### ✅ Adaptive Features
- Shrinkage methods (OAS, LedoitWolf, none)
- Iterations
- Feasibility guards
- Fallback handling

### ✅ Backward Compatibility
- Legacy kwargs mapping
- Deprecation warnings
- Default behavior preservation

## Test Statistics

- **Total test functions**: 49
- **Test files**: 4
- **Coverage**: Core functions, API, pipelines, edge cases
- **Status**: ✅ All tests passing

## Notes

- Tests use NumPy-only implementations for metrics (no SciPy dependency)
- Tests are designed to run with or without pytest
- All tests use fixed random seeds for reproducibility
- Edge cases include error handling and validation tests

