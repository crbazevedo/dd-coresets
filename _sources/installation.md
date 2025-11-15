# Installation

## Requirements

- Python >= 3.8
- NumPy >= 1.24
- scikit-learn >= 1.2

## Install from PyPI

```bash
pip install dd-coresets
```

## Install from GitHub

```bash
pip install git+https://github.com/crbazevedo/dd-coresets.git
```

## Install for Development

```bash
git clone https://github.com/crbazevedo/dd-coresets.git
cd dd-coresets
pip install -e .
```

## Verify Installation

```python
import dd_coresets
print(dd_coresets.__version__)

# Test basic functionality
from dd_coresets import fit_ddc_coreset
import numpy as np

X = np.random.randn(1000, 5)
S, w, info = fit_ddc_coreset(X, k=50)
print(f"Created coreset with {len(S)} representatives")
```

## Optional Dependencies

For visualization and advanced features:
- `matplotlib >= 3.7` (for plotting)
- `pandas >= 2.0` (for data handling)
- `umap-learn` (for dimensionality reduction visualization)

These are included in the base installation but can be installed separately if needed.

