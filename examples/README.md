# Example Notebooks

This directory contains self-contained Jupyter notebooks demonstrating `dd-coresets` usage from basic to advanced scenarios.

All notebooks are designed to run on **Kaggle** and **Google Colab** without external dependencies.

## Notebooks (in order of complexity)

### 1. [basic_tabular.ipynb](basic_tabular.ipynb) - **Basic**

**Target**: Beginners, first-time users  
**Focus**: Simple tabular data, basic API, comparison with Random

**What you'll learn**:
- How to install and import `dd-coresets`
- Basic API: `fit_ddc_coreset`, `fit_random_coreset`, `fit_stratified_coreset`
- Understanding distributional metrics (Mean, Covariance, Wasserstein-1)
- When DDC is better than Random (clustered data)

**Dataset**: Gaussian mixture with 3 clusters, 8 features  
**Preset**: `mode="euclidean"` (default, simplest)

---

### 2. [multimodal_clusters.ipynb](multimodal_clusters.ipynb) - **Intermediate**

**Target**: Users familiar with basic DDC  
**Focus**: Clustered data, multiple modes, spatial coverage

**What you'll learn**:
- DDC preserves cluster structure better
- Spatial coverage matters for clustered data
- How to interpret coverage metrics

**Dataset**: Gaussian mixture with 6 clusters of varying sizes  
**Preset**: `mode="euclidean"`, `preset="balanced"`

---

### 3. [adaptive_distances.ipynb](adaptive_distances.ipynb) - **Intermediate-Advanced**

**Target**: Users ready for advanced features  
**Focus**: Adaptive distances, presets, when to use what

**What you'll learn**:
- When to use adaptive distances (elliptical clusters, d ≥ 20)
- Understanding presets: `fast`, `balanced`, `robust`
- Auto mode for dimensionality handling

**Dataset**: Elliptical clusters (15 features)  
**Presets**: All modes and presets demonstrated

---

### 4. [label_aware_classification.ipynb](label_aware_classification.ipynb) - **Advanced**

**Target**: Users working on supervised problems  
**Focus**: Label-aware DDC, preserving class proportions

**What you'll learn**:
- When to use label-aware DDC (supervised problems)
- Preserving class proportions matters for classification
- Impact on downstream model performance

**Dataset**: Binary classification (Adult Census Income or synthetic)  
**Preset**: `mode="auto"`, `preset="balanced"`

---

### 5. [high_dimensional.ipynb](high_dimensional.ipynb) - **Advanced**

**Target**: Advanced users, high-dimensional data  
**Focus**: High-dimensional data, PCA auto-reduction, performance

**What you'll learn**:
- How DDC handles high-dimensional data
- Automatic dimensionality reduction
- Performance trade-offs

**Dataset**: High-dimensional (60 features, 20k samples)  
**Preset**: `mode="auto"` (triggers PCA), `preset="balanced"`

---

## Running the Notebooks

All notebooks are self-contained and can be run on:

- **Kaggle**: Upload notebook, enable internet, run all cells
- **Google Colab**: Upload notebook, run all cells
- **Local Jupyter**: Install dependencies, run all cells

### Installation

For Kaggle/Colab, uncomment the installation cell at the beginning:

```python
# !pip install dd-coresets
```

For local Jupyter:

```bash
pip install dd-coresets
```

## Dependencies

All notebooks use only standard libraries:

- `dd-coresets` (install via `pip install dd-coresets`)
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`
- Optional: `umap-learn` (for visualization, with PCA fallback)

## Notebook Structure

Each notebook follows this structure:

1. **Introduction** - What you'll learn, dataset description
2. **Setup** - Installation and imports
3. **Data Generation/Loading** - Create or load dataset
4. **Fit Coresets** - Apply DDC and baselines
5. **Compute Metrics** - Distributional metrics comparison
6. **Visualizations** - Spatial coverage, marginals, metrics
7. **Key Takeaways** - When to use what, next steps

## Additional Resources

- **Documentation**: See `docs/DDC_ADVANTAGE_CASES.md` for comprehensive analysis
- **Technical Details**: See `docs/ADAPTIVE_DISTANCES_EXPLAINED.md` for adaptive distances
- **API Reference**: See main `README.md` for full API documentation

## Contributing

If you create new example notebooks, please:

1. Follow the structure above
2. Include clear explanations
3. Use fixed random seeds for reproducibility
4. Test on Kaggle/Colab
5. Update this README

