# Example Notebooks Plan

## Overview

Create 5 self-contained, Kaggle-style notebooks demonstrating DDC from basic to advanced, focusing on scenarios where DDC has clear advantages.

## Notebook Selection (Basic → Advanced)

### 1. `basic_tabular.ipynb` - **Basic**
**Target**: Beginners, first-time users  
**Focus**: Simple tabular data, basic API, comparison with Random  
**Key Features**:
- Simple synthetic dataset (Gaussian mixture, 2-3 clusters, 5-10 features)
- Basic `fit_ddc_coreset` usage with default parameters
- Comparison: DDC vs Random vs Stratified
- Metrics: Mean/Cov/Corr errors, Wasserstein-1 marginals
- Visualizations: 2D scatter (UMAP), marginal histograms, metrics table
- **Preset**: `mode="euclidean"` (default, simplest)

**Learning Outcomes**:
- How to install and import `dd-coresets`
- Basic API usage
- Understanding distributional metrics
- When DDC is better than Random (clustered data)

---

### 2. `multimodal_clusters.ipynb` - **Intermediate**
**Target**: Users familiar with basic DDC  
**Focus**: Clustered data, multiple modes, spatial coverage  
**Key Features**:
- Gaussian mixture with 4-8 clusters, varying sizes
- Demonstrates DDC advantage in well-separated clusters
- Comparison: DDC vs Random (shows Random can miss small clusters)
- Metrics: Spatial coverage per cluster, W1, KS, Cov error
- Visualizations: 2D scatter with cluster labels, coverage heatmap, metrics comparison
- **Preset**: `mode="euclidean"`, `preset="balanced"`

**Learning Outcomes**:
- DDC preserves cluster structure better
- Spatial coverage matters for clustered data
- How to interpret coverage metrics

---

### 3. `adaptive_distances.ipynb` - **Intermediate-Advanced**
**Target**: Users ready for advanced features  
**Focus**: Adaptive distances, presets, when to use what  
**Key Features**:
- Elliptical clusters (demonstrates adaptive advantage)
- Comparison: Euclidean vs Adaptive vs Auto
- Demonstrates presets: `fast`, `balanced`, `robust`
- Shows dimensionality-aware pipeline (auto PCA)
- Metrics: Joint and marginal distribution preservation
- Visualizations: Cluster shape comparison, preset comparison table
- **Presets**: All modes and presets demonstrated

**Learning Outcomes**:
- When to use adaptive distances (elliptical clusters, d ≥ 20)
- Understanding presets and their trade-offs
- Auto mode for dimensionality handling

---

### 4. `label_aware_classification.ipynb` - **Advanced**
**Target**: Users working on supervised problems  
**Focus**: Label-aware DDC, preserving class proportions  
**Key Features**:
- Real-world binary classification dataset (Adult Census or synthetic)
- Comparison: Global DDC vs Label-aware DDC vs Random vs Stratified
- Demonstrates `fit_ddc_coreset_by_label`
- Impact on downstream model performance (Logistic Regression)
- Metrics: Class proportions, distribution metrics, AUC, Brier Score
- Visualizations: ROC curves, class distribution comparison, metrics table
- **Preset**: `mode="auto"`, `preset="balanced"`

**Learning Outcomes**:
- When to use label-aware DDC (supervised problems)
- Preserving class proportions matters for classification
- Impact on model performance

**Note**: We already have `binary_classification_ddc.ipynb` - can refactor/enhance it

---

### 5. `high_dimensional.ipynb` - **Advanced**
**Target**: Advanced users, high-dimensional data  
**Focus**: High-dimensional data, PCA auto-reduction, performance  
**Key Features**:
- High-dimensional dataset (d=50-100, n=10k-50k)
- Demonstrates automatic PCA reduction (d ≥ 30)
- Comparison: Euclidean vs Adaptive vs Auto (with PCA)
- Performance comparison (time, memory)
- Metrics: Distribution preservation after PCA, explained variance
- Visualizations: PCA explained variance, 2D projections, performance comparison
- **Preset**: `mode="auto"` (triggers PCA), `preset="balanced"` or `"robust"`

**Learning Outcomes**:
- How DDC handles high-dimensional data
- Automatic dimensionality reduction
- Performance trade-offs

---

## Implementation Strategy

### Common Structure for All Notebooks

1. **Setup** (Cell 1)
   - Install `dd-coresets` (with fallback for Kaggle/Colab)
   - Imports (numpy, pandas, matplotlib, sklearn, dd_coresets)
   - Set random seed, plotting style

2. **Introduction** (Markdown)
   - What this notebook demonstrates
   - When to use this approach
   - Key concepts

3. **Data Generation/Loading** (Cell 2)
   - Generate or load dataset
   - Show data shape, basic stats
   - Visualize original data

4. **DDC Fitting** (Cell 3)
   - Fit DDC with appropriate preset
   - Show info dict (pipeline decisions, configs)
   - Explain parameters

5. **Baseline Comparison** (Cell 4)
   - Fit Random/Stratified baselines
   - Show comparison

6. **Metrics Computation** (Cell 5)
   - Compute all relevant metrics
   - Create comparison table

7. **Visualizations** (Cell 6)
   - Spatial coverage
   - Marginal distributions
   - Metrics comparison

8. **Discussion** (Markdown)
   - Key takeaways
   - When to use this approach
   - Links to other notebooks

### Code Style

- **Kaggle/Colab compatible**: Self-contained, no external files
- **Educational**: Clear comments, explanations
- **Professional**: No emojis, clean code
- **Reproducible**: Fixed random seeds, deterministic

### Dependencies

All notebooks use only:
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn` (for metrics, PCA, etc.)
- `dd-coresets`
- Optional: `umap-learn` for visualization (with fallback to PCA)

---

## File Structure

```
examples/
├── basic_tabular.ipynb              # 1. Basic
├── multimodal_clusters.ipynb        # 2. Intermediate
├── adaptive_distances.ipynb          # 3. Intermediate-Advanced
├── label_aware_classification.ipynb  # 4. Advanced (refactor existing)
├── high_dimensional.ipynb           # 5. Advanced
└── README.md                         # Index with links and descriptions
```

---

## README.md for Examples

```markdown
# Example Notebooks

This directory contains self-contained Jupyter notebooks demonstrating `dd-coresets` usage from basic to advanced scenarios.

## Notebooks (in order of complexity)

1. **[basic_tabular.ipynb](basic_tabular.ipynb)** - Basic usage with simple tabular data
2. **[multimodal_clusters.ipynb](multimodal_clusters.ipynb)** - Clustered data and spatial coverage
3. **[adaptive_distances.ipynb](adaptive_distances.ipynb)** - Adaptive distances and presets
4. **[label_aware_classification.ipynb](label_aware_classification.ipynb)** - Supervised problems with label-aware DDC
5. **[high_dimensional.ipynb](high_dimensional.ipynb)** - High-dimensional data and automatic PCA

## Running the Notebooks

All notebooks are self-contained and can be run on:
- **Kaggle**: Upload notebook, enable internet, run all cells
- **Google Colab**: Upload notebook, run all cells
- **Local Jupyter**: Install dependencies, run all cells

## Dependencies

- `dd-coresets` (install via `pip install dd-coresets`)
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`
- Optional: `umap-learn` (for visualization)
```

---

## Priority Order

1. **basic_tabular.ipynb** - Start here, most users
2. **multimodal_clusters.ipynb** - Shows clear DDC advantage
3. **adaptive_distances.ipynb** - Demonstrates new v0.2.0 features
4. **label_aware_classification.ipynb** - Refactor existing notebook
5. **high_dimensional.ipynb** - Advanced use case

---

## Success Criteria

Each notebook should:
- ✅ Run end-to-end without errors
- ✅ Be self-contained (no external data files)
- ✅ Clearly demonstrate DDC advantage
- ✅ Use appropriate presets for the scenario
- ✅ Include clear explanations and takeaways
- ✅ Be reproducible (fixed seeds)
- ✅ Work on Kaggle/Colab

