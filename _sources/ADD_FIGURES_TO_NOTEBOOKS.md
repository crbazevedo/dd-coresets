# Adding Figures to Notebooks - Implementation Guide

## Overview

This guide explains how to modify tutorial notebooks to save figures and reference them in the documentation.

## Directory Structure

```
docs/book/
├── images/
│   ├── tutorials/
│   │   ├── basic_tabular/
│   │   ├── multimodal_clusters/
│   │   ├── adaptive_distances/
│   │   ├── label_aware_classification/
│   │   └── high_dimensional/
│   ├── concepts/
│   └── pipelines/
└── tutorials/
    ├── basic_tabular.ipynb
    └── ...
```

## Implementation Steps

### Step 1: Modify Notebook Cells

For each visualization cell, add:

```python
# Before plt.show()
import os
os.makedirs('images/tutorials/basic_tabular', exist_ok=True)
plt.savefig('images/tutorials/basic_tabular/spatial_coverage.png', 
            dpi=150, bbox_inches='tight')
plt.show()
```

### Step 2: Add Markdown References

After each visualization, add:

```markdown
![Spatial Coverage Comparison](images/tutorials/basic_tabular/spatial_coverage.png)

*Figure 1: 2D projection showing spatial coverage of DDC (left), Random (center), and Stratified (right) coresets. DDC ensures all clusters are represented, even small ones.*
```

### Step 3: Figure Naming Convention

- **Format**: `{notebook_name}_{figure_type}_{description}.png`
- **Examples**:
  - `basic_tabular_spatial_coverage.png`
  - `basic_tabular_marginal_distributions.png`
  - `basic_tabular_metrics_comparison.png`
  - `adaptive_distances_euclidean_vs_adaptive.png`

### Step 4: Figure Types by Notebook

#### basic_tabular.ipynb
1. `spatial_coverage_2d.png` - 2D projection with all methods
2. `marginal_distributions.png` - Histogram comparison (4 features)
3. `metrics_comparison.png` - Bar charts of metrics

#### multimodal_clusters.ipynb
1. `cluster_coverage.png` - Cluster representation comparison
2. `spatial_coverage_metrics.png` - Coverage statistics

#### adaptive_distances.ipynb
1. `euclidean_vs_adaptive.png` - Side-by-side comparison
2. `elliptical_cluster_demo.png` - Adaptive distance advantage

#### label_aware_classification.ipynb
1. `pca_projection.png` - 2D PCA with class labels
2. `roc_curves.png` - Model performance comparison

#### high_dimensional.ipynb
1. `pca_explained_variance.png` - Variance retention
2. `projection_comparison.png` - Euclidean vs Auto mode

## Best Practices

1. **Alt Text**: Always include descriptive alt text
2. **Captions**: Add contextual captions explaining what the figure shows
3. **Resolution**: Use 150 DPI for PNG, SVG for diagrams
4. **Size**: Keep file sizes reasonable (< 500KB per image)
5. **Context**: Reference figures where they add value, not just decoration

## Automation Script

A script can be created to:
1. Execute notebooks
2. Extract figures automatically
3. Generate markdown references
4. Validate image paths

