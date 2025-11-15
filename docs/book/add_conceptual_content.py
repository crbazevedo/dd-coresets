#!/usr/bin/env python3
"""
Add conceptual content to notebooks for Jupyter Book documentation.
Inserts markdown cells with theoretical explanations at strategic points.
"""

import json
import sys
from pathlib import Path

def add_conceptual_cells(notebook_path, notebook_type):
    """Add conceptual markdown cells to a notebook based on its type."""
    
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    new_cells = []
    
    if notebook_type == 'basic_tabular':
        # Add conceptual content after dataset generation
        for i, cell in enumerate(cells):
            new_cells.append(cell)
            
            # After dataset generation (around cell with make_blobs)
            if i > 0 and 'make_blobs' in ''.join(cell.get('source', [])):
                new_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "### Why Gaussian Mixtures?\n",
                        "\n",
                        "Gaussian mixtures with well-separated clusters are ideal for demonstrating DDC because:\n",
                        "\n",
                        "- **Clustered structure**: DDC excels when data has clear modes (clusters)\n",
                        "- **Spatial coverage**: DDC guarantees all clusters are represented, even small ones\n",
                        "- **Distribution preservation**: The weighted coreset preserves both the location (mean) and shape (covariance) of each cluster\n",
                        "\n",
                        "**Conceptual note**: In high dimensions, k-NN density estimation works well when clusters are separated. Points in dense regions (clusters) have many close neighbors, leading to high density estimates. DDC uses these estimates to prioritize important regions while ensuring diversity (spatial coverage)."
                    ]
                })
            
            # After coreset fitting
            if i > 0 and 'fit_ddc_coreset' in ''.join(cell.get('source', [])):
                new_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "### What Happens During DDC Fitting?\n",
                        "\n",
                        "The `fit_ddc_coreset` function performs three main steps:\n",
                        "\n",
                        "1. **Density Estimation**: Estimates local density for each point using k-nearest neighbors. Points in dense regions (clusters) get high density scores.\n",
                        "\n",
                        "2. **Greedy Selection**: Iteratively selects points that balance high density (important regions) with diversity (spatial coverage). The `alpha` parameter (default 0.3) controls this trade-off.\n",
                        "\n",
                        "3. **Weight Assignment**: Assigns weights to selected points using soft assignments. A point with weight 0.1 \"stands for\" 10% of the original data in that region.\n",
                        "\n",
                        "**Why weights matter**: Unlike simple sampling where each point represents 1/n of the data, weights allow a small coreset to accurately represent the full distribution. This is similar to how a histogram uses bin counts, but DDC uses actual data points with weights.\n",
                        "\n",
                        "See [Algorithm Overview](../concepts/algorithm.md) for more details."
                    ]
                })
    
    elif notebook_type == 'multimodal_clusters':
        # Add after cluster visualization
        for i, cell in enumerate(cells):
            new_cells.append(cell)
            
            cell_source = ''.join(cell.get('source', [])).lower()
            if i > 0 and 'scatter' in cell_source and 'cluster' in cell_source:
                new_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "### Spatial Coverage: Why It Matters\n",
                        "\n",
                        "When data has multiple clusters of varying sizes, **spatial coverage** becomes crucial. Random sampling may miss small clusters entirely, especially when k (number of representatives) is small relative to the number of clusters.\n",
                        "\n",
                        "**DDC's guarantee**: The diversity term in DDC's selection score ensures that representatives are well-distributed across all clusters. This is similar to facility location problems, where we want to place k facilities to serve all regions.\n",
                        "\n",
                        "**Mathematical intuition**: The selection score combines density (p(x)^α) and diversity (distance^(1-α)). The diversity term prevents selecting points too close to already-selected representatives, naturally ensuring spatial coverage.\n",
                        "\n",
                        "See [Algorithm Overview](../concepts/algorithm.md) for the mathematical details."
                    ]
                })
    
    elif notebook_type == 'adaptive_distances':
        # Add after adaptive fitting
        for i, cell in enumerate(cells):
            new_cells.append(cell)
            
            if i > 0 and 'mode=\"adaptive\"' in ''.join(cell.get('source', [])):
                new_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "### Why Adaptive Distances?\n",
                        "\n",
                        "In high dimensions (d ≥ 20), Euclidean distance becomes less meaningful due to the **curse of dimensionality**: all points become roughly equidistant, and k-NN density estimates become unreliable.\n",
                        "\n",
                        "**Mahalanobis distance solution**: Adaptive distances use local covariance matrices to account for the shape of the data. Instead of measuring distance in the original space, we \"stretch\" the space along principal directions, making clusters more \"spherical\" in the transformed space.\n",
                        "\n",
                        "**Mathematical intuition**: Instead of ||x - y||², we use (x-y)ᵀ C⁻¹ (x-y), where C is the local covariance. This accounts for elliptical clusters and improves density estimation accuracy.\n",
                        "\n",
                        "**When to use**: Adaptive distances help when:\n",
                        "- Data has elliptical (non-spherical) clusters\n",
                        "- Dimensions are medium to high (d ≥ 20)\n",
                        "- You want better marginal distribution preservation\n",
                        "\n",
                        "See [Adaptive Distances](../concepts/adaptive_distances.md) for a detailed explanation."
                    ]
                })
    
    elif notebook_type == 'label_aware':
        # Add after label-aware fitting
        for i, cell in enumerate(cells):
            new_cells.append(cell)
            
            if i > 0 and 'fit_ddc_coreset_by_label' in ''.join(cell.get('source', [])):
                new_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "### Why Label-Aware DDC?\n",
                        "\n",
                        "Standard DDC is **unsupervised**—it only looks at feature distributions, not labels. For supervised learning tasks, this can distort class proportions because DDC may select more points from dense classes.\n",
                        "\n",
                        "**Label-aware solution**: `fit_ddc_coreset_by_label` applies DDC separately within each class, then combines the results. This:\n",
                        "\n",
                        "1. **Preserves class proportions**: Allocates k per class proportionally\n",
                        "2. **Preserves within-class structure**: Each class gets its own density-diversity coreset\n",
                        "3. **Maintains distributional fidelity**: The weighted coreset still approximates the full distribution\n",
                        "\n",
                        "**Trade-off**: Label-aware DDC may have slightly worse marginal distribution metrics (because it optimizes within-class structure), but it preserves class balance and often leads to better model performance (AUC).\n",
                        "\n",
                        "**When to use**: Always use label-aware DDC for supervised learning tasks where class balance matters."
                    ]
                })
    
    elif notebook_type == 'high_dimensional':
        # Add after PCA discussion
        for i, cell in enumerate(cells):
            new_cells.append(cell)
            
            if i > 0 and 'PCA' in ''.join(cell.get('source', [])) and 'explained' in ''.join(cell.get('source', []).lower()):
                new_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "### Why PCA Before Density Estimation?\n",
                        "\n",
                        "The **curse of dimensionality** makes k-NN density estimation unreliable in high dimensions (d ≥ 30). When dimensions are high:\n",
                        "\n",
                        "- All points become roughly equidistant\n",
                        "- Volume concentrates in the \"shell\" of high-dimensional spheres\n",
                        "- Density estimates become uniform (no discrimination)\n",
                        "\n",
                        "**PCA solution**: DDC automatically applies PCA when d ≥ 30 (configurable via `dim_threshold_adaptive`). PCA reduces dimensions while retaining most variance (typically 95%), making density estimation feasible.\n",
                        "\n",
                        "**Important**: Representatives are always returned in the **original feature space**, not the reduced space. DDC applies PCA internally for density estimation, but maps representatives back to original dimensions.\n",
                        "\n",
                        "**Mathematical intuition**: PCA finds the directions of maximum variance. By working in this reduced space, we can estimate density more accurately, then project back to original space.\n",
                        "\n",
                        "See [Density Estimation](../concepts/density_estimation.md) for more on the curse of dimensionality."
                    ]
                })
    
    # If no matches, return original
    if len(new_cells) == len(cells):
        return nb
    
    nb['cells'] = new_cells
    return nb

def main():
    """Process all notebooks in tutorials/ directory."""
    tutorials_dir = Path(__file__).parent / 'tutorials'
    
    notebook_map = {
        'basic_tabular.ipynb': 'basic_tabular',
        'multimodal_clusters.ipynb': 'multimodal_clusters',
        'adaptive_distances.ipynb': 'adaptive_distances',
        'label_aware_classification.ipynb': 'label_aware',
        'high_dimensional.ipynb': 'high_dimensional',
    }
    
    for nb_file, nb_type in notebook_map.items():
        nb_path = tutorials_dir / nb_file
        if not nb_path.exists():
            print(f"Skipping {nb_file} (not found)")
            continue
        
        print(f"Processing {nb_file}...")
        nb = add_conceptual_cells(nb_path, nb_type)
        
        with open(nb_path, 'w') as f:
            json.dump(nb, f, indent=1)
        
        print(f"  Added conceptual content to {nb_file}")

if __name__ == '__main__':
    main()

