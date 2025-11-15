#!/usr/bin/env python3
"""
Add figure saving and markdown references to tutorial notebooks.

This script:
1. Modifies visualization cells to save figures
2. Adds markdown cells with figure references after visualizations
3. Creates necessary directories
"""

import json
import os
from pathlib import Path

# Notebook configurations
NOTEBOOKS = {
    'basic_tabular': {
        'figures': [
            {
                'cell_idx': 14,  # Original data visualization
                'filename': 'original_data_2d.png',
                'caption': 'Original dataset projected to 2D using UMAP (or PCA). Colors represent different clusters.',
                'alt': '2D projection of original Gaussian mixture data with 3 clusters'
            },
            {
                'cell_idx': 24,  # Spatial coverage
                'filename': 'spatial_coverage_comparison.png',
                'caption': 'Spatial coverage comparison: DDC (left), Random (center), and Stratified (right) coresets. Point sizes are proportional to weights. DDC ensures all clusters are represented, even small ones.',
                'alt': '2D projection comparing spatial coverage of DDC, Random, and Stratified coresets'
            },
            {
                'cell_idx': 25,  # Marginal distributions
                'filename': 'marginal_distributions.png',
                'caption': 'Marginal distribution comparison for the first 4 features. Gray histogram shows full data; colored lines show weighted coreset distributions. DDC better preserves the shape of marginal distributions.',
                'alt': 'Histogram comparison of marginal distributions for DDC, Random, and Stratified coresets'
            },
            {
                'cell_idx': 26,  # Metrics comparison
                'filename': 'metrics_comparison.png',
                'caption': 'Distributional metrics comparison (left) and DDC improvement over Random (right). DDC excels at mean preservation but may trade off some covariance accuracy for better cluster coverage.',
                'alt': 'Bar charts comparing distributional metrics across methods'
            }
        ]
    },
    'multimodal_clusters': {
        'figures': [
            {
                'cell_idx': 8,  # Cluster coverage
                'filename': 'cluster_coverage.png',
                'caption': 'Cluster coverage visualization showing how DDC ensures all clusters are represented.',
                'alt': 'Cluster coverage comparison'
            },
            {
                'cell_idx': 10,  # Spatial coverage metrics
                'filename': 'spatial_coverage_metrics.png',
                'caption': 'Spatial coverage metrics comparing minimum and mean distances to nearest representatives.',
                'alt': 'Spatial coverage metrics bar chart'
            }
        ]
    },
    'adaptive_distances': {
        'figures': [
            {
                'cell_idx': 10,  # Euclidean vs Adaptive
                'filename': 'euclidean_vs_adaptive.png',
                'caption': 'Comparison of Euclidean (left) and Adaptive (right) distance modes. Adaptive distances better capture elliptical cluster shapes.',
                'alt': 'Euclidean vs Adaptive distance coreset comparison'
            },
            {
                'cell_idx': 12,  # Elliptical cluster demo
                'filename': 'elliptical_cluster_demo.png',
                'caption': 'Demonstration of adaptive distances on elliptical clusters. Mahalanobis distance accounts for local covariance structure.',
                'alt': 'Elliptical cluster visualization with adaptive distances'
            }
        ]
    },
    'label_aware_classification': {
        'figures': [
            {
                'cell_idx': 20,  # PCA projection
                'filename': 'pca_projection_with_labels.png',
                'caption': '2D PCA projection showing label-aware DDC coreset. Colors represent class labels; point sizes are proportional to weights.',
                'alt': 'PCA projection of label-aware DDC coreset with class labels'
            },
            {
                'cell_idx': 22,  # ROC curves
                'filename': 'roc_curves_comparison.png',
                'caption': 'ROC curves comparing model performance on different coresets. Label-aware DDC preserves class proportions while maintaining distributional fidelity.',
                'alt': 'ROC curve comparison for different coreset methods'
            }
        ]
    },
    'high_dimensional': {
        'figures': [
            {
                'cell_idx': 8,  # PCA explained variance
                'filename': 'pca_explained_variance.png',
                'caption': 'PCA explained variance showing how dimensionality reduction preserves information. The auto mode automatically applies PCA when d ≥ 50.',
                'alt': 'PCA explained variance plot'
            },
            {
                'cell_idx': 9,  # Projection comparison
                'filename': 'projection_comparison.png',
                'caption': '2D projection comparison: Euclidean mode (left) vs Auto mode with PCA (right). Auto mode handles high-dimensional data more effectively.',
                'alt': '2D projection comparison for Euclidean and Auto modes'
            }
        ]
    }
}

def add_figure_saving_to_cell(cell, image_dir, filename):
    """Add figure saving code before plt.show()."""
    if cell['cell_type'] != 'code':
        return False
    
    source = ''.join(cell.get('source', []))
    if 'plt.show()' not in source:
        return False
    
    # Create directory creation and save code
    save_code = f"""
# Save figure
import os
os.makedirs('{image_dir}', exist_ok=True)
plt.savefig('{image_dir}/{filename}', dpi=150, bbox_inches='tight')
"""
    
    # Insert before plt.show()
    new_source = source.replace('plt.show()', save_code + '\nplt.show()')
    cell['source'] = new_source.split('\n')
    return True

def add_figure_markdown_cell(notebook, insert_after_idx, filename, caption, alt):
    """Add a markdown cell with figure reference after specified cell."""
    markdown_cell = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            f'![{alt}]({filename})\n',
            f'\n',
            f'*{caption}*\n'
        ]
    }
    
    # Insert after the specified cell
    notebook['cells'].insert(insert_after_idx + 1, markdown_cell)
    return insert_after_idx + 2  # Return new index accounting for insertion

def process_notebook(notebook_name, config):
    """Process a single notebook."""
    notebook_path = Path(f'tutorials/{notebook_name}.ipynb')
    
    if not notebook_path.exists():
        print(f"⚠️  Notebook not found: {notebook_path}")
        return False
    
    print(f"\n📝 Processing {notebook_name}.ipynb...")
    
    # Load notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Create image directory
    image_dir = f'images/tutorials/{notebook_name}'
    os.makedirs(image_dir, exist_ok=True)
    
    # Process each figure
    offset = 0  # Track cell index offset from insertions
    for fig_config in config['figures']:
        cell_idx = fig_config['cell_idx'] + offset
        filename = fig_config['filename']
        caption = fig_config['caption']
        alt = fig_config['alt']
        
        if cell_idx >= len(notebook['cells']):
            print(f"  ⚠️  Cell {cell_idx} not found, skipping {filename}")
            continue
        
        cell = notebook['cells'][cell_idx]
        
        # Add figure saving
        if add_figure_saving_to_cell(cell, image_dir, filename):
            print(f"  ✓ Added save code to cell {cell_idx}: {filename}")
        else:
            print(f"  ⚠️  Could not add save code to cell {cell_idx}")
        
        # Add markdown reference
        offset = add_figure_markdown_cell(
            notebook, cell_idx + offset, 
            f'{image_dir}/{filename}', caption, alt
        ) - cell_idx - 1
        print(f"  ✓ Added markdown reference for {filename}")
    
    # Save notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"  ✅ Saved {notebook_name}.ipynb")
    return True

def main():
    """Main function."""
    print("=" * 70)
    print("Adding Figures to Tutorial Notebooks")
    print("=" * 70)
    
    # Change to docs/book directory
    os.chdir(Path(__file__).parent)
    
    success_count = 0
    for notebook_name, config in NOTEBOOKS.items():
        if process_notebook(notebook_name, config):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"✅ Processed {success_count}/{len(NOTEBOOKS)} notebooks")
    print("=" * 70)

if __name__ == '__main__':
    main()

