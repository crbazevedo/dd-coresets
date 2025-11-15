#!/usr/bin/env python3
"""
Generate all figures for tutorial notebooks.

This script executes notebooks programmatically to ensure all cells run
and all figures are generated, even if nbconvert has issues.
"""

import sys
import os
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, '/Users/59388/coding/dd-coresets')

# Setup matplotlib
import matplotlib
matplotlib.use('Agg')

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_classification, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

# Try to import umap
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    umap = None

# Import dd-coresets
from dd_coresets import (
    fit_ddc_coreset,
    fit_random_coreset,
    fit_stratified_coreset,
    fit_ddc_coreset_by_label,
)

def execute_notebook(notebook_path):
    """Execute a notebook and generate all figures."""
    print(f"\n{'=' * 70}")
    print(f"Executing: {notebook_path}")
    print(f"{'=' * 70}")
    
    # Load notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Create namespace with all imports
    namespace = {
        '__name__': '__main__',
        '__builtins__': __builtins__,
        'np': np,
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'make_blobs': make_blobs,
        'make_classification': make_classification,
        'make_moons': make_moons,
        'StandardScaler': StandardScaler,
        'PCA': PCA,
        'NearestNeighbors': NearestNeighbors,
        'train_test_split': train_test_split,
        'LogisticRegression': LogisticRegression,
        'roc_auc_score': roc_auc_score,
        'roc_curve': roc_curve,
        'brier_score_loss': brier_score_loss,
        'fit_ddc_coreset': fit_ddc_coreset,
        'fit_random_coreset': fit_random_coreset,
        'fit_stratified_coreset': fit_stratified_coreset,
        'fit_ddc_coreset_by_label': fit_ddc_coreset_by_label,
        'HAS_UMAP': HAS_UMAP,
        'umap': umap,
        'os': os,
        'Path': Path,
    }
    
    # Execute cells
    executed = 0
    errors = 0
    figures_generated = []
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell.get('source', []))
        
        # Skip empty cells
        if not source.strip():
            continue
        
        # Skip pure comment cells (unless they have plt.savefig)
        if source.strip().startswith('#') and 'plt.savefig' not in source:
            continue
        
        try:
            exec(source, namespace)
            executed += 1
            
            # Check if figure was saved
            if 'plt.savefig' in source:
                # Extract path from savefig call
                import re
                match = re.search(r"plt\.savefig\(['\"]([^'\"]+)['\"]", source)
                if match:
                    fig_path = match.group(1)
                    if os.path.exists(fig_path):
                        figures_generated.append(fig_path)
                        size = os.path.getsize(fig_path) / 1024
                        print(f"  ✅ Cell {i}: Generated {os.path.basename(fig_path)} ({size:.1f} KB)")
        
        except Exception as e:
            errors += 1
            if errors <= 3:  # Show first 3 errors
                print(f"  ⚠️  Cell {i}: {str(e)[:80]}")
    
    print(f"\nSummary: {executed} cells executed, {errors} errors")
    print(f"Figures generated: {len(figures_generated)}")
    
    return figures_generated

def main():
    """Generate figures for all tutorial notebooks."""
    notebooks = [
        'tutorials/basic_tabular.ipynb',
        'tutorials/multimodal_clusters.ipynb',
        'tutorials/adaptive_distances.ipynb',
        'tutorials/label_aware_classification.ipynb',
        'tutorials/high_dimensional.ipynb',
    ]
    
    all_figures = []
    
    for nb_path in notebooks:
        if not os.path.exists(nb_path):
            print(f"⚠️  {nb_path} not found, skipping")
            continue
        
        figures = execute_notebook(nb_path)
        all_figures.extend(figures)
    
    print(f"\n{'=' * 70}")
    print(f"TOTAL: {len(all_figures)} figures generated")
    print(f"{'=' * 70}")
    
    return all_figures

if __name__ == '__main__':
    main()

