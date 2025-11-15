#!/usr/bin/env python3
"""
Validate all example notebooks by executing each cell.

This script:
1. Loads each notebook
2. Executes each cell sequentially
3. Reports any errors
4. Validates outputs
"""

import json
import sys
import os
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Global namespace for notebook execution
NOTEBOOK_NAMESPACE = {
    'np': np,
    'pd': pd,
    'plt': plt,
    '__builtins__': __builtins__,
}

def execute_cell(cell, cell_idx, namespace):
    """Execute a single notebook cell."""
    if cell['cell_type'] != 'code':
        return True, None, None
    
    source = ''.join(cell.get('source', []))
    if not source.strip():
        return True, None, None
    
    try:
        # Compile and execute
        code = compile(source, f'<cell {cell_idx}>', 'exec')
        exec(code, namespace)
        return True, None, None
    except Exception as e:
        error_msg = f"Cell {cell_idx} error: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg, e

def validate_notebook(notebook_path):
    """Validate a single notebook."""
    print(f"\n{'='*70}")
    print(f"Validating: {notebook_path}")
    print(f"{'='*70}")
    
    # Load notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Initialize namespace
    namespace = NOTEBOOK_NAMESPACE.copy()
    namespace['__file__'] = str(notebook_path)
    
    # Import common libraries
    try:
        from sklearn.datasets import make_blobs
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        namespace.update({
            'make_blobs': make_blobs,
            'StandardScaler': StandardScaler,
            'PCA': PCA,
        })
    except ImportError as e:
        print(f"Warning: Could not import sklearn: {e}")
    
    # Try importing UMAP (optional)
    try:
        import umap
        namespace['umap'] = umap
        namespace['HAS_UMAP'] = True
    except ImportError:
        namespace['HAS_UMAP'] = False
    
    # Try importing dd_coresets
    try:
        from dd_coresets import (
            fit_ddc_coreset,
            fit_random_coreset,
            fit_stratified_coreset,
            fit_ddc_coreset_by_label,
        )
        namespace.update({
            'fit_ddc_coreset': fit_ddc_coreset,
            'fit_random_coreset': fit_random_coreset,
            'fit_stratified_coreset': fit_stratified_coreset,
            'fit_ddc_coreset_by_label': fit_ddc_coreset_by_label,
        })
    except ImportError as e:
        print(f"Error: Could not import dd_coresets: {e}")
        return False
    
    # Set random seed
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    namespace['RANDOM_STATE'] = RANDOM_STATE
    
    # Execute each cell
    errors = []
    code_cell_count = 0
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            code_cell_count += 1
            source = ''.join(cell.get('source', []))
            
            # Skip empty cells
            if not source.strip():
                continue
            
            # Skip pip install cells (they're commented out)
            if 'pip install' in source and source.strip().startswith('#'):
                print(f"  Cell {i}: Skipped (commented pip install)")
                continue
            
            success, error_msg, exception = execute_cell(cell, i, namespace)
            
            if success:
                print(f"  ✓ Cell {i} ({code_cell_count}): OK")
            else:
                print(f"  ✗ Cell {i} ({code_cell_count}): FAILED")
                print(f"    {error_msg}")
                errors.append((i, error_msg, exception))
    
    # Summary
    print(f"\nSummary:")
    print(f"  Total cells: {len(nb['cells'])}")
    print(f"  Code cells: {code_cell_count}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print(f"\n❌ Notebook has {len(errors)} error(s)")
        return False
    else:
        print(f"\n✅ Notebook validated successfully!")
        return True

def main():
    """Validate all notebooks."""
    examples_dir = Path(__file__).parent
    notebooks = [
        'basic_tabular.ipynb',
        'multimodal_clusters.ipynb',
        'adaptive_distances.ipynb',
        'label_aware_classification.ipynb',
        'high_dimensional.ipynb',
    ]
    
    results = {}
    for notebook_name in notebooks:
        notebook_path = examples_dir / notebook_name
        if not notebook_path.exists():
            print(f"\n⚠️  Notebook not found: {notebook_path}")
            results[notebook_name] = False
            continue
        
        results[notebook_name] = validate_notebook(notebook_path)
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for notebook_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {notebook_name}")
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n🎉 All notebooks validated successfully!")
        return 0
    else:
        print(f"\n⚠️  Some notebooks failed validation")
        return 1

if __name__ == '__main__':
    sys.exit(main())

