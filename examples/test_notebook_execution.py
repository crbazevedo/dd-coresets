#!/usr/bin/env python3
"""
Script to execute and validate the binary_classification_ddc.ipynb notebook
cell by cell, checking results at each step.
"""

import json
import sys
import os
import traceback

# Add parent directory to path to import dd_coresets
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

# Optional imports for visualization (may not be available in test environment)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    # Create mock objects if matplotlib not available
    class MockPlot:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    plt = MockPlot()
    sns = MockPlot()
    HAS_MATPLOTLIB = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, accuracy_score
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

from scipy.stats import wasserstein_distance, ks_2samp

from dd_coresets import fit_ddc_coreset

def execute_notebook():
    """Execute notebook cells sequentially and validate results."""
    
    # Load notebook
    with open('binary_classification_ddc.ipynb', 'r') as f:
        nb = json.load(f)
    
    # Execution environment (shared state across cells)
    env = {
        '__name__': '__main__',
        '__builtins__': __builtins__,
        'np': np,
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'train_test_split': train_test_split,
        'StandardScaler': StandardScaler,
        'LogisticRegression': LogisticRegression,
        'roc_auc_score': roc_auc_score,
        'roc_curve': roc_curve,
        'brier_score_loss': brier_score_loss,
        'accuracy_score': accuracy_score,
        'PCA': PCA,
        'make_classification': make_classification,
        'wasserstein_distance': wasserstein_distance,
        'ks_2samp': ks_2samp,
        'fit_ddc_coreset': fit_ddc_coreset,
    }
    
    results = {
        'executed': 0,
        'skipped': 0,
        'errors': [],
        'warnings': [],
        'checkpoints': {}
    }
    
    print("=" * 70)
    print("EXECUTANDO NOTEBOOK CÉLULA POR CÉLULA")
    print("=" * 70)
    print()
    
    for i, cell in enumerate(nb['cells']):
        cell_type = cell['cell_type']
        source = ''.join(cell['source']) if 'source' in cell else ''
        
        # Skip markdown cells (just for documentation)
        if cell_type == 'markdown':
            # Extract section headers for context
            if source.startswith('##'):
                section = source.split('\n')[0].replace('#', '').strip()
                print(f"\n{'='*70}")
                print(f"📝 {section}")
                print('='*70)
            continue
        
        # Skip installation cells
        if 'pip install' in source or '!pip' in source:
            print(f"⏭️  Cell {i}: Skipping installation cell")
            results['skipped'] += 1
            continue
        
            # Skip pure display cells
        if source.strip() in ['df.head()', 'plt.show()']:
            print(f"⏭️  Cell {i}: Skipping display cell")
            results['skipped'] += 1
            continue
        
        # Skip visualization cells if matplotlib not available
        if not HAS_MATPLOTLIB and ('plt.' in source or 'matplotlib' in source or 'seaborn' in source):
            if 'import' not in source:  # Don't skip imports, just visualization code
                print(f"⏭️  Cell {i}: Skipping visualization (matplotlib not available)")
                results['skipped'] += 1
                continue
        
        # Execute code cell
        print(f"\n🔧 Cell {i}: Executing...")
        try:
            # Handle imports cell specially - allow partial failures
            if 'import' in source and ('matplotlib' in source or 'seaborn' in source):
                # Try to execute, but catch import errors gracefully
                try:
                    exec(source, env)
                except ImportError as e:
                    if 'matplotlib' in str(e) or 'seaborn' in str(e):
                        print(f"   ⚠️  Matplotlib/seaborn not available, using mocks")
                        # Set mocks in environment
                        if 'plt' not in env:
                            env['plt'] = plt
                        if 'sns' not in env:
                            env['sns'] = sns
                        # Try to execute rest of imports
                        lines = source.split('\n')
                        for line in lines:
                            if 'matplotlib' not in line and 'seaborn' not in line:
                                try:
                                    exec(line, env)
                                except:
                                    pass
                        # Set RANDOM_STATE if it's in the cell
                        if 'RANDOM_STATE' in source:
                            exec("RANDOM_STATE = 42\nnp.random.seed(RANDOM_STATE)", env)
                        print(f"   ✅ Imports handled (with mocks for visualization)")
                        results['executed'] += 1
                        continue
                    else:
                        raise
            
            # Execute in shared environment
            exec(source, env)
            results['executed'] += 1
            
            # Check for expected variables after key cells
            if i == 3:  # After imports
                assert 'RANDOM_STATE' in env, "RANDOM_STATE not set"
                print(f"   ✅ Imports successful, RANDOM_STATE = {env['RANDOM_STATE']}")
            
            elif 'USE_SYNTHETIC' in source and 'df' in source:
                if 'df' in env:
                    df = env['df']
                    print(f"   ✅ Dataset loaded: {df.shape}")
                    print(f"      Label distribution: {df['target'].value_counts(normalize=True).to_dict()}")
                    results['checkpoints']['data_loaded'] = True
            
            elif 'X_train' in source and 'train_test_split' in source:
                if 'X_train' in env and 'y_train' in env:
                    X_train = env['X_train']
                    y_train = env['y_train']
                    print(f"   ✅ Train/test split: train={X_train.shape[0]:,}, test={env['X_test'].shape[0]:,}")
                    print(f"      Train class distribution: {np.bincount(y_train) / len(y_train)}")
                    results['checkpoints']['preprocessing'] = True
            
            elif 'baseline_auc' in source or 'lr_full' in source:
                if 'baseline_auc' in env:
                    print(f"   ✅ Baseline model: AUC={env['baseline_auc']:.4f}, "
                          f"Brier={env['baseline_brier']:.4f}")
                    results['checkpoints']['baseline'] = True
            
            elif 'k_reps' in source and '=' in source:
                if 'k_reps' in env:
                    print(f"   ✅ k_reps set to: {env['k_reps']}")
            
            elif 'X_random' in source or ('random_indices' in source and 'X_random' in env):
                if 'X_random' in env:
                    print(f"   ✅ Random subset: {env['X_random'].shape}, "
                          f"class dist: {np.bincount(env['y_random']) / len(env['y_random'])}")
                    results['checkpoints']['random_subset'] = True
            
            elif 'X_strat' in source or ('strat_indices' in source and 'X_strat' in env):
                if 'X_strat' in env:
                    print(f"   ✅ Stratified subset: {env['X_strat'].shape}, "
                          f"class dist: {np.bincount(env['y_strat']) / len(env['y_strat'])}")
                    results['checkpoints']['stratified_subset'] = True
            
            elif 'S_global' in source or ('fit_ddc_coreset' in source and 'S_global' in env):
                if 'S_global' in env and 'w_global' in env:
                    S_global = env['S_global']
                    w_global = env['w_global']
                    y_global = env.get('y_global', None)
                    print(f"   ✅ Global DDC coreset: {S_global.shape}")
                    print(f"      Weights sum: {w_global.sum():.6f}")
                    if y_global is not None:
                        print(f"      Class distribution: {np.bincount(y_global) / len(y_global)}")
                        orig_dist = np.bincount(env['y_train']) / len(env['y_train'])
                        print(f"      Original: {orig_dist}")
                        # Check if proportions changed
                        shift = np.abs((np.bincount(y_global) / len(y_global)) - orig_dist)
                        if np.any(shift > 0.05):
                            print(f"      ⚠️  Class proportion shift detected (expected for global DDC)")
                    results['checkpoints']['global_ddc'] = True
            
            elif 'S_labelwise' in source or ('vstack' in source and 'S_labelwise' in env):
                if 'S_labelwise' in env and 'w_labelwise' in env:
                    S_labelwise = env['S_labelwise']
                    w_labelwise = env['w_labelwise']
                    y_labelwise = env.get('y_labelwise', None)
                    print(f"   ✅ Label-wise DDC coreset: {S_labelwise.shape}")
                    print(f"      Weights sum: {w_labelwise.sum():.6f}")
                    if y_labelwise is not None:
                        print(f"      Class distribution: {np.bincount(y_labelwise) / len(y_labelwise)}")
                        orig_dist = np.bincount(env['y_train']) / len(env['y_train'])
                        print(f"      Original: {orig_dist}")
                        # Check if proportions preserved
                        diff = np.abs((np.bincount(y_labelwise) / len(y_labelwise)) - orig_dist)
                        if np.all(diff < 0.01):
                            print(f"      ✅ Class proportions preserved (within 1%)")
                        else:
                            print(f"      ⚠️  Class proportion difference: {diff}")
                    results['checkpoints']['labelwise_ddc'] = True
            
            elif 'dist_results_df' in source or 'results.append' in source:
                if 'dist_results_df' in env:
                    df = env['dist_results_df']
                    print(f"   ✅ Distribution comparison computed for {len(df)} features")
                    results['checkpoints']['distribution_comparison'] = True
            
            elif 'comparison_df' in source or ('model_results' in source and 'comparison_df' in env):
                if 'comparison_df' in env:
                    df = env['comparison_df']
                    print(f"   ✅ Model comparison computed: {len(df)} methods")
                    if 'baseline_auc' in env:
                        print(f"      Baseline AUC: {env['baseline_auc']:.4f}")
                        for _, row in df.iterrows():
                            if row['method'] != 'Full Data':
                                diff = row['auc'] - env['baseline_auc']
                                print(f"      {row['method']:15s}: AUC={row['auc']:.4f} ({diff:+.4f})")
                    results['checkpoints']['model_comparison'] = True
            
            # Progress indicator
            if results['executed'] % 3 == 0:
                print(f"   📊 Progress: {results['executed']} cells executed")
                
        except Exception as e:
            error_msg = f"Cell {i}: {type(e).__name__}: {str(e)[:150]}"
            results['errors'].append((i, error_msg, traceback.format_exc()))
            print(f"   ❌ ERROR: {error_msg}")
            # Don't stop - continue to find all issues
            if len(results['errors']) <= 3:
                print(f"   Traceback:\n{traceback.format_exc()[:300]}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("RESUMO DA EXECUÇÃO")
    print("=" * 70)
    print(f"✅ Células executadas: {results['executed']}")
    print(f"⏭️  Células puladas: {results['skipped']}")
    print(f"❌ Erros: {len(results['errors'])}")
    
    print(f"\n📋 Checkpoints alcançados:")
    checkpoints = [
        ('data_loaded', 'Data Loading'),
        ('preprocessing', 'Preprocessing'),
        ('baseline', 'Baseline Model'),
        ('random_subset', 'Random Subset'),
        ('stratified_subset', 'Stratified Subset'),
        ('global_ddc', 'Global DDC'),
        ('labelwise_ddc', 'Label-wise DDC'),
        ('distribution_comparison', 'Distribution Comparison'),
        ('model_comparison', 'Model Comparison'),
    ]
    
    for key, name in checkpoints:
        status = "✅" if results['checkpoints'].get(key, False) else "❌"
        print(f"   {status} {name}")
    
    if results['errors']:
        print(f"\n⚠️  Erros encontrados:")
        for cell_idx, error_msg, _ in results['errors'][:5]:
            print(f"   Cell {cell_idx}: {error_msg[:100]}")
    
    # Final validation
    print(f"\n🔍 Validação Final:")
    
    # Check key variables exist
    key_vars = {
        'X_train': 'Training features',
        'y_train': 'Training labels',
        'X_test': 'Test features',
        'y_test': 'Test labels',
        'S_global': 'Global DDC coreset',
        'w_global': 'Global DDC weights',
        'S_labelwise': 'Label-wise DDC coreset',
        'w_labelwise': 'Label-wise DDC weights',
        'baseline_auc': 'Baseline AUC',
        'comparison_df': 'Model comparison table',
    }
    
    all_present = True
    for var, desc in key_vars.items():
        exists = var in env
        status = "✅" if exists else "❌"
        print(f"   {status} {desc}: {var}")
        if not exists:
            all_present = False
    
    if all_present and len(results['errors']) == 0:
        print(f"\n✅ NOTEBOOK EXECUTADO COM SUCESSO!")
        print(f"   Todas as variáveis-chave foram criadas")
        print(f"   Nenhum erro encontrado")
        return 0
    else:
        print(f"\n⚠️  NOTEBOOK EXECUTADO COM AVISOS")
        if not all_present:
            print(f"   Algumas variáveis-chave não foram criadas")
        if results['errors']:
            print(f"   {len(results['errors'])} erros encontrados")
        return 1

if __name__ == '__main__':
    sys.exit(execute_notebook())

