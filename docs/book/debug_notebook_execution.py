#!/usr/bin/env python3
"""
Systematic debugging of notebook execution errors.

This script tests hypotheses about why notebooks fail to execute.
"""

import sys
import os
import json
import numpy as np
import traceback

def test_hypothesis_1_version_mismatch():
    """Test if installed version supports mode parameter."""
    print("=" * 70)
    print("HYPOTHESIS 1: Version Mismatch")
    print("=" * 70)
    
    try:
        from dd_coresets.ddc import fit_ddc_coreset
        import inspect
        
        sig = inspect.signature(fit_ddc_coreset)
        has_mode = 'mode' in str(sig)
        
        if has_mode:
            print("✅ REJECTED: API supports mode parameter")
            return False, None
        else:
            print("✅ CONFIRMED: API does NOT support mode parameter")
            return True, "Remove mode parameter from notebooks"
    except Exception as e:
        print(f"❌ Error testing: {e}")
        return None, str(e)

def test_hypothesis_2_actual_call():
    """Test if actual function call works."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Actual Function Call")
    print("=" * 70)
    
    try:
        from dd_coresets.ddc import fit_ddc_coreset
        
        X = np.random.randn(100, 5)
        
        # Test with mode
        try:
            S, w, info = fit_ddc_coreset(X, k=10, mode='euclidean', random_state=42)
            print("✅ REJECTED: Call with mode='euclidean' works")
            return False, None
        except TypeError as e:
            print(f"✅ CONFIRMED: Call fails with TypeError: {e}")
            
            # Test without mode
            try:
                S, w, info = fit_ddc_coreset(X, k=10, random_state=42)
                print("   → Works without mode parameter")
                return True, "Remove mode parameter from notebooks"
            except Exception as e2:
                print(f"   → Still fails: {e2}")
                return True, f"Different issue: {e2}"
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, str(e)

def test_hypothesis_3_notebook_context():
    """Test execution in notebook-like context."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: Notebook Execution Context")
    print("=" * 70)
    
    # Change to notebook directory
    original_dir = os.getcwd()
    notebook_dir = os.path.join(original_dir, 'tutorials')
    
    if not os.path.exists(notebook_dir):
        print("⚠️  Notebook directory not found")
        return None, "Directory issue"
    
    os.chdir(notebook_dir)
    
    try:
        # Add parent to path
        sys.path.insert(0, original_dir)
        sys.path.insert(0, os.path.dirname(original_dir))
        
        from dd_coresets.ddc import fit_ddc_coreset
        
        X = np.random.randn(100, 5)
        
        try:
            S, w, info = fit_ddc_coreset(X, k=10, mode='euclidean', random_state=42)
            print("✅ REJECTED: Works in notebook context")
            os.chdir(original_dir)
            return False, None
        except Exception as e:
            print(f"✅ CONFIRMED: Fails in notebook context: {e}")
            os.chdir(original_dir)
            return True, str(e)
    except Exception as e:
        os.chdir(original_dir)
        print(f"❌ Error: {e}")
        return None, str(e)

def test_hypothesis_4_full_notebook_simulation():
    """Simulate full notebook execution."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: Full Notebook Simulation")
    print("=" * 70)
    
    try:
        # Setup like notebook
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.datasets import make_blobs
        from sklearn.preprocessing import StandardScaler
        from dd_coresets import fit_ddc_coreset, fit_random_coreset, fit_stratified_coreset
        
        RANDOM_STATE = 42
        np.random.seed(RANDOM_STATE)
        
        # Generate data
        n_samples = 10000
        n_features = 8
        n_clusters = 3
        
        X, cluster_labels = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=1.5,
            center_box=(-10, 10),
            random_state=RANDOM_STATE,
        )
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Test the exact call from notebook
        k = 200
        print(f"Testing: fit_ddc_coreset(X, k={k}, mode='euclidean', random_state={RANDOM_STATE})")
        
        try:
            S_ddc, w_ddc, info_ddc = fit_ddc_coreset(
                X, k=k, mode='euclidean', random_state=RANDOM_STATE
            )
            print("✅ REJECTED: Full simulation works")
            print(f"   Generated: {S_ddc.shape}")
            return False, None
        except TypeError as e:
            print(f"✅ CONFIRMED: Full simulation fails: {e}")
            return True, "Remove mode parameter"
        except Exception as e:
            print(f"✅ CONFIRMED: Different error: {e}")
            return True, str(e)
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        traceback.print_exc()
        return None, str(e)

def main():
    """Run all hypothesis tests."""
    print("\n" + "=" * 70)
    print("NOTEBOOK EXECUTION DEBUG - SYSTEMATIC HYPOTHESIS TESTING")
    print("=" * 70)
    
    results = []
    
    # Test each hypothesis
    results.append(("H1: Version Mismatch", test_hypothesis_1_version_mismatch()))
    results.append(("H2: Actual Call", test_hypothesis_2_actual_call()))
    results.append(("H3: Notebook Context", test_hypothesis_3_notebook_context()))
    results.append(("H4: Full Simulation", test_hypothesis_4_full_notebook_simulation()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    confirmed = []
    rejected = []
    
    for name, (confirmed_flag, fix) in results:
        if confirmed_flag is True:
            confirmed.append((name, fix))
        elif confirmed_flag is False:
            rejected.append(name)
    
    if confirmed:
        print("\n✅ CONFIRMED HYPOTHESES:")
        for name, fix in confirmed:
            print(f"   {name}")
            print(f"      Fix: {fix}")
    
    if rejected:
        print("\n❌ REJECTED HYPOTHESES:")
        for name in rejected:
            print(f"   {name}")
    
    # Recommended fix
    if confirmed:
        print("\n" + "=" * 70)
        print("RECOMMENDED FIX")
        print("=" * 70)
        print(confirmed[0][1])  # Use first confirmed hypothesis fix
    
    return confirmed, rejected

if __name__ == '__main__':
    main()

