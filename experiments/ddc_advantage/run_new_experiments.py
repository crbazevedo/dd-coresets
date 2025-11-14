#!/usr/bin/env python3
"""
Run all new experiments from the proposal.

Executes:
- nested_clusters.py
- rare_clusters.py
- multi_scale_clusters.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import nested_clusters
import rare_clusters
import multi_scale_clusters


def main():
    """Run all new experiments."""
    print("=" * 70)
    print("RUNNING ALL NEW EXPERIMENTS")
    print("=" * 70)
    
    all_results = []
    
    # Run each experiment
    print("\n" + "=" * 70)
    print("1. Nested Clusters")
    print("=" * 70)
    try:
        results = nested_clusters.run_experiment()
        all_results.extend(results)
    except Exception as e:
        print(f"Error in nested_clusters: {e}")
    
    print("\n" + "=" * 70)
    print("2. Rare Clusters")
    print("=" * 70)
    try:
        results = rare_clusters.run_experiment()
        all_results.extend(results)
    except Exception as e:
        print(f"Error in rare_clusters: {e}")
    
    print("\n" + "=" * 70)
    print("3. Multi-Scale Clusters")
    print("=" * 70)
    try:
        results = multi_scale_clusters.run_experiment()
        all_results.extend(results)
    except Exception as e:
        print(f"Error in multi_scale_clusters: {e}")
    
    print("\n" + "=" * 70)
    print("ALL NEW EXPERIMENTS COMPLETED")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    main()

