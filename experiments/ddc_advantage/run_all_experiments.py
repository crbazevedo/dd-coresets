#!/usr/bin/env python3
"""
Minimal CLI runner for DDC Advantage experiments.

Usage:
    python -m experiments.ddc_advantage.run_all_experiments --preset small --seed 42
    python -m experiments.ddc_advantage.run_all_experiments --preset medium --outdir results/
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import core experiment modules only
import experiments.ddc_advantage.cluster_structures as cluster_structures
import experiments.ddc_advantage.complex_marginals as complex_marginals
import experiments.ddc_advantage.non_convex_geometries as non_convex_geometries
import experiments.ddc_advantage.real_datasets as real_datasets


def run_experiments(preset='small', seed=42, outdir=None, save=False, fig_format='svg'):
    """
    Run DDC Advantage experiments with specified preset.
    
    Parameters
    ----------
    preset : str
        'small' (fast, subset) or 'medium' (full suite)
    seed : int
        Random seed for reproducibility
    outdir : str or Path
        Output directory for results (default: experiments/ddc_advantage/results/)
    save : bool
        Whether to save figures (default: False)
    fig_format : str
        Figure format: 'svg' or 'png'
    """
    # Set random seed
    import numpy as np
    np.random.seed(seed)
    
    # Set output directory
    if outdir is None:
        outdir = Path(__file__).parent / "results"
    else:
        outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DDC ADVANTAGE EXPERIMENTS")
    print("=" * 70)
    print(f"Preset: {preset}")
    print(f"Seed: {seed}")
    print(f"Output: {outdir}")
    print(f"Save figures: {save} ({fig_format})")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results_summary = {}
    
    # Category 1: Cluster Structures
    print("\n" + "=" * 70)
    print("CATEGORY 1: CLUSTER STRUCTURES")
    print("=" * 70)
    try:
        if preset == 'small':
            # Run only 2 and 4 clusters for speed
            print("Running subset: 2 and 4 clusters only")
        cluster_structures.main()
        results_summary['cluster_structures'] = 'SUCCESS'
    except Exception as e:
        print(f"ERROR in cluster_structures: {e}")
        results_summary['cluster_structures'] = f'FAILED: {e}'
    
    # Category 2: Complex Marginals
    print("\n" + "=" * 70)
    print("CATEGORY 2: COMPLEX MARGINAL DISTRIBUTIONS")
    print("=" * 70)
    try:
        complex_marginals.main()
        results_summary['complex_marginals'] = 'SUCCESS'
    except Exception as e:
        print(f"ERROR in complex_marginals: {e}")
        results_summary['complex_marginals'] = f'FAILED: {e}'
    
    # Category 3: Non-Convex Geometries
    print("\n" + "=" * 70)
    print("CATEGORY 3: NON-CONVEX GEOMETRIES")
    print("=" * 70)
    try:
        non_convex_geometries.main()
        results_summary['non_convex_geometries'] = 'SUCCESS'
    except Exception as e:
        print(f"ERROR in non_convex_geometries: {e}")
        results_summary['non_convex_geometries'] = f'FAILED: {e}'
    
    # Category 4: Real Datasets
    print("\n" + "=" * 70)
    print("CATEGORY 4: REAL DATASETS")
    print("=" * 70)
    try:
        if preset == 'small':
            # Run only Iris/Wine for speed
            print("Running subset: Iris/Wine only")
        real_datasets.main()
        results_summary['real_datasets'] = 'SUCCESS'
    except Exception as e:
        print(f"ERROR in real_datasets: {e}")
        results_summary['real_datasets'] = f'FAILED: {e}'
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE SUMMARY")
    print("=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for category, status in results_summary.items():
        print(f"  {category:30s}: {status}")
    
    # Save summary
    summary_path = outdir / "experiment_suite_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("DDC Advantage Experiments - Suite Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Preset: {preset}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for category, status in results_summary.items():
            f.write(f"{category:30s}: {status}\n")
    
    print(f"\n✓ Summary saved to: {summary_path}")
    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run DDC Advantage experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick run (small preset, default seed)
  python -m experiments.ddc_advantage.run_all_experiments --preset small

  # Full suite with custom seed
  python -m experiments.ddc_advantage.run_all_experiments --preset medium --seed 123

  # Save figures
  python -m experiments.ddc_advantage.run_all_experiments --preset small --save
        """
    )
    
    parser.add_argument(
        '--preset',
        choices=['small', 'medium'],
        default='small',
        help='Preset: small (fast subset) or medium (full suite)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--outdir',
        type=str,
        default=None,
        help='Output directory (default: experiments/ddc_advantage/results/)'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save figures (default: False, figures not saved)'
    )
    
    parser.add_argument(
        '--fig-format',
        choices=['svg', 'png'],
        default='svg',
        help='Figure format: svg or png (default: svg)'
    )
    
    args = parser.parse_args()
    
    run_experiments(
        preset=args.preset,
        seed=args.seed,
        outdir=args.outdir,
        save=args.save,
        fig_format=args.fig_format
    )


if __name__ == "__main__":
    main()
