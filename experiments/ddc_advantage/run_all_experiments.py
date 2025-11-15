"""
Script unificado para executar todos os experimentos de vantagem do DDC.

Executa sistematicamente todas as categorias de experimentos e gera relatório consolidado.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import all experiment modules
import experiments.ddc_advantage.cluster_structures as cluster_structures
import experiments.ddc_advantage.complex_marginals as complex_marginals
import experiments.ddc_advantage.non_convex_geometries as non_convex_geometries
import experiments.ddc_advantage.small_k_cases as small_k_cases
import experiments.ddc_advantage.real_datasets as real_datasets
import experiments.ddc_advantage.specific_use_cases as specific_use_cases

# Import new advanced experiments
import experiments.ddc_advantage.nested_clusters as nested_clusters
import experiments.ddc_advantage.rare_clusters as rare_clusters
import experiments.ddc_advantage.multi_scale_clusters as multi_scale_clusters
import experiments.ddc_advantage.cifar10_experiment as cifar10_experiment
import experiments.ddc_advantage.varying_separability as varying_separability


def run_all_experiments():
    """Run all experiment categories."""
    print("=" * 70)
    print("DDC ADVANTAGE EXPERIMENTS - FULL SUITE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results_summary = {}
    
    # Category 1: Cluster Structures
    print("\n" + "=" * 70)
    print("CATEGORY 1: CLUSTER STRUCTURES")
    print("=" * 70)
    try:
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
    
    # Category 4: Small k Cases
    print("\n" + "=" * 70)
    print("CATEGORY 4: SMALL k CASES")
    print("=" * 70)
    try:
        small_k_cases.main()
        results_summary['small_k_cases'] = 'SUCCESS'
    except Exception as e:
        print(f"ERROR in small_k_cases: {e}")
        results_summary['small_k_cases'] = f'FAILED: {e}'
    
    # Category 5: Real Datasets
    print("\n" + "=" * 70)
    print("CATEGORY 5: REAL DATASETS")
    print("=" * 70)
    try:
        real_datasets.main()
        results_summary['real_datasets'] = 'SUCCESS'
    except Exception as e:
        print(f"ERROR in real_datasets: {e}")
        results_summary['real_datasets'] = f'FAILED: {e}'
    
    # Category 6: Specific Use Cases
    print("\n" + "=" * 70)
    print("CATEGORY 6: SPECIFIC USE CASES")
    print("=" * 70)
    try:
        specific_use_cases.main()
        results_summary['specific_use_cases'] = 'SUCCESS'
    except Exception as e:
        print(f"ERROR in specific_use_cases: {e}")
        results_summary['specific_use_cases'] = f'FAILED: {e}'
    
    # Category 7: Advanced Cluster Structures
    print("\n" + "=" * 70)
    print("CATEGORY 7: ADVANCED CLUSTER STRUCTURES")
    print("=" * 70)
    
    # 7.1 Nested Clusters
    print("\n--- 7.1: Nested Clusters ---")
    try:
        nested_clusters.run_experiment()
        results_summary['nested_clusters'] = 'SUCCESS'
    except Exception as e:
        print(f"ERROR in nested_clusters: {e}")
        results_summary['nested_clusters'] = f'FAILED: {e}'
    
    # 7.2 Rare Clusters
    print("\n--- 7.2: Rare Clusters ---")
    try:
        rare_clusters.run_experiment()
        results_summary['rare_clusters'] = 'SUCCESS'
    except Exception as e:
        print(f"ERROR in rare_clusters: {e}")
        results_summary['rare_clusters'] = f'FAILED: {e}'
    
    # 7.3 Multi-Scale Clusters
    print("\n--- 7.3: Multi-Scale Clusters ---")
    try:
        multi_scale_clusters.run_experiment()
        results_summary['multi_scale_clusters'] = 'SUCCESS'
    except Exception as e:
        print(f"ERROR in multi_scale_clusters: {e}")
        results_summary['multi_scale_clusters'] = f'FAILED: {e}'
    
    # 7.4 CIFAR-10
    print("\n--- 7.4: CIFAR-10 ---")
    try:
        cifar10_experiment.run_experiment()
        results_summary['cifar10'] = 'SUCCESS'
    except Exception as e:
        print(f"ERROR in cifar10_experiment: {e}")
        results_summary['cifar10'] = f'FAILED: {e}'
    
    # 7.5 Varying Separability
    print("\n--- 7.5: Varying Separability ---")
    try:
        varying_separability.run_experiment()
        results_summary['varying_separability'] = 'SUCCESS'
    except Exception as e:
        print(f"ERROR in varying_separability: {e}")
        results_summary['varying_separability'] = f'FAILED: {e}'
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE SUMMARY")
    print("=" * 70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for category, status in results_summary.items():
        print(f"  {category:30s}: {status}")
    
    # Save summary
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = results_dir / "experiment_suite_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("DDC Advantage Experiments - Suite Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for category, status in results_summary.items():
            f.write(f"{category:30s}: {status}\n")
    
    print(f"\nSummary saved to: {summary_path}")
    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()

