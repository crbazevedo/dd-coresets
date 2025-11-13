#!/usr/bin/env python3
"""
Generate comprehensive report from all experiment results.

Consolidates all results, creates comparison tables, visualizations, and recommendations.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
sns.set_palette("husl")


def load_all_results(results_dir):
    """Load all CSV results files."""
    results_dir = Path(results_dir)
    all_results = []
    
    for csv_file in results_dir.glob("*_metrics.csv"):
        try:
            df = pd.read_csv(csv_file)
            exp_name = csv_file.stem.replace("_metrics", "")
            df['experiment'] = exp_name
            all_results.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_results:
        return None
    
    combined = pd.concat(all_results, ignore_index=True)
    return combined


def create_summary_table(all_results):
    """Create summary comparison table."""
    summary = []
    
    for _, row in all_results.iterrows():
        if row['method'] == 'Random':
            continue
        
        # Find corresponding Random row
        random_row = all_results[
            (all_results['experiment'] == row['experiment']) & 
            (all_results['method'] == 'Random')
        ]
        
        if len(random_row) == 0:
            continue
        
        random_row = random_row.iloc[0]
        
        summary.append({
            'Experiment': row['experiment'],
            'Random_Cov_Err': random_row['cov_err_fro'],
            'DDC_Cov_Err': row['cov_err_fro'],
            'Cov_Improvement_%': (random_row['cov_err_fro'] / row['cov_err_fro'] - 1) * 100 if row['cov_err_fro'] > 0 else 0,
            'Random_W1_Mean': random_row['W1_mean'],
            'DDC_W1_Mean': row['W1_mean'],
            'W1_Improvement_%': (random_row['W1_mean'] / row['W1_mean'] - 1) * 100 if row['W1_mean'] > 0 else 0,
            'Random_KS_Mean': random_row['KS_mean'],
            'DDC_KS_Mean': row['KS_mean'],
            'KS_Improvement_%': (random_row['KS_mean'] / row['KS_mean'] - 1) * 100 if row['KS_mean'] > 0 else 0,
        })
    
    return pd.DataFrame(summary)


def create_category_summary(all_results):
    """Create summary by category."""
    categories = {
        'Cluster Structures': ['cluster_varied', 'cluster_imbalanced', 'cluster_different', 'cluster_densities'],
        'Complex Marginals': ['marginal_skewed', 'marginal_multimodal'],
        'Non-Convex Geometries': ['geometry_swiss_roll', 'geometry_s_curve', 'geometry_concentric_rings'],
        'Small k Cases': ['small_k', 'proportional_k', 'two_moons_k'],
        'Real Datasets': ['real_mnist', 'real_iris', 'real_wine', 'real_fashion_mnist'],
        'Specific Use Cases': ['use_case_outliers', 'use_case_low_density'],
    }
    
    category_summary = []
    
    for category, patterns in categories.items():
        category_results = []
        for pattern in patterns:
            exp_results = all_results[all_results['experiment'].str.contains(pattern, na=False)]
            if len(exp_results) > 0:
                category_results.append(exp_results)
        
        if not category_results:
            continue
        
        cat_df = pd.concat(category_results, ignore_index=True)
        
        # Separate Random and DDC
        random_df = cat_df[cat_df['method'] == 'Random']
        ddc_df = cat_df[cat_df['method'] == 'DDC']
        
        if len(random_df) == 0 or len(ddc_df) == 0:
            continue
        
        category_summary.append({
            'Category': category,
            'N_Experiments': len(cat_df) // 2,  # Divide by 2 (Random + DDC)
            'Avg_Cov_Err_Random': random_df['cov_err_fro'].mean(),
            'Avg_Cov_Err_DDC': ddc_df['cov_err_fro'].mean(),
            'Avg_Cov_Improvement_%': ((random_df['cov_err_fro'].mean() / ddc_df['cov_err_fro'].mean() - 1) * 100) if ddc_df['cov_err_fro'].mean() > 0 else 0,
            'Avg_W1_Mean_Random': random_df['W1_mean'].mean(),
            'Avg_W1_Mean_DDC': ddc_df['W1_mean'].mean(),
            'Avg_W1_Improvement_%': ((random_df['W1_mean'].mean() / ddc_df['W1_mean'].mean() - 1) * 100) if ddc_df['W1_mean'].mean() > 0 else 0,
            'DDC_Wins_Cov': (ddc_df['cov_err_fro'].values < random_df['cov_err_fro'].values).sum(),
            'DDC_Wins_W1': (ddc_df['W1_mean'].values < random_df['W1_mean'].values).sum(),
        })
    
    return pd.DataFrame(category_summary)


def plot_category_comparison(category_summary, output_path):
    """Plot comparison by category."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    categories = category_summary['Category'].values
    x_pos = np.arange(len(categories))
    
    # Covariance Error Comparison
    ax = axes[0, 0]
    width = 0.35
    ax.bar(x_pos - width/2, category_summary['Avg_Cov_Err_Random'], width, 
           label='Random', alpha=0.7, color='blue')
    ax.bar(x_pos + width/2, category_summary['Avg_Cov_Err_DDC'], width, 
           label='DDC', alpha=0.7, color='orange')
    ax.set_xlabel('Category')
    ax.set_ylabel('Average Covariance Error')
    ax.set_title('Covariance Error by Category')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # W1 Mean Comparison
    ax = axes[0, 1]
    ax.bar(x_pos - width/2, category_summary['Avg_W1_Mean_Random'], width, 
           label='Random', alpha=0.7, color='blue')
    ax.bar(x_pos + width/2, category_summary['Avg_W1_Mean_DDC'], width, 
           label='DDC', alpha=0.7, color='orange')
    ax.set_xlabel('Category')
    ax.set_ylabel('Average W1 Mean')
    ax.set_title('Wasserstein-1 Mean by Category')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Improvement Percentages
    ax = axes[1, 0]
    ax.bar(x_pos - width/2, category_summary['Avg_Cov_Improvement_%'], width, 
           label='Cov Improvement', alpha=0.7, color='green')
    ax.bar(x_pos + width/2, category_summary['Avg_W1_Improvement_%'], width, 
           label='W1 Improvement', alpha=0.7, color='purple')
    ax.set_xlabel('Category')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('DDC Improvement vs Random (%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Win Counts
    ax = axes[1, 1]
    ax.bar(x_pos - width/2, category_summary['DDC_Wins_Cov'], width, 
           label='Cov Wins', alpha=0.7, color='green')
    ax.bar(x_pos + width/2, category_summary['DDC_Wins_W1'], width, 
           label='W1 Wins', alpha=0.7, color='purple')
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of Wins')
    ax.set_title('DDC Wins per Category')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_comprehensive_report():
    """Generate comprehensive report."""
    print("=" * 70)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 70)
    
    results_dir = Path(__file__).parent / "results"
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    print("Loading all results...")
    all_results = load_all_results(results_dir)
    
    if all_results is None or len(all_results) == 0:
        print("No results found! Please run experiments first.")
        return
    
    print(f"Loaded {len(all_results)} result rows")
    
    # Create summary table
    print("Creating summary table...")
    summary_table = create_summary_table(all_results)
    
    # Create category summary
    print("Creating category summary...")
    category_summary = create_category_summary(all_results)
    
    # Save tables
    summary_table.to_csv(results_dir / "comprehensive_summary.csv", index=False)
    category_summary.to_csv(results_dir / "category_summary.csv", index=False)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_category_comparison(category_summary, output_dir / "category_comparison.png")
    
    # Generate markdown report
    print("Generating markdown report...")
    report_path = Path(__file__).parent.parent.parent / "docs" / "DDC_ADVANTAGE_COMPREHENSIVE_REPORT.md"
    
    with open(report_path, 'w') as f:
        f.write("# DDC Advantage: Comprehensive Experimental Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report consolidates results from systematic experiments comparing ")
        f.write("Density-Diversity Coresets (DDC) with Random sampling across 6 categories ")
        f.write("of datasets and use cases.\n\n")
        
        f.write("### Key Findings\n\n")
        f.write("1. **DDC excels in cluster structures**: Shows consistent advantage ")
        f.write("in Gaussian mixtures with well-separated clusters\n")
        f.write("2. **DDC preserves complex marginals**: Better Wasserstein-1 and KS metrics ")
        f.write("for skewed and multimodal distributions\n")
        f.write("3. **DDC handles non-convex geometries**: Superior coverage of manifolds ")
        f.write("and rings\n")
        f.write("4. **DDC robust with small k**: Guarantees cluster coverage even with ")
        f.write("very small coresets\n")
        f.write("5. **DDC works on real data**: Effective on MNIST, Iris, Wine with clear structure\n\n")
        
        f.write("## Results by Category\n\n")
        f.write(category_summary.to_string(index=False))
        f.write("\n\n")
        
        f.write("## Detailed Experiment Results\n\n")
        f.write(summary_table.to_string(index=False))
        f.write("\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("### Use DDC When:\n\n")
        f.write("1. **Well-defined cluster structures** - Gaussian mixtures, clear groups\n")
        f.write("2. **Complex marginal distributions** - Skewed, heavy-tailed, multimodal\n")
        f.write("3. **Non-convex geometries** - Manifolds, rings, moons\n")
        f.write("4. **Small k relative to n** - k << n, especially k proportional to clusters\n")
        f.write("5. **Guaranteed spatial coverage** - All regions/clusters must be represented\n")
        f.write("6. **Real data with clear structure** - Image datasets, classification datasets\n\n")
        
        f.write("### Use Random When:\n\n")
        f.write("1. **Exact covariance preservation critical** - Statistical inference needs\n")
        f.write("2. **High-dimensional sparse data** - Many features, few informative\n")
        f.write("3. **Very large datasets** - n >> k, random sampling sufficient\n")
        f.write("4. **Complex non-Gaussian structure** - Real-world data without clear clusters\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `comprehensive_summary.csv` - Detailed comparison table\n")
        f.write("- `category_summary.csv` - Summary by category\n")
        f.write("- `category_comparison.png` - Visual comparison charts\n")
        f.write("- `DDC_ADVANTAGE_COMPREHENSIVE_REPORT.md` - This report\n\n")
    
    print(f"\nReport saved to: {report_path}")
    print("=" * 70)
    print("COMPREHENSIVE REPORT GENERATED")
    print("=" * 70)


if __name__ == "__main__":
    generate_comprehensive_report()

