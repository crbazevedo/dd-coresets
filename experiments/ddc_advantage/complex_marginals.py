"""
Categoria 2: Distribuições Marginais Complexas

Experimentos demonstrando vantagem do DDC em:
- Distribuições skewed/heavy-tailed
- Distribuições multimodais por feature
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.preprocessing import StandardScaler
from experiments.ddc_advantage.utils import (
    compute_all_metrics, fit_random_coreset, fit_ddc_coreset_optimized,
    plot_marginal_distributions, plot_metrics_comparison,
    save_results, RANDOM_STATE
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_skewed_distributions(n_samples, n_features, random_state=None):
    """Generate dataset with skewed/heavy-tailed marginal distributions."""
    if random_state is None:
        random_state = RANDOM_STATE
    
    rng = np.random.RandomState(random_state)
    X = np.zeros((n_samples, n_features))
    
    # Mix of different skewed distributions
    for i in range(n_features):
        if i % 4 == 0:
            # Log-normal (right-skewed)
            X[:, i] = stats.lognorm.rvs(s=1.0, scale=np.exp(0), size=n_samples, random_state=rng)
        elif i % 4 == 1:
            # Gamma (right-skewed)
            X[:, i] = stats.gamma.rvs(a=2.0, scale=2.0, size=n_samples, random_state=rng)
        elif i % 4 == 2:
            # Pareto (heavy-tailed)
            X[:, i] = stats.pareto.rvs(b=2.0, size=n_samples, random_state=rng)
        else:
            # Mixture of normals with different variances (bimodal)
            mix = rng.binomial(1, 0.5, n_samples)
            X[:, i] = mix * rng.normal(0, 1, n_samples) + (1 - mix) * rng.normal(5, 2, n_samples)
    
    return X


def generate_multimodal_per_feature(n_samples, n_features, n_modes_per_feature=3, random_state=None):
    """Generate dataset where each feature has multimodal distribution."""
    if random_state is None:
        random_state = RANDOM_STATE
    
    rng = np.random.RandomState(random_state)
    X = np.zeros((n_samples, n_features))
    
    for i in range(n_features):
        # Create multimodal distribution
        modes = []
        for mode_idx in range(n_modes_per_feature):
            center = (mode_idx - n_modes_per_feature / 2) * 3
            weight = 1.0 / n_modes_per_feature
            modes.append((center, weight))
        
        # Sample from mixture
        samples = []
        for _ in range(n_samples):
            mode_idx = rng.choice(n_modes_per_feature)
            center, _ = modes[mode_idx]
            samples.append(rng.normal(center, 0.8))
        
        X[:, i] = np.array(samples)
    
    return X


def compute_tail_metrics(X_full, S, w, quantiles=[0.05, 0.95]):
    """Compute metrics focusing on distribution tails."""
    tail_metrics = {}
    
    for q in quantiles:
        w1_tails = []
        for dim in range(X_full.shape[1]):
            # Sample from weighted coreset
            indices = np.random.choice(len(S), size=5000, p=w, replace=True)
            S_sample = S[indices, dim]
            
            # Compute quantiles
            q_full = np.quantile(X_full[:, dim], q)
            q_coreset = np.quantile(S_sample, q)
            
            w1_tails.append(abs(q_full - q_coreset))
        
        tail_metrics[f'quantile_{q}_error'] = np.mean(w1_tails)
    
    return tail_metrics


def run_experiment_2_1_skewed():
    """Experiment 2.1: Skewed and heavy-tailed distributions."""
    print("=" * 70)
    print("Experiment 2.1: Skewed/Heavy-tailed Distributions")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = 20_000
    n_features = 8
    k_reps = 1000
    
    # Generate data
    X = generate_skewed_distributions(n_samples, n_features, random_state=RANDOM_STATE)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit coresets
    S_random, w_random = fit_random_coreset(X_scaled, k_reps, random_state=RANDOM_STATE)
    S_ddc, w_ddc, _ = fit_ddc_coreset_optimized(X_scaled, k_reps, n0=None, 
                                                 random_state=RANDOM_STATE)
    
    # Compute metrics
    metrics_random = compute_all_metrics(X_scaled, S_random, w_random, 'Random')
    metrics_ddc = compute_all_metrics(X_scaled, S_ddc, w_ddc, 'DDC')
    
    # Compute tail metrics
    tail_random = compute_tail_metrics(X_scaled, S_random, w_random)
    tail_ddc = compute_tail_metrics(X_scaled, S_ddc, w_ddc)
    
    # Save results
    exp_name = "marginal_skewed"
    save_results(metrics_random, metrics_ddc, exp_name, results_dir)
    
    # Visualizations
    plot_marginal_distributions(
        X_scaled, S_random, w_random, S_ddc, w_ddc,
        n_features=min(8, n_features),
        title="Marginal Distributions: Skewed/Heavy-tailed",
        output_path=output_dir / f"{exp_name}_marginals.png"
    )
    
    plot_metrics_comparison(
        metrics_random, metrics_ddc,
        output_path=output_dir / f"{exp_name}_metrics.png"
    )
    
    print(f"\nTail Metrics (Quantile Errors):")
    print(f"  Random Q0.05: {tail_random['quantile_0.05_error']:.4f}")
    print(f"  DDC Q0.05: {tail_ddc['quantile_0.05_error']:.4f}")
    print(f"  Random Q0.95: {tail_random['quantile_0.95_error']:.4f}")
    print(f"  DDC Q0.95: {tail_ddc['quantile_0.95_error']:.4f}")
    print(f"\nDDC W1 Mean: {metrics_ddc['W1_mean']:.4f} "
          f"({(metrics_random['W1_mean'] / metrics_ddc['W1_mean'] - 1) * 100:+.1f}% vs Random)")
    print(f"DDC KS Mean: {metrics_ddc['KS_mean']:.4f} "
          f"({(metrics_random['KS_mean'] / metrics_ddc['KS_mean'] - 1) * 100:+.1f}% vs Random)")
    
    return metrics_random, metrics_ddc


def run_experiment_2_2_multimodal():
    """Experiment 2.2: Multimodal per-feature distributions."""
    print("\n" + "=" * 70)
    print("Experiment 2.2: Multimodal Per-Feature Distributions")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    results_dir = Path(__file__).parent / "results"
    
    n_samples = 20_000
    n_features = 8
    n_modes = 3
    k_reps = 1000
    
    # Generate data
    X = generate_multimodal_per_feature(n_samples, n_features, n_modes_per_feature=n_modes,
                                        random_state=RANDOM_STATE)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit coresets
    S_random, w_random = fit_random_coreset(X_scaled, k_reps, random_state=RANDOM_STATE)
    S_ddc, w_ddc, _ = fit_ddc_coreset_optimized(X_scaled, k_reps, n0=None, 
                                                 random_state=RANDOM_STATE)
    
    # Compute metrics
    metrics_random = compute_all_metrics(X_scaled, S_random, w_random, 'Random')
    metrics_ddc = compute_all_metrics(X_scaled, S_ddc, w_ddc, 'DDC')
    
    # Save results
    exp_name = "marginal_multimodal"
    save_results(metrics_random, metrics_ddc, exp_name, results_dir)
    
    # Visualizations
    plot_marginal_distributions(
        X_scaled, S_random, w_random, S_ddc, w_ddc,
        n_features=min(8, n_features),
        title=f"Marginal Distributions: {n_modes}-Modal Per Feature",
        output_path=output_dir / f"{exp_name}_marginals.png"
    )
    
    print(f"DDC W1 Mean: {metrics_ddc['W1_mean']:.4f} "
          f"({(metrics_random['W1_mean'] / metrics_ddc['W1_mean'] - 1) * 100:+.1f}% vs Random)")
    print(f"DDC KS Mean: {metrics_ddc['KS_mean']:.4f} "
          f"({(metrics_random['KS_mean'] / metrics_ddc['KS_mean'] - 1) * 100:+.1f}% vs Random)")
    
    return metrics_random, metrics_ddc


def main():
    """Run all complex marginal distribution experiments."""
    print("Running all Complex Marginal Distribution experiments...")
    print("=" * 70)
    
    results_2_1 = run_experiment_2_1_skewed()
    results_2_2 = run_experiment_2_2_multimodal()
    
    print("\n" + "=" * 70)
    print("All Complex Marginal Distribution experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

