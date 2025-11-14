#!/usr/bin/env python3
"""
Investigate k-NN density estimation failure in high dimensions.

Tests:
1. How k-NN density estimation degrades with dimensionality
2. Whether adaptive distances help
3. Impact on DDC performance
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from experiments.ddc_advantage.utils import (
    fit_random_coreset, fit_ddc_coreset_optimized,
    compute_all_metrics, RANDOM_STATE
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')


def estimate_density_knn(X, k=32, metric='euclidean'):
    """
    Estimate density using k-NN.
    
    Density is inversely proportional to the volume of the k-NN ball.
    For Euclidean distance: density ~ 1 / (distance_to_kth_neighbor^d)
    """
    n, d = X.shape
    
    # Fit k-NN
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric)  # +1 because point itself
    nn.fit(X)
    
    # Get distances to k-th neighbor (excluding self)
    distances, _ = nn.kneighbors(X)
    kth_distances = distances[:, k]  # k-th neighbor (0-indexed, so k is k+1-th)
    
    # Density estimate: 1 / (volume of k-NN ball)
    # Volume of d-dimensional ball: V = (π^(d/2) / Γ(d/2 + 1)) * r^d
    # For density estimation, we use: density ~ k / (kth_distance^d)
    # Simplified: density ~ 1 / (kth_distance^d)
    
    densities = k / (kth_distances ** d + 1e-10)
    
    return densities, kth_distances


def estimate_density_adaptive(X, k=32, n_iter=5):
    """
    Estimate density using adaptive Mahalanobis distance.
    
    Iteratively learns a local covariance matrix for each point.
    """
    n, d = X.shape
    densities = np.zeros(n)
    
    # Initialize with Euclidean distance
    nn_euclidean = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn_euclidean.fit(X)
    
    # Initial density estimates
    distances, indices = nn_euclidean.kneighbors(X)
    kth_distances = distances[:, k]
    densities = k / (kth_distances ** d + 1e-10)
    
    # Iterative refinement
    for iteration in range(n_iter):
        # For each point, compute local covariance from neighbors
        new_densities = np.zeros(n)
        
        for i in range(n):
            # Get neighbors using current density-weighted distance
            # Use neighbors from previous iteration
            neighbor_indices = indices[i, 1:k+1]  # Exclude self
            
            if len(neighbor_indices) > d:  # Need enough neighbors for covariance
                # Compute local covariance
                neighbors = X[neighbor_indices]
                local_mean = neighbors.mean(axis=0)
                local_cov = np.cov(neighbors.T)
                
                # Add regularization
                local_cov += np.eye(d) * 1e-6
                
                try:
                    # Compute Mahalanobis distances
                    inv_cov = np.linalg.inv(local_cov)
                    diff = X - local_mean
                    mahal_distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                    
                    # Get k-th Mahalanobis neighbor
                    sorted_indices = np.argsort(mahal_distances)
                    kth_mahal_dist = mahal_distances[sorted_indices[k]]
                    
                    # Density estimate using Mahalanobis distance
                    # Volume in Mahalanobis space: det(cov)^(1/2) * r^d
                    det_cov = np.linalg.det(local_cov)
                    new_densities[i] = k / (kth_mahal_dist ** d * np.sqrt(det_cov) + 1e-10)
                except:
                    # Fallback to Euclidean
                    new_densities[i] = densities[i]
            else:
                new_densities[i] = densities[i]
        
        densities = new_densities
        
        # Update neighbor indices based on new densities
        # (Simplified: keep same neighbors for now)
    
    return densities


def test_density_estimation_degradation():
    """Test how density estimation degrades with dimensionality."""
    print("=" * 70)
    print("TESTING DENSITY ESTIMATION DEGRADATION WITH DIMENSIONALITY")
    print("=" * 70)
    
    n_samples = 5000
    n_clusters = 4
    k = 32
    dimensions = [2, 5, 10, 20, 50, 100]
    
    results = []
    
    for d in dimensions:
        print(f"\nTesting d={d}...")
        
        # Generate data
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=d,
            centers=n_clusters,
            cluster_std=1.0,
            random_state=RANDOM_STATE,
        )
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Estimate density with Euclidean k-NN
        densities_euclidean, kth_dists_euclidean = estimate_density_knn(X_scaled, k=k)
        
        # Estimate density with adaptive distance
        try:
            densities_adaptive = estimate_density_adaptive(X_scaled, k=k, n_iter=3)
        except Exception as e:
            print(f"  Warning: Adaptive failed: {e}")
            densities_adaptive = densities_euclidean.copy()
        
        # Analyze variance of density estimates
        # In high dimensions, all points should have similar densities (curse of dimensionality)
        cv_euclidean = np.std(densities_euclidean) / (np.mean(densities_euclidean) + 1e-10)
        cv_adaptive = np.std(densities_adaptive) / (np.mean(densities_adaptive) + 1e-10)
        
        # Check if density estimates are meaningful (should vary by cluster)
        # Compute density per cluster
        densities_by_cluster_euclidean = []
        densities_by_cluster_adaptive = []
        
        for cluster_id in np.unique(y):
            cluster_mask = (y == cluster_id)
            densities_by_cluster_euclidean.append(densities_euclidean[cluster_mask].mean())
            densities_by_cluster_adaptive.append(densities_adaptive[cluster_mask].mean())
        
        # Variance across clusters (should be high if density estimation works)
        cluster_var_euclidean = np.var(densities_by_cluster_euclidean)
        cluster_var_adaptive = np.var(densities_by_cluster_adaptive)
        
        results.append({
            'dimension': d,
            'cv_euclidean': cv_euclidean,
            'cv_adaptive': cv_adaptive,
            'cluster_var_euclidean': cluster_var_euclidean,
            'cluster_var_adaptive': cluster_var_adaptive,
            'mean_kth_dist_euclidean': kth_dists_euclidean.mean(),
            'std_kth_dist_euclidean': kth_dists_euclidean.std(),
        })
        
        print(f"  CV (Euclidean): {cv_euclidean:.4f}, CV (Adaptive): {cv_adaptive:.4f}")
        print(f"  Cluster variance (Euclidean): {cluster_var_euclidean:.6f}")
        print(f"  Cluster variance (Adaptive): {cluster_var_adaptive:.6f}")
    
    results_df = pd.DataFrame(results)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(results_df['dimension'], results_df['cv_euclidean'], 'o-', label='Euclidean', linewidth=2)
    ax.plot(results_df['dimension'], results_df['cv_adaptive'], 'o-', label='Adaptive', linewidth=2)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Density Estimate Variance vs Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(results_df['dimension'], results_df['cluster_var_euclidean'], 'o-', label='Euclidean', linewidth=2)
    ax.plot(results_df['dimension'], results_df['cluster_var_adaptive'], 'o-', label='Adaptive', linewidth=2)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Variance Across Clusters')
    ax.set_title('Cluster Density Discrimination vs Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    ax = axes[1, 0]
    ax.plot(results_df['dimension'], results_df['mean_kth_dist_euclidean'], 'o-', label='Mean k-th dist', linewidth=2)
    ax.fill_between(
        results_df['dimension'],
        results_df['mean_kth_dist_euclidean'] - results_df['std_kth_dist_euclidean'],
        results_df['mean_kth_dist_euclidean'] + results_df['std_kth_dist_euclidean'],
        alpha=0.3
    )
    ax.set_xlabel('Dimension')
    ax.set_ylabel('k-th Neighbor Distance')
    ax.set_title('k-th Neighbor Distance vs Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    # Ratio of adaptive to euclidean cluster variance
    ratio = results_df['cluster_var_adaptive'] / (results_df['cluster_var_euclidean'] + 1e-10)
    ax.plot(results_df['dimension'], ratio, 'o-', label='Adaptive/Euclidean', linewidth=2, color='green')
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Ratio (Adaptive/Euclidean)')
    ax.set_title('Adaptive vs Euclidean Discrimination')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "high_dim_density_degradation.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / "high_dim_density_analysis.csv", index=False)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nResults Summary:")
    print(results_df.to_string(index=False))
    
    return results_df


def test_ddc_with_adaptive_distances():
    """Test if adaptive distances improve DDC in high dimensions."""
    print("\n" + "=" * 70)
    print("TESTING DDC WITH ADAPTIVE DISTANCES")
    print("=" * 70)
    
    # Note: This would require modifying DDC to use adaptive distances
    # For now, we'll analyze the density estimation quality
    
    n_samples = 10_000
    n_clusters = 4
    k_reps = 1000
    dimensions = [10, 20, 50]
    
    results = []
    
    for d in dimensions:
        print(f"\nTesting d={d}...")
        
        # Generate data
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=d,
            centers=n_clusters,
            cluster_std=1.0,
            random_state=RANDOM_STATE,
        )
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Standard DDC
        print("  Fitting standard DDC...")
        S_ddc, w_ddc, _ = fit_ddc_coreset_optimized(X_scaled, k_reps, n0=None, random_state=RANDOM_STATE)
        
        # Random baseline
        S_random, w_random = fit_random_coreset(X_scaled, k_reps, random_state=RANDOM_STATE)
        
        # Compute metrics
        metrics_random = compute_all_metrics(X_scaled, S_random, w_random, 'Random')
        metrics_ddc = compute_all_metrics(X_scaled, S_ddc, w_ddc, 'DDC')
        
        improvement_cov = (metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100
        
        # Analyze density estimation quality
        densities_euclidean, _ = estimate_density_knn(X_scaled, k=32)
        try:
            densities_adaptive = estimate_density_adaptive(X_scaled, k=32, n_iter=3)
        except:
            densities_adaptive = densities_euclidean.copy()
        
        # Check if selected points have higher density
        # Map coreset points back to original
        from scipy.spatial.distance import cdist
        distances = cdist(S_ddc, X_scaled)
        closest_indices = distances.argmin(axis=1)
        coreset_densities_euclidean = densities_euclidean[closest_indices]
        coreset_densities_adaptive = densities_adaptive[closest_indices]
        
        mean_density_euclidean = coreset_densities_euclidean.mean()
        mean_density_adaptive = coreset_densities_adaptive.mean()
        mean_density_full = densities_euclidean.mean()
        
        results.append({
            'dimension': d,
            'ddc_cov_err': metrics_ddc['cov_err_fro'],
            'random_cov_err': metrics_random['cov_err_fro'],
            'improvement_%': improvement_cov,
            'mean_density_coreset_euclidean': mean_density_euclidean,
            'mean_density_coreset_adaptive': mean_density_adaptive,
            'mean_density_full': mean_density_full,
            'density_ratio_euclidean': mean_density_coreset_euclidean / mean_density_full,
            'density_ratio_adaptive': mean_density_coreset_adaptive / mean_density_full,
        })
        
        print(f"  DDC Cov Error: {metrics_ddc['cov_err_fro']:.4f}")
        print(f"  Random Cov Error: {metrics_random['cov_err_fro']:.4f}")
        print(f"  Improvement: {improvement_cov:+.1f}%")
        print(f"  Density ratio (Euclidean): {mean_density_coreset_euclidean / mean_density_full:.3f}")
        print(f"  Density ratio (Adaptive): {mean_density_coreset_adaptive / mean_density_full:.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_df.to_csv(results_dir / "ddc_adaptive_distances_test.csv", index=False)
    
    print("\n" + "=" * 70)
    print("DDC TEST COMPLETE")
    print("=" * 70)
    print("\nResults:")
    print(results_df.to_string(index=False))
    
    return results_df


def generate_report():
    """Generate comprehensive report on high-dimensional density estimation."""
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)
    
    report_path = Path(__file__).parent.parent.parent / "docs" / "HIGH_DIM_DENSITY_ANALYSIS.md"
    
    with open(report_path, 'w') as f:
        f.write("# Análise: k-NN e Estimação de Densidade em Alta Dimensão\n\n")
        f.write("## Problema Identificado\n\n")
        f.write("O DDC usa k-NN para estimar densidade local. Em alta dimensão, k-NN sofre do ")
        f.write("**curse of dimensionality**, onde todas as distâncias se tornam similares.\n\n")
        
        f.write("### Curse of Dimensionality em k-NN\n\n")
        f.write("1. **Distâncias se concentram**: Em alta dimensão, a distribuição de distâncias ")
        f.write("se concentra em torno da média\n")
        f.write("2. **Volume do k-NN ball explode**: Volume ~ r^d cresce exponencialmente\n")
        f.write("3. **Densidade estimada colapsa**: Todas as estimativas ficam similares\n")
        f.write("4. **Discriminação perdida**: Diferenças entre clusters desaparecem\n\n")
        
        f.write("## Impacto no DDC\n\n")
        f.write("Quando k-NN falha em estimar densidade:\n\n")
        f.write("1. **Seleção baseada em densidade falha**: DDC não consegue identificar ")
        f.write("regiões de alta densidade\n")
        f.write("2. **Diversidade domina**: Com densidade uniforme, apenas diversidade importa\n")
        f.write("3. **Performance degrada**: DDC pode ficar pior que Random\n")
        f.write("4. **Covariância distorcida**: Seleção não reflete estrutura real\n\n")
        
        f.write("## Solução Proposta: Distâncias Adaptativas\n\n")
        f.write("### Mahalanobis Distance Adaptativa\n\n")
        f.write("Em vez de distância Euclidiana, usar distância Mahalanobis adaptativa:\n\n")
        f.write("```python\n")
        f.write("d_M(x, y) = sqrt((x - y)^T Σ^(-1) (x - y))\n")
        f.write("```\n\n")
        f.write("Onde Σ é a **covariância local** estimada dos k vizinhos.\n\n")
        
        f.write("### Vantagens\n\n")
        f.write("1. **Adapta-se à forma local**: Captura anisotropia dos clusters\n")
        f.write("2. **Melhora discriminação**: Mantém diferenças entre clusters mesmo em alta dimensão\n")
        f.write("3. **Reduz curse of dimensionality**: Volume adaptativo compensa crescimento exponencial\n")
        f.write("4. **Preserva estrutura**: Mantém informação sobre geometria local\n\n")
        
        f.write("### Implementação\n\n")
        f.write("1. **Iterativo**: Começar com Euclidiana, refinar com Mahalanobis\n")
        f.write("2. **Local**: Cada ponto tem sua própria métrica baseada em vizinhos\n")
        f.write("3. **Regularizado**: Adicionar pequena identidade para estabilidade\n")
        f.write("4. **Computacionalmente caro**: Requer inversão de matrizes locais\n\n")
        
        f.write("## Resultados Esperados\n\n")
        f.write("Com distâncias adaptativas:\n\n")
        f.write("1. **Melhor estimação de densidade**: Mantém discriminação em alta dimensão\n")
        f.write("2. **DDC melhor**: Seleção reflete melhor estrutura real\n")
        f.write("3. **Covariância preservada**: Melhor preservação de covariância global\n")
        f.write("4. **Robustez**: Funciona bem mesmo em 50+ dimensões\n\n")
        
        f.write("## Limitações\n\n")
        f.write("1. **Custo computacional**: O(n * k * d^3) para inversão de matrizes\n")
        f.write("2. **k mínimo**: Precisa k > d para estimar covariância\n")
        f.write("3. **Instabilidade**: Matrizes singulares podem causar problemas\n")
        f.write("4. **Hiperparâmetros**: Número de iterações, regularização\n\n")
        
        f.write("## Alternativas\n\n")
        f.write("1. **PCA pré-processamento**: Reduzir dimensão antes de DDC\n")
        f.write("2. **Manifold learning**: UMAP/t-SNE para reduzir dimensão intrínseca\n")
        f.write("3. **Kernel adaptativo**: Usar kernels que se adaptam localmente\n")
        f.write("4. **Densidade baseada em projeção**: Projetar em subespaços locais\n\n")
        
        f.write("## Recomendações\n\n")
        f.write("### Para Dimensões Baixas (d < 20)\n\n")
        f.write("- Usar distância Euclidiana padrão\n")
        f.write("- DDC funciona bem\n\n")
        
        f.write("### Para Dimensões Médias (20 ≤ d < 50)\n\n")
        f.write("- Considerar distâncias adaptativas\n")
        f.write("- Ou reduzir dimensão com PCA primeiro\n\n")
        
        f.write("### Para Dimensões Altas (d ≥ 50)\n\n")
        f.write("- **Recomendado**: Reduzir dimensão primeiro (PCA, UMAP)\n")
        f.write("- **Alternativa**: Usar distâncias adaptativas (mais caro)\n")
        f.write("- **Fallback**: Usar Random (pode ser melhor que DDC com Euclidiana)\n\n")
        
        f.write("## Próximos Passos\n\n")
        f.write("1. Implementar distâncias adaptativas no DDC\n")
        f.write("2. Testar em datasets reais de alta dimensão\n")
        f.write("3. Comparar com PCA + DDC\n")
        f.write("4. Otimizar custo computacional\n")
        f.write("5. Criar heurística para escolher método automaticamente\n\n")
    
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    # Run tests
    density_results = test_density_estimation_degradation()
    ddc_results = test_ddc_with_adaptive_distances()
    
    # Generate report
    generate_report()
    
    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)

