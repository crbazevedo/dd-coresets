#!/usr/bin/env python3
"""
Heuristics for setting DDC parameters (k, alpha, gamma, m_neighbors, refine_iters).

Strategies:
1. Quick parameter tuning on small sample
2. Heuristics based on dataset characteristics
3. Adaptive parameter selection
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons
from sklearn.decomposition import PCA
from experiments.ddc_advantage.utils import (
    fit_ddc_coreset_optimized, compute_all_metrics, fit_random_coreset,
    RANDOM_STATE
)

from dd_coresets import fit_ddc_coreset


def estimate_n_clusters(X, max_clusters=10):
    """Estimate number of clusters using simple heuristic."""
    from sklearn.cluster import KMeans
    
    # Use elbow method on small sample
    n_sample = min(5000, len(X))
    indices = np.random.RandomState(RANDOM_STATE).choice(len(X), size=n_sample, replace=False)
    X_sample = X[indices]
    
    inertias = []
    for k in range(2, min(max_clusters + 1, n_sample // 10)):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
        kmeans.fit(X_sample)
        inertias.append(kmeans.inertia_)
    
    # Simple elbow detection
    if len(inertias) < 2:
        return 2
    
    # Find elbow (largest decrease)
    decreases = np.diff(inertias)
    if len(decreases) == 0:
        return 2
    
    elbow_idx = np.argmax(np.abs(decreases)) + 1
    n_clusters = min(elbow_idx + 2, max_clusters)
    
    return n_clusters


def estimate_data_complexity(X):
    """Estimate data complexity (simple heuristic)."""
    # Check if data is roughly Gaussian
    from scipy.stats import normaltest
    
    n_sample = min(1000, len(X))
    indices = np.random.RandomState(RANDOM_STATE).choice(len(X), size=n_sample, replace=False)
    X_sample = X[indices]
    
    # Test normality on sample of features
    n_features_to_test = min(10, X.shape[1])
    non_normal_count = 0
    
    for i in range(n_features_to_test):
        _, p_value = normaltest(X_sample[:, i])
        if p_value < 0.05:
            non_normal_count += 1
    
    complexity_score = non_normal_count / n_features_to_test
    
    # Check for multimodality (simple heuristic)
    multimodal_score = 0
    for i in range(min(5, X.shape[1])):
        hist, _ = np.histogram(X_sample[:, i], bins=20)
        peaks = len([j for j in range(1, len(hist)-1) if hist[j] > hist[j-1] and hist[j] > hist[j+1]])
        if peaks > 2:
            multimodal_score += 1
    
    multimodal_score = multimodal_score / min(5, X.shape[1])
    
    return {
        'non_gaussian_ratio': complexity_score,
        'multimodal_ratio': multimodal_score,
        'complexity': 'high' if complexity_score > 0.5 else 'medium' if complexity_score > 0.2 else 'low',
    }


def quick_parameter_tuning(X, k_target, n_sample=5000, n_clusters_est=None):
    """
    Quick parameter tuning on small sample.
    
    Tests different parameter combinations and returns best.
    """
    print("=" * 70)
    print("QUICK PARAMETER TUNING")
    print("=" * 70)
    
    # Sample for tuning
    if len(X) > n_sample:
        indices = np.random.RandomState(RANDOM_STATE).choice(len(X), size=n_sample, replace=False)
        X_tune = X[indices]
    else:
        X_tune = X
        n_sample = len(X)
    
    # Scale
    scaler = StandardScaler()
    X_tune_scaled = scaler.fit_transform(X_tune)
    
    # Estimate clusters if not provided
    if n_clusters_est is None:
        n_clusters_est = estimate_n_clusters(X_tune_scaled)
        print(f"Estimated clusters: {n_clusters_est}")
    
    # Estimate complexity
    complexity = estimate_data_complexity(X_tune_scaled)
    print(f"Data complexity: {complexity['complexity']}")
    print(f"  Non-Gaussian ratio: {complexity['non_gaussian_ratio']:.2f}")
    print(f"  Multimodal ratio: {complexity['multimodal_ratio']:.2f}")
    
    # Adjust k for tuning (smaller)
    k_tune = min(k_target, n_sample // 10)
    
    # Parameter grid (reduced for speed)
    if complexity['complexity'] == 'low':
        # Simple data: focus on diversity
        param_grid = [
            {'alpha': 0.1, 'gamma': 2.0, 'm_neighbors': 16, 'refine_iters': 2},
            {'alpha': 0.2, 'gamma': 1.5, 'm_neighbors': 16, 'refine_iters': 2},
            {'alpha': 0.1, 'gamma': 1.5, 'm_neighbors': 20, 'refine_iters': 2},
        ]
    elif complexity['complexity'] == 'medium':
        # Medium complexity: balance
        param_grid = [
            {'alpha': 0.1, 'gamma': 2.0, 'm_neighbors': 16, 'refine_iters': 2},
            {'alpha': 0.2, 'gamma': 1.5, 'm_neighbors': 16, 'refine_iters': 2},
            {'alpha': 0.15, 'gamma': 1.8, 'm_neighbors': 20, 'refine_iters': 2},
        ]
    else:
        # High complexity: more refinement
        param_grid = [
            {'alpha': 0.1, 'gamma': 2.0, 'm_neighbors': 16, 'refine_iters': 3},
            {'alpha': 0.15, 'gamma': 1.8, 'm_neighbors': 20, 'refine_iters': 3},
            {'alpha': 0.2, 'gamma': 1.5, 'm_neighbors': 24, 'refine_iters': 2},
        ]
    
    # Baseline: Random
    S_random, w_random = fit_random_coreset(X_tune_scaled, k_tune, random_state=RANDOM_STATE)
    metrics_random = compute_all_metrics(X_tune_scaled, S_random, w_random, 'Random')
    
    # Test each parameter combination
    results = []
    
    for params in param_grid:
        try:
            S_ddc, w_ddc, _ = fit_ddc_coreset(
                X_tune_scaled, k=k_tune, n0=None,
                alpha=params['alpha'], gamma=params['gamma'],
                m_neighbors=params['m_neighbors'], refine_iters=params['refine_iters'],
                reweight_full=True, random_state=RANDOM_STATE,
            )
            
            metrics_ddc = compute_all_metrics(X_tune_scaled, S_ddc, w_ddc, 'DDC')
            
            # Composite score (lower is better)
            score = metrics_ddc['cov_err_fro'] + 0.5 * metrics_ddc['corr_err_fro']
            
            results.append({
                **params,
                'cov_err': metrics_ddc['cov_err_fro'],
                'corr_err': metrics_ddc['corr_err_fro'],
                'W1_mean': metrics_ddc['W1_mean'],
                'score': score,
                'improvement_vs_random': (metrics_random['cov_err_fro'] / metrics_ddc['cov_err_fro'] - 1) * 100,
            })
        except Exception as e:
            print(f"  Error with params {params}: {e}")
            continue
    
    if not results:
        print("No valid results!")
        return None
    
    results_df = pd.DataFrame(results)
    best_params = results_df.loc[results_df['score'].idxmin()]
    
    print("\nParameter Tuning Results:")
    print(results_df.to_string(index=False))
    print(f"\nBest parameters: {best_params.to_dict()}")
    
    return best_params.to_dict()


def heuristic_k_selection(n, n_clusters=None, use_case='general'):
    """
    Heuristic for selecting k.
    
    Rules:
    - k should be at least 2-3x number of clusters
    - k/n should be < 0.05 for DDC advantage
    - Adjust based on use case
    """
    if n_clusters is None:
        # Default: assume moderate number of clusters
        n_clusters = max(4, int(np.sqrt(n) / 10))
    
    if use_case == 'guaranteed_coverage':
        # Need at least 1 point per cluster, preferably 2-3
        k_min = n_clusters * 3
    elif use_case == 'balanced':
        # Balance between coverage and efficiency
        k_min = n_clusters * 2
    else:
        # General use
        k_min = n_clusters * 2
    
    # Upper bound: k/n < 0.05 for DDC advantage
    k_max = int(n * 0.05)
    
    # Recommended: between min and max
    k_recommended = min(max(k_min, int(n * 0.01)), k_max)
    
    return {
        'k_min': k_min,
        'k_max': k_max,
        'k_recommended': k_recommended,
        'k_n_ratio': k_recommended / n,
    }


def heuristic_parameters(X, k, n_clusters=None, complexity=None):
    """
    Heuristic parameter selection based on dataset characteristics.
    """
    if complexity is None:
        complexity_info = estimate_data_complexity(X)
        complexity = complexity_info['complexity']
    
    if n_clusters is None:
        n_clusters = estimate_n_clusters(X)
    
    # Base parameters
    if complexity == 'low':
        # Simple data: can use more diversity
        params = {
            'alpha': 0.1,
            'gamma': 2.0,
            'm_neighbors': 16,
            'refine_iters': 2,
        }
    elif complexity == 'medium':
        # Medium: balance
        params = {
            'alpha': 0.15,
            'gamma': 1.8,
            'm_neighbors': 20,
            'refine_iters': 2,
        }
    else:
        # High: more refinement needed
        params = {
            'alpha': 0.1,
            'gamma': 2.0,
            'm_neighbors': 24,
            'refine_iters': 3,
        }
    
    # Adjust based on k and n_clusters
    if k < n_clusters * 2:
        # Very small k: increase diversity
        params['alpha'] = max(0.05, params['alpha'] - 0.05)
        params['gamma'] = min(2.5, params['gamma'] + 0.3)
    
    if k > n_clusters * 10:
        # Large k: can reduce refinement
        params['refine_iters'] = max(1, params['refine_iters'] - 1)
    
    return params


def generate_heuristics_report():
    """Generate comprehensive heuristics report."""
    print("=" * 70)
    print("GENERATING PARAMETER HEURISTICS REPORT")
    print("=" * 70)
    
    # Test heuristics on different scenarios
    scenarios = []
    
    # Scenario 1: Small k, well-separated clusters
    X1, _ = make_blobs(n_samples=20_000, n_features=10, centers=4, random_state=RANDOM_STATE)
    k1 = heuristic_k_selection(len(X1), n_clusters=4, use_case='guaranteed_coverage')
    params1 = heuristic_parameters(X1, k1['k_recommended'], n_clusters=4)
    scenarios.append({
        'scenario': 'Small k, 4 clusters',
        'n': len(X1),
        'n_clusters': 4,
        'k_heuristic': k1,
        'params_heuristic': params1,
    })
    
    # Scenario 2: Medium k, complex data
    X2, _ = make_moons(n_samples=5_000, noise=0.1, random_state=RANDOM_STATE)
    k2 = heuristic_k_selection(len(X2), n_clusters=2, use_case='balanced')
    params2 = heuristic_parameters(X2, k2['k_recommended'], n_clusters=2)
    scenarios.append({
        'scenario': 'Two Moons',
        'n': len(X2),
        'n_clusters': 2,
        'k_heuristic': k2,
        'params_heuristic': params2,
    })
    
    # Generate report
    report_path = Path(__file__).parent.parent.parent / "docs" / "DDC_PARAMETER_HEURISTICS.md"
    
    with open(report_path, 'w') as f:
        f.write("# Heurísticas para Seleção de Parâmetros do DDC\n\n")
        f.write("Este documento fornece heurísticas práticas para selecionar parâmetros do DDC.\n\n")
        
        f.write("## 1. Seleção de k\n\n")
        f.write("### Regras Gerais\n\n")
        f.write("1. **k deve ser pelo menos 2-3x o número de clusters**\n")
        f.write("2. **k/n < 0.05** para manter vantagem do DDC\n")
        f.write("3. **k mínimo**: 2-3 pontos por cluster\n")
        f.write("4. **k máximo**: 5% de n (k/n = 0.05)\n\n")
        
        f.write("### Heurística por Caso de Uso\n\n")
        f.write("| Caso de Uso | k mínimo | k recomendado | k máximo |\n")
        f.write("|-------------|----------|---------------|----------|\n")
        f.write("| Cobertura garantida | 3× clusters | 1% de n | 5% de n |\n")
        f.write("| Balanceado | 2× clusters | 1-2% de n | 5% de n |\n")
        f.write("| Geral | 2× clusters | 1% de n | 5% de n |\n\n")
        
        f.write("### Exemplos\n\n")
        for scenario in scenarios:
            f.write(f"**{scenario['scenario']}**:\n")
            f.write(f"- n={scenario['n']:,}, n_clusters={scenario['n_clusters']}\n")
            f.write(f"- k recomendado: {scenario['k_heuristic']['k_recommended']} ")
            f.write(f"(k/n={scenario['k_heuristic']['k_n_ratio']:.3f})\n")
            f.write(f"- k range: {scenario['k_heuristic']['k_min']} - {scenario['k_heuristic']['k_max']}\n\n")
        
        f.write("## 2. Seleção de alpha (Balance Densidade-Diversidade)\n\n")
        f.write("### Regras\n\n")
        f.write("- **alpha baixo (0.05-0.1)**: Mais diversidade, melhor para clusters bem separados\n")
        f.write("- **alpha médio (0.15-0.2)**: Balance, uso geral\n")
        f.write("- **alpha alto (0.3+)**: Mais densidade, melhor para dados simples\n\n")
        
        f.write("### Heurística\n\n")
        f.write("```python\n")
        f.write("if n_clusters > 8 or clusters_well_separated:\n")
        f.write("    alpha = 0.1  # More diversity\n")
        f.write("elif data_complexity == 'high':\n")
        f.write("    alpha = 0.1  # More diversity for complex data\n")
        f.write("else:\n")
        f.write("    alpha = 0.15-0.2  # Balanced\n")
        f.write("```\n\n")
        
        f.write("## 3. Seleção de gamma (Peso de Diversidade)\n\n")
        f.write("### Regras\n\n")
        f.write("- **gamma baixo (1.0-1.5)**: Menos ênfase em diversidade\n")
        f.write("- **gamma médio (1.5-2.0)**: Balance\n")
        f.write("- **gamma alto (2.0-2.5)**: Mais diversidade\n\n")
        
        f.write("### Heurística\n\n")
        f.write("```python\n")
        f.write("if k < n_clusters * 2:\n")
        f.write("    gamma = 2.0-2.5  # More diversity for small k\n")
        f.write("elif clusters_well_separated:\n")
        f.write("    gamma = 2.0  # Diversity helps\n")
        f.write("else:\n")
        f.write("    gamma = 1.5-1.8  # Balanced\n")
        f.write("```\n\n")
        
        f.write("## 4. Seleção de m_neighbors (Vizinhos para Densidade)\n\n")
        f.write("### Regras\n\n")
        f.write("- **m_neighbors baixo (8-16)**: Dados simples, clusters claros\n")
        f.write("- **m_neighbors médio (16-20)**: Uso geral\n")
        f.write("- **m_neighbors alto (20-32)**: Dados complexos, estruturas não-lineares\n\n")
        
        f.write("### Heurística\n\n")
        f.write("```python\n")
        f.write("if data_complexity == 'low':\n")
        f.write("    m_neighbors = 16\n")
        f.write("elif data_complexity == 'medium':\n")
        f.write("    m_neighbors = 20\n")
        f.write("else:\n")
        f.write("    m_neighbors = 24-32  # More neighbors for complex data\n")
        f.write("```\n\n")
        
        f.write("## 5. Seleção de refine_iters (Iterações de Refinamento)\n\n")
        f.write("### Regras\n\n")
        f.write("- **refine_iters = 1**: Dados simples, k grande\n")
        f.write("- **refine_iters = 2**: Uso geral (recomendado)\n")
        f.write("- **refine_iters = 3+**: Dados complexos, k pequeno\n\n")
        
        f.write("### Heurística\n\n")
        f.write("```python\n")
        f.write("if data_complexity == 'high' or k < n_clusters * 3:\n")
        f.write("    refine_iters = 3\n")
        f.write("elif k > n_clusters * 10:\n")
        f.write("    refine_iters = 1  # Large k, less refinement needed\n")
        f.write("else:\n")
        f.write("    refine_iters = 2  # Default\n")
        f.write("```\n\n")
        
        f.write("## 6. Quick Parameter Tuning\n\n")
        f.write("### Estratégia: Tune em Amostra Pequena\n\n")
        f.write("1. **Sample dataset**: Use 5k-10k pontos para tuning\n")
        f.write("2. **Estimate clusters**: Use K-means elbow method\n")
        f.write("3. **Estimate complexity**: Teste normalidade, multimodalidade\n")
        f.write("4. **Test parameter grid**: Teste 3-5 combinações\n")
        f.write("5. **Select best**: Escolha baseado em composite score\n")
        f.write("6. **Apply to full data**: Use parâmetros encontrados no dataset completo\n\n")
        
        f.write("### Código de Exemplo\n\n")
        f.write("```python\n")
        f.write("from experiments.ddc_advantage.parameter_heuristics import quick_parameter_tuning\n\n")
        f.write("# Quick tuning on sample\n")
        f.write("best_params = quick_parameter_tuning(X, k_target=1000, n_sample=5000)\n\n")
        f.write("# Apply to full data\n")
        f.write("S, w, info = fit_ddc_coreset(\n")
        f.write("    X, k=1000, n0=None,\n")
        f.write("    alpha=best_params['alpha'],\n")
        f.write("    gamma=best_params['gamma'],\n")
        f.write("    m_neighbors=best_params['m_neighbors'],\n")
        f.write("    refine_iters=best_params['refine_iters'],\n")
        f.write("    reweight_full=True,\n")
        f.write(")\n")
        f.write("```\n\n")
        
        f.write("## 7. Workflow Recomendado\n\n")
        f.write("### Passo a Passo\n\n")
        f.write("1. **Analisar dataset**:\n")
        f.write("   - Estimar número de clusters\n")
        f.write("   - Estimar complexidade\n")
        f.write("   - Verificar estrutura\n\n")
        f.write("2. **Selecionar k**:\n")
        f.write("   - Usar heurística baseada em n e n_clusters\n")
        f.write("   - Garantir k/n < 0.05\n")
        f.write("   - Ajustar baseado em caso de uso\n\n")
        f.write("3. **Selecionar parâmetros**:\n")
        f.write("   - Usar heurísticas baseadas em complexidade\n")
        f.write("   - OU fazer quick tuning em amostra pequena\n\n")
        f.write("4. **Validar**:\n")
        f.write("   - Comparar com Random baseline\n")
        f.write("   - Verificar métricas (cov, W1, KS)\n")
        f.write("   - Ajustar se necessário\n\n")
        
        f.write("## 8. Tabela de Referência Rápida\n\n")
        f.write("| Característica | alpha | gamma | m_neighbors | refine_iters |\n")
        f.write("|----------------|-------|-------|-------------|--------------|\n")
        f.write("| Clusters bem separados | 0.1 | 2.0 | 16 | 2 |\n")
        f.write("| Dados simples | 0.15 | 1.8 | 16 | 2 |\n")
        f.write("| Dados complexos | 0.1 | 2.0 | 24 | 3 |\n")
        f.write("| k muito pequeno | 0.05-0.1 | 2.0-2.5 | 16-20 | 3 |\n")
        f.write("| k grande | 0.15-0.2 | 1.5-1.8 | 16-20 | 1-2 |\n")
        f.write("| Multimodal | 0.1 | 2.0 | 20 | 2 |\n")
        f.write("| Não-convexo | 0.1 | 2.0 | 24 | 2-3 |\n\n")
    
    print(f"\nReport saved to: {report_path}")
    print("=" * 70)
    print("PARAMETER HEURISTICS REPORT GENERATED")
    print("=" * 70)


if __name__ == "__main__":
    generate_heuristics_report()
    
    # Test quick tuning
    print("\n" + "=" * 70)
    print("TESTING QUICK PARAMETER TUNING")
    print("=" * 70)
    
    X_test, _ = make_blobs(n_samples=20_000, n_features=10, centers=4, random_state=RANDOM_STATE)
    best_params = quick_parameter_tuning(X_test, k_target=1000, n_sample=5000)
    
    if best_params:
        print("\n" + "=" * 70)
        print("QUICK TUNING COMPLETE")
        print("=" * 70)

