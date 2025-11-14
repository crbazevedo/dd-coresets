#!/usr/bin/env python3
"""
Investigate why DDC fails in certain scenarios.

Focus on:
- Skewed/heavy-tailed distributions
- Complex real datasets (MNIST, Fashion-MNIST)
- Concentric rings
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from scipy import stats
from experiments.ddc_advantage.utils import (
    compute_all_metrics, fit_random_coreset, fit_ddc_coreset_optimized,
    weighted_mean, weighted_cov, RANDOM_STATE
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')


def investigate_skewed_distributions():
    """Investigate why DDC fails in skewed distributions."""
    print("=" * 70)
    print("INVESTIGAÇÃO: Por que DDC falha em Skewed Distributions?")
    print("=" * 70)
    
    n_samples = 20_000
    n_features = 8
    k_reps = 1000
    
    # Generate skewed data
    rng = np.random.RandomState(RANDOM_STATE)
    X = np.zeros((n_samples, n_features))
    
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
            # Normal (baseline)
            X[:, i] = rng.randn(n_samples)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit coresets
    S_random, w_random = fit_random_coreset(X_scaled, k_reps, random_state=RANDOM_STATE)
    S_ddc, w_ddc, _ = fit_ddc_coreset_optimized(X_scaled, k_reps, n0=None, random_state=RANDOM_STATE)
    
    # Analyze selection bias
    print("\nAnálise de Viés de Seleção:")
    
    # Check quantiles
    quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    
    for feat_idx in range(min(4, n_features)):
        print(f"\nFeature {feat_idx}:")
        
        # Full data quantiles
        q_full = np.quantile(X_scaled[:, feat_idx], quantiles)
        
        # Random quantiles (unweighted)
        q_random = np.quantile(S_random[:, feat_idx], quantiles)
        
        # DDC quantiles (weighted)
        indices = np.random.choice(len(S_ddc), size=5000, p=w_ddc, replace=True)
        S_ddc_sample = S_ddc[indices, feat_idx]
        q_ddc = np.quantile(S_ddc_sample, quantiles)
        
        print(f"  Quantile    Full      Random    DDC       Random Err  DDC Err")
        for q, qf, qr, qd in zip(quantiles, q_full, q_random, q_ddc):
            err_r = abs(qf - qr)
            err_d = abs(qf - qd)
            print(f"  {q:6.2f}  {qf:8.4f}  {qr:8.4f}  {qd:8.4f}  {err_r:8.4f}  {err_d:8.4f}")
    
    # Check where DDC selects points
    print("\nAnálise de Densidade de Seleção:")
    
    for feat_idx in range(min(2, n_features)):
        # Bin data
        bins = np.linspace(X_scaled[:, feat_idx].min(), X_scaled[:, feat_idx].max(), 20)
        hist_full, _ = np.histogram(X_scaled[:, feat_idx], bins=bins)
        hist_random, _ = np.histogram(S_random[:, feat_idx], bins=bins)
        
        # Weighted histogram for DDC
        hist_ddc = np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            mask = (S_ddc[:, feat_idx] >= bins[i]) & (S_ddc[:, feat_idx] < bins[i+1])
            hist_ddc[i] = w_ddc[mask].sum()
        
        # Normalize
        hist_full = hist_full / hist_full.sum()
        hist_random = hist_random / hist_random.sum()
        hist_ddc = hist_ddc / hist_ddc.sum()
        
        print(f"\nFeature {feat_idx} - Distribuição de seleção:")
        print(f"  Full data: mean={X_scaled[:, feat_idx].mean():.4f}, std={X_scaled[:, feat_idx].std():.4f}")
        print(f"  Random: mean={S_random[:, feat_idx].mean():.4f}, std={S_random[:, feat_idx].std():.4f}")
        print(f"  DDC: mean={weighted_mean(S_ddc[:, feat_idx:feat_idx+1], w_ddc)[0]:.4f}, "
              f"std={np.sqrt(weighted_cov(S_ddc[:, feat_idx:feat_idx+1], w_ddc)[0,0]):.4f}")
        
        # Check tail coverage
        tail_threshold = np.quantile(X_scaled[:, feat_idx], 0.95)
        tail_full = np.sum(X_scaled[:, feat_idx] > tail_threshold) / len(X_scaled)
        tail_random = np.sum(S_random[:, feat_idx] > tail_threshold) / len(S_random)
        tail_ddc = np.sum(w_ddc[S_ddc[:, feat_idx] > tail_threshold])
        
        print(f"\n  Tail coverage (Q0.95):")
        print(f"    Full: {tail_full:.4f}")
        print(f"    Random: {tail_random:.4f}")
        print(f"    DDC: {tail_ddc:.4f}")
    
    return {
        'X': X_scaled,
        'S_random': S_random,
        'w_random': w_random,
        'S_ddc': S_ddc,
        'w_ddc': w_ddc,
    }


def investigate_mnist_failure():
    """Investigate why DDC fails in MNIST."""
    print("\n" + "=" * 70)
    print("INVESTIGAÇÃO: Por que DDC falha em MNIST?")
    print("=" * 70)
    
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.decomposition import PCA
        
        print("Loading MNIST...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X_full = mnist.data.astype(np.float32)
        y_full = mnist.target.astype(int)
        
        # Sample subset
        indices = np.random.RandomState(RANDOM_STATE).choice(len(X_full), size=10_000, replace=False)
        X = X_full[indices]
        y = y_full[indices]
        
        # Reduce dimensionality
        pca = PCA(n_components=50, random_state=RANDOM_STATE)
        X_reduced = pca.fit_transform(X)
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        
        k_reps = 1000
        
        # Fit coresets
        S_random, w_random = fit_random_coreset(X_scaled, k_reps, random_state=RANDOM_STATE)
        S_ddc, w_ddc, _ = fit_ddc_coreset_optimized(X_scaled, k_reps, n0=None, random_state=RANDOM_STATE)
        
        # Analyze structure
        print("\nAnálise de Estrutura:")
        
        # Check if data is Gaussian
        from scipy.stats import normaltest
        print("\nTeste de Normalidade (por feature, amostra):")
        non_normal_count = 0
        for i in range(min(10, X_scaled.shape[1])):
            _, p_value = normaltest(X_scaled[:1000, i])
            if p_value < 0.05:
                non_normal_count += 1
        print(f"  Features não-normais: {non_normal_count}/10")
        
        # Check digit distribution in coresets
        print("\nDistribuição de Dígitos nos Coresets:")
        
        # Map coreset points back to original indices (approximate)
        from scipy.spatial.distance import cdist
        distances_random = cdist(S_random, X_scaled)
        closest_random = distances_random.argmin(axis=1)
        y_random_coreset = y[closest_random]
        
        distances_ddc = cdist(S_ddc, X_scaled)
        closest_ddc = distances_ddc.argmin(axis=1)
        y_ddc_coreset = y[closest_ddc]
        
        print("\n  Full data:")
        unique, counts = np.unique(y, return_counts=True)
        for digit, count in zip(unique, counts):
            print(f"    {digit}: {count/len(y):.2%}")
        
        print("\n  Random coreset:")
        unique, counts = np.unique(y_random_coreset, return_counts=True)
        for digit, count in zip(unique, counts):
            print(f"    {digit}: {count/len(y_random_coreset):.2%}")
        
        print("\n  DDC coreset:")
        unique, counts = np.unique(y_ddc_coreset, return_counts=True)
        for digit, count in zip(unique, counts):
            print(f"    {digit}: {count/len(y_ddc_coreset):.2%}")
        
        # Check covariance structure
        print("\nAnálise de Covariância:")
        cov_full = np.cov(X_scaled, rowvar=False)
        cov_random = np.cov(S_random, rowvar=False)
        cov_ddc = weighted_cov(S_ddc, w_ddc)
        
        print(f"  Full data: trace={np.trace(cov_full):.4f}, det={np.linalg.det(cov_full):.2e}")
        print(f"  Random: trace={np.trace(cov_random):.4f}, det={np.linalg.det(cov_random):.2e}")
        print(f"  DDC: trace={np.trace(cov_ddc):.4f}, det={np.linalg.det(cov_ddc):.2e}")
        
        # Check eigenvalues
        eigenvals_full = np.linalg.eigvals(cov_full)
        eigenvals_random = np.linalg.eigvals(cov_random)
        eigenvals_ddc = np.linalg.eigvals(cov_ddc)
        
        print(f"\n  Top 5 eigenvalues:")
        print(f"    Full: {np.sort(eigenvals_full)[-5:][::-1]}")
        print(f"    Random: {np.sort(eigenvals_random)[-5:][::-1]}")
        print(f"    DDC: {np.sort(eigenvals_ddc)[-5:][::-1]}")
        
        return {
            'X': X_scaled,
            'y': y,
            'S_random': S_random,
            'S_ddc': S_ddc,
            'w_ddc': w_ddc,
        }
        
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        return None


def investigate_concentric_rings():
    """Investigate why DDC fails in concentric rings."""
    print("\n" + "=" * 70)
    print("INVESTIGAÇÃO: Por que DDC falha em Concentric Rings?")
    print("=" * 70)
    
    n_samples = 10_000
    n_rings = 3
    k_reps = 1000
    
    # Generate concentric rings
    rng = np.random.RandomState(RANDOM_STATE)
    X_list = []
    labels_list = []
    
    samples_per_ring = n_samples // n_rings
    
    for ring_idx in range(n_rings):
        radius = 1.0 + ring_idx * 2.0
        angles = rng.uniform(0, 2 * np.pi, samples_per_ring)
        
        x = radius * np.cos(angles) + rng.normal(0, 0.1, samples_per_ring)
        y = radius * np.sin(angles) + rng.normal(0, 0.1, samples_per_ring)
        
        X_ring = np.column_stack([x, y])
        X_list.append(X_ring)
        labels_list.append(np.full(samples_per_ring, ring_idx))
    
    X = np.vstack(X_list)
    labels = np.hstack(labels_list)
    
    # Shuffle
    indices = np.arange(len(X))
    rng.shuffle(indices)
    X = X[indices]
    labels = labels[indices]
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit coresets
    S_random, w_random = fit_random_coreset(X_scaled, k_reps, random_state=RANDOM_STATE)
    S_ddc, w_ddc, _ = fit_ddc_coreset_optimized(X_scaled, k_reps, n0=None, random_state=RANDOM_STATE)
    
    # Analyze ring coverage
    print("\nAnálise de Cobertura por Anel:")
    
    from scipy.spatial.distance import cdist
    
    for ring_idx in range(n_rings):
        ring_mask = (labels == ring_idx)
        ring_points = X_scaled[ring_mask]
        
        # Distance from coreset to ring
        dists_random = cdist(S_random, ring_points).min(axis=1)
        dists_ddc = cdist(S_ddc, ring_points).min(axis=1)
        
        # Count points close to this ring
        threshold = 0.5  # Distance threshold
        random_close = np.sum(dists_random < threshold)
        ddc_close = np.sum(dists_ddc < threshold)
        
        print(f"\n  Ring {ring_idx} (radius ~{1.0 + ring_idx * 2.0:.1f}):")
        print(f"    Points in ring: {np.sum(ring_mask)}")
        print(f"    Random coreset points close: {random_close}/{k_reps}")
        print(f"    DDC coreset points close: {ddc_close}/{k_reps}")
        print(f"    Mean distance (Random): {dists_random.mean():.4f}")
        print(f"    Mean distance (DDC): {dists_ddc.mean():.4f}")
    
    # Check if DDC selects more from certain rings
    print("\nDistribuição de Seleção por Anel:")
    
    # Find which ring each coreset point is closest to
    ring_centers = []
    for ring_idx in range(n_rings):
        ring_mask = (labels == ring_idx)
        center = X_scaled[ring_mask].mean(axis=0)
        ring_centers.append(center)
    ring_centers = np.array(ring_centers)
    
    dists_random_to_centers = cdist(S_random, ring_centers)
    closest_ring_random = dists_random_to_centers.argmin(axis=1)
    
    dists_ddc_to_centers = cdist(S_ddc, ring_centers)
    closest_ring_ddc = dists_ddc_to_centers.argmin(axis=1)
    
    print("\n  Random coreset:")
    for ring_idx in range(n_rings):
        count = np.sum(closest_ring_random == ring_idx)
        print(f"    Ring {ring_idx}: {count}/{k_reps} ({count/k_reps:.1%})")
    
    print("\n  DDC coreset:")
    for ring_idx in range(n_rings):
        count = np.sum(closest_ring_ddc == ring_idx)
        weight = w_ddc[closest_ring_ddc == ring_idx].sum()
        print(f"    Ring {ring_idx}: {count}/{k_reps} ({count/k_reps:.1%}), weight={weight:.3f}")
    
    return {
        'X': X_scaled,
        'labels': labels,
        'S_random': S_random,
        'S_ddc': S_ddc,
        'w_ddc': w_ddc,
    }


def generate_failure_report():
    """Generate comprehensive failure analysis report."""
    print("\n" + "=" * 70)
    print("GENERATING FAILURE ANALYSIS REPORT")
    print("=" * 70)
    
    results_dir = Path(__file__).parent / "results"
    output_dir = Path(__file__).parent.parent.parent / "docs" / "images" / "ddc_advantage"
    
    # Run investigations
    skewed_results = investigate_skewed_distributions()
    mnist_results = investigate_mnist_failure()
    rings_results = investigate_concentric_rings()
    
    # Generate report
    report_path = Path(__file__).parent.parent.parent / "docs" / "DDC_FAILURE_ANALYSIS.md"
    
    with open(report_path, 'w') as f:
        f.write("# Análise de Falhas do DDC\n\n")
        f.write("Este relatório investiga por que DDC falha em certos cenários.\n\n")
        
        f.write("## 1. Distribuições Skewed/Heavy-tailed\n\n")
        f.write("### Problema Identificado\n\n")
        f.write("DDC falha em preservar caudas de distribuições skewed/heavy-tailed.\n\n")
        f.write("### Causa Provável\n\n")
        f.write("1. **Seleção baseada em densidade**: DDC seleciona pontos em regiões de alta densidade\n")
        f.write("2. **Caudas têm baixa densidade**: Outliers e caudas são sub-representados\n")
        f.write("3. **Diversidade não compensa**: Mesmo com diversidade, DDC não captura bem regiões esparsas\n")
        f.write("4. **Pesos não corrigem**: Os pesos não conseguem compensar a falta de pontos nas caudas\n\n")
        
        f.write("### Solução Proposta\n\n")
        f.write("1. **Ajustar alpha**: Reduzir alpha para dar mais peso à diversidade\n")
        f.write("2. **Aumentar m_neighbors**: Melhorar estimativa de densidade local\n")
        f.write("3. **Pré-processamento**: Transformar para distribuição mais simétrica\n")
        f.write("4. **Híbrido**: Combinar DDC com amostragem de caudas\n\n")
        
        f.write("## 2. Datasets Reais Complexos (MNIST, Fashion-MNIST)\n\n")
        f.write("### Problema Identificado\n\n")
        f.write("DDC tem covariância muito pior (-75% a -80%) em datasets de imagens.\n\n")
        f.write("### Causa Provável\n\n")
        f.write("1. **Estrutura não-Gaussiana complexa**: Distribuições muito diferentes de Gaussian\n")
        f.write("2. **Alta dimensionalidade**: Mesmo após PCA, estrutura é complexa\n")
        f.write("3. **Múltiplas escalas**: Diferentes dígitos têm diferentes variâncias\n")
        f.write("4. **Correlações não-lineares**: DDC não captura bem dependências não-lineares\n")
        f.write("5. **Viés de seleção**: Seleção baseada em densidade distorce covariância global\n\n")
        
        f.write("### Solução Proposta\n\n")
        f.write("1. **Label-aware DDC**: Aplicar DDC por classe separadamente\n")
        f.write("2. **Pré-clustering**: Aplicar DDC dentro de clusters pré-definidos\n")
        f.write("3. **Parâmetros específicos**: Ajustar alpha, gamma para estrutura específica\n")
        f.write("4. **Usar Random**: Para datasets muito complexos, Random pode ser melhor\n\n")
        
        f.write("## 3. Anéis Concêntricos\n\n")
        f.write("### Problema Identificado\n\n")
        f.write("DDC falha em estruturas de anéis concêntricos (-77% covariância, -78% W1).\n\n")
        f.write("### Causa Provável\n\n")
        f.write("1. **Estrutura muito específica**: Anéis são estruturas não-convexas muito específicas\n")
        f.write("2. **Densidade uniforme**: Dentro de cada anel, densidade é uniforme\n")
        f.write("3. **Seleção concentrada**: DDC pode concentrar seleção em alguns anéis\n")
        f.write("4. **Geometria não-linear**: DDC não captura bem geometria circular\n\n")
        
        f.write("### Solução Proposta\n\n")
        f.write("1. **Aumentar diversidade**: Reduzir alpha, aumentar gamma\n")
        f.write("2. **Estratificação por anel**: Aplicar DDC por anel separadamente\n")
        f.write("3. **Métrica de distância adaptada**: Usar distância angular para anéis\n")
        f.write("4. **Usar Random**: Para estruturas muito específicas, Random pode ser melhor\n\n")
        
        f.write("## Recomendações Gerais\n\n")
        f.write("### Quando DDC Falha\n\n")
        f.write("1. **Distribuições com caudas importantes**: Use Random ou híbrido\n")
        f.write("2. **Datasets muito complexos**: Use Random ou Label-aware DDC\n")
        f.write("3. **Estruturas muito específicas**: Teste primeiro, pode precisar de Random\n\n")
        
        f.write("### Estratégias de Mitigação\n\n")
        f.write("1. **Pré-análise**: Verificar estrutura antes de escolher método\n")
        f.write("2. **Híbrido**: Combinar DDC com Random para caudas\n")
        f.write("3. **Label-aware**: Usar DDC por classe/cluster quando aplicável\n")
        f.write("4. **Parâmetros adaptativos**: Ajustar parâmetros baseado em características do dataset\n\n")
    
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    generate_failure_report()

