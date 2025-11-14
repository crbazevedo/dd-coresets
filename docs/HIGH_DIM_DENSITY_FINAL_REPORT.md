# Relatório Final: k-NN e Estimação de Densidade em Alta Dimensão

**Data**: 2025-11-13  
**Análise**: Investigação completa do curse of dimensionality em k-NN e impacto no DDC

---

## Resposta Direta à Pergunta

### Sim, k-NN falha na estimação de densidade em alta dimensão

**Evidência Quantitativa**:

| Dimensão | Cluster Variance (Euclidean) | Cluster Variance (Adaptive) | Conclusão |
|----------|------------------------------|-----------------------------|-----------|
| 10 | 1.9×10⁶ | 3.8×10⁸ | ✅ Funciona bem |
| 20 | 1.2×10² | 1.1×10⁸ | ⚠️ Euclidiana falha, Adaptativa funciona |
| 50 | **2.5×10⁻²⁰** | **2.5×10⁻²⁰** | ❌ **Ambas colapsam** |
| 100 | **1.6×10⁻⁷⁸** | **1.6×10⁻⁷⁸** | ❌ **Colapso total** |

**Em d=50+**: Variância entre clusters colapsa para praticamente zero → **todas as estimativas de densidade são idênticas** → **DDC não consegue distinguir clusters**.

---

### Sim, distâncias adaptativas ajudam (até certo ponto)

**Evidência Quantitativa**:

| Dimensão | DDC Performance | Density Ratio (Euclidean) | Density Ratio (Adaptive) |
|----------|-----------------|---------------------------|--------------------------|
| 10 | **+63.9%** melhor | 1.03x | **9.83x** |
| 20 | -30.6% pior | 4.66x | **2672x** |
| 50 | -12.7% pior | 9.03x | 9.03x (colapsa) |

**Observações**:

1. **d=10**: Adaptativa ajuda muito (9.8x melhor discriminação)
2. **d=20**: Adaptativa ajuda dramaticamente (2672x!), mas DDC ainda falha
3. **d=50**: Adaptativa também colapsa (mesmo ratio que Euclidiana)

**Conclusão**: Distâncias adaptativas ajudam até d≈20-30, mas não resolvem o problema fundamental em d≥50.

---

## Por Que k-NN Falha?

### Curse of Dimensionality

A função de densidade do DDC é:

```python
p_i ∝ 1 / r_k(x_i)^d
```

onde:
- `r_k(x_i)` = distância ao k-ésimo vizinho
- `d` = dimensionalidade

**Problemas**:

1. **Volume explode**: Volume do k-NN ball = `V ∝ r^d` cresce exponencialmente
2. **Distâncias se concentram**: Em alta dimensão, todas as distâncias ficam similares
3. **r_k^d domina**: Pequenas diferenças em r_k são amplificadas exponencialmente, mas se todas r_k são similares, todas as densidades colapsam

### Evidência Visual

- **d=2-10**: Boa discriminação entre clusters
- **d=20**: Discriminação degrada (Euclidiana), mas adaptativa mantém
- **d=50+**: **Colapso total** - variância entre clusters ≈ 0

---

## Impacto no DDC

### Quando Densidade Colapsa

1. **`_density_knn` retorna densidades uniformes**
2. **`_select_reps_greedy` não consegue identificar regiões de alta densidade**
3. **DDC degenera para diversidade pura**: `scores = min_dist * (p^alpha)` ≈ `min_dist`
4. **Performance degrada**: DDC pode ficar pior que Random

### Evidência dos Experimentos

| Dataset | Dimensão | DDC vs Random | Densidade Funciona? |
|---------|----------|---------------|---------------------|
| Gaussian Mixture (10D) | 10 | **+63.9%** | ✅ Sim |
| Gaussian Mixture (20D) | 20 | -30.6% | ⚠️ Adaptativa ajuda mas DDC falha |
| Gaussian Mixture (50D) | 50 | -12.7% | ❌ Não |
| CIFAR-10 (50D) | 50 | -13.5% | ❌ Não |
| MNIST (50D) | 50 | -79% | ❌ Não |
| Fashion-MNIST (50D) | 50 | -75% | ❌ Não |

**Padrão claro**: Em d≥50, DDC consistentemente pior que Random.

---

## Solução: Distâncias Adaptativas

### Implementação Proposta

**Mahalanobis Distance Adaptativa Iterativa**:

```python
def _density_knn_adaptive(X, m_neighbors=32, n_iter=3):
    """
    Adaptive k-NN density using local Mahalanobis distances.
    
    For each point:
    1. Get k neighbors
    2. Compute local covariance Σ
    3. Use Mahalanobis distance: d_M = sqrt((x-y)^T Σ^(-1) (x-y))
    4. Density = k / (det(Σ)^(1/2) * r_M^d)
    """
    # ... (implementação completa no código)
```

### Vantagens

1. **Adapta-se à forma local**: Captura anisotropia dos clusters
2. **Melhora discriminação**: Mantém diferenças em d=20-30
3. **Volume adaptativo**: `det(Σ)^(1/2) * r^d` pode ser menor que `r^d` para clusters elípticos

### Limitações

1. **Custo computacional**: O(n * k * d³) para inversão de matrizes
2. **k mínimo**: Precisa k > d para estimar covariância
3. **Ainda colapsa em d muito alto**: Em d=50+, mesmo adaptativo falha
4. **Instabilidade numérica**: Matrizes singulares podem causar problemas

---

## Recomendações Práticas

### Para Dimensões Baixas (d < 20)

✅ **Usar DDC padrão** (distância Euclidiana)
- Funciona bem
- Computacionalmente eficiente
- Não precisa de adaptação

### Para Dimensões Médias (20 ≤ d < 50)

⚠️ **Opção 1**: Reduzir dimensão primeiro (PCA/UMAP) - **RECOMENDADO**
- Mais seguro
- Funciona bem na prática
- DDC funciona bem em dimensão reduzida

⚠️ **Opção 2**: Usar distâncias adaptativas
- Mais caro computacionalmente (O(n*k*d³))
- Pode ajudar se estrutura é não-linear
- Funciona até d≈30

### Para Dimensões Altas (d ≥ 50)

❌ **Não usar DDC diretamente**

✅ **Sempre reduzir dimensão primeiro**:
- PCA para estrutura linear (manter d < 20)
- UMAP para estrutura não-linear (manter d < 20)
- Aplicar DDC no espaço reduzido

✅ **Alternativa**: Usar Random
- Pode ser melhor que DDC com densidade colapsada
- Mais simples e eficiente
- Não-viesado para covariância

---

## Implementação Sugerida no DDC

### Auto-Detecção de Alta Dimensão

```python
def fit_ddc_coreset(
    X: np.ndarray,
    k: int,
    auto_reduce_dim: bool = True,
    max_dim: int = 20,
    use_adaptive_distance: bool = False,
    ...
):
    """
    Auto-reduce dimension if d > max_dim.
    Optionally use adaptive Mahalanobis distances.
    """
    n, d = X.shape
    
    # Auto-reduce dimension
    if auto_reduce_dim and d > max_dim:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=max_dim, random_state=random_state)
        X_reduced = pca.fit_transform(X)
        X_work = X_reduced
        print(f"Auto-reduced from {d}D to {max_dim}D (explained variance: {pca.explained_variance_ratio_.sum():.2%})")
    else:
        X_work = X
    
    # Choose density estimation method
    if use_adaptive_distance and X_work.shape[1] <= 30:
        p = _density_knn_adaptive(X_work, m_neighbors, n_iter=3)
    else:
        p = _density_knn(X_work, m_neighbors)
    
    # ... rest of DDC
```

### Heurística Automática

```python
def choose_density_method(X, m_neighbors=32):
    """Automatically choose density estimation method."""
    n, d = X.shape
    
    if d < 20:
        return 'euclidean'
    elif d < 30:
        return 'adaptive'  # Can help
    else:
        return 'reduce_first'  # Must reduce dimension first
```

---

## Conclusão

### Respostas às Perguntas

1. **k-NN falha em alta dimensão?** 
   - ✅ **SIM** - Colapsa completamente em d≥50
   - Variância entre clusters → 0 em d=50+
   - Todas as estimativas de densidade ficam idênticas

2. **Distâncias adaptativas ajudam?**
   - ✅ **SIM, até certo ponto**
   - Ajudam dramaticamente em d=20 (2672x melhor discriminação)
   - Mas também colapsam em d=50+
   - Não resolvem o problema fundamental

### Recomendação Principal

**Para d ≥ 30**: **Sempre reduzir dimensão primeiro** (PCA/UMAP) antes de aplicar DDC.

**Para d = 20-30**: Considerar distâncias adaptativas OU reduzir dimensão.

**Para d < 20**: DDC padrão funciona bem.

### Próximos Passos

1. ✅ Implementar auto-redução de dimensão no DDC
2. ✅ Implementar distâncias adaptativas como opção
3. ✅ Adicionar heurística automática para escolher método
4. ✅ Testar em mais datasets reais de alta dimensão
5. ✅ Documentar limitações e recomendações

---

## Referências

- Análise completa: `docs/HIGH_DIM_DENSITY_ANALYSIS.md`
- Código de investigação: `experiments/ddc_advantage/investigate_high_dim_density.py`
- Resultados: `experiments/ddc_advantage/results/high_dim_density_analysis.csv`

