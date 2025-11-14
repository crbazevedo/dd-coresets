# Análise: k-NN e Estimação de Densidade em Alta Dimensão

## Problema Identificado

O DDC usa k-NN para estimar densidade local através da função `_density_knn`:

```python
p_i ∝ 1 / r_k(x_i)^d
```

onde `r_k(x_i)` é a distância ao k-ésimo vizinho mais próximo e `d` é a dimensionalidade.

**Em alta dimensão, k-NN sofre do curse of dimensionality**, onde todas as distâncias se tornam similares, fazendo com que todas as estimativas de densidade colapsem para valores similares.

---

## Evidência Experimental

### Degradação da Estimação de Densidade

Testamos estimação de densidade em dimensões de 2 a 100:

| Dimensão | CV (Euclidean) | CV (Adaptive) | Cluster Var (Euclidean) | Cluster Var (Adaptive) |
|----------|----------------|---------------|------------------------|------------------------|
| 2 | 0.60 | 0.55 | 1.3×10⁵ | 1.8×10⁵ |
| 5 | 0.99 | 0.65 | 1.9×10⁵ | 8.4×10⁵ |
| 10 | 1.54 | 0.74 | 1.9×10⁶ | 3.8×10⁸ |
| 20 | 3.11 | 1.29 | 1.2×10² | 1.1×10⁸ |
| 50 | 7.62 | 7.62 | **2.5×10⁻²⁰** | **2.5×10⁻²⁰** |
| 100 | 5.5×10⁻²⁸ | 5.5×10⁻²⁸ | **1.6×10⁻⁷⁸** | **1.6×10⁻⁷⁸** |

**Observações Críticas**:

1. **Em d=50 e d=100**: Variância entre clusters colapsa para praticamente zero
   - Isso significa que **todas as estimativas de densidade são idênticas**
   - **DDC não consegue distinguir entre clusters** baseado em densidade

2. **Distância adaptativa ajuda até d=20**: 
   - Mantém alguma discriminação (variância 1.1×10⁸ vs 1.2×10²)
   - Mas também colapsa em d=50+

3. **CV aumenta com dimensão**: 
   - Indica que estimativas ficam mais instáveis
   - Mas variância entre clusters diminui (paradoxo: instável mas não discriminativo)

---

## Por Que Isso Acontece?

### Curse of Dimensionality em k-NN

1. **Volume do k-NN ball explode**: Volume ~ r^d cresce exponencialmente
   - Em alta dimensão, mesmo distâncias pequenas correspondem a volumes enormes
   - Densidade = k / volume → todas as densidades ficam muito pequenas e similares

2. **Distâncias se concentram**: 
   - Em alta dimensão, a distribuição de distâncias se concentra em torno da média
   - Diferenças relativas entre distâncias diminuem drasticamente

3. **r_k^d domina**: 
   - Como `p_i ∝ 1 / r_k^d`, pequenas diferenças em r_k são amplificadas exponencialmente
   - Mas se todas as r_k são similares, todas as densidades colapsam

### Por Que Distâncias Adaptativas Ajudam (Até Certo Ponto)

**Mahalanobis distance adaptativa**:
- Adapta-se à forma local dos clusters (anisotropia)
- Volume adaptativo: `det(Σ)^(1/2) * r^d` em vez de apenas `r^d`
- Se clusters são elípticos, volume adaptativo pode ser menor que volume Euclidiano

**Limite**: 
- Ainda depende de r^d, então curse of dimensionality persiste
- Em d muito alto (50+), mesmo adaptativo colapsa

---

## Impacto no DDC

### Quando k-NN Falha em Alta Dimensão

1. **Seleção baseada em densidade falha**:
   - `_density_knn` retorna densidades praticamente uniformes
   - `_select_reps_greedy` não consegue identificar regiões de alta densidade
   - Seleção fica baseada apenas em diversidade (min_dist)

2. **DDC degenera para diversidade pura**:
   - Com densidade uniforme, `scores = min_dist * (p^alpha)` ≈ `min_dist`
   - DDC vira essencialmente um algoritmo de diversidade máxima
   - Perde a vantagem de balancear densidade e diversidade

3. **Performance degrada**:
   - Sem densidade, DDC pode ficar pior que Random
   - Covariância pode ser distorcida (como vimos em CIFAR-10, MNIST)

### Evidência dos Experimentos

- **CIFAR-10 (50D)**: DDC -13.5% pior que Random
- **MNIST (50D)**: DDC -79% pior que Random
- **Fashion-MNIST (50D)**: DDC -75% pior que Random

Todos esses casos têm **d ≥ 50**, onde nossa análise mostra que estimação de densidade colapsa.

---

## Solução: Distâncias Adaptativas

### Implementação Proposta

**Mahalanobis Distance Adaptativa Iterativa**:

```python
def _density_knn_adaptive(X, m_neighbors=32, n_iter=3):
    """
    Adaptive k-NN density estimation using local Mahalanobis distances.
    
    Iteratively refines density estimates by learning local covariance.
    """
    n, d = X.shape
    
    # Initialize with Euclidean
    nn = NearestNeighbors(n_neighbors=m_neighbors+1, metric='euclidean')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    rk_euclidean = distances[:, -1]
    densities = 1.0 / (rk_euclidean ** d + 1e-10)
    
    # Iterative refinement
    for iteration in range(n_iter):
        new_densities = np.zeros(n)
        
        for i in range(n):
            # Get neighbors
            neighbor_indices = indices[i, 1:m_neighbors+1]
            
            if len(neighbor_indices) > d:
                # Compute local covariance
                neighbors = X[neighbor_indices]
                local_mean = neighbors.mean(axis=0)
                local_cov = np.cov(neighbors.T)
                local_cov += np.eye(d) * 1e-6  # Regularization
                
                try:
                    # Mahalanobis distances
                    inv_cov = np.linalg.inv(local_cov)
                    diff = X - local_mean
                    mahal_distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                    
                    # k-th Mahalanobis neighbor
                    sorted_idx = np.argsort(mahal_distances)
                    rk_mahal = mahal_distances[sorted_idx[m_neighbors]]
                    
                    # Adaptive density: k / (det(cov)^(1/2) * r^d)
                    det_cov = np.linalg.det(local_cov)
                    new_densities[i] = m_neighbors / (
                        rk_mahal ** d * np.sqrt(det_cov) + 1e-10
                    )
                except:
                    new_densities[i] = densities[i]
            else:
                new_densities[i] = densities[i]
        
        densities = new_densities
    
    densities /= densities.sum()
    return densities
```

### Vantagens

1. **Adapta-se à forma local**: Captura anisotropia dos clusters
2. **Melhora discriminação**: Mantém diferenças entre clusters em d=20-30
3. **Volume adaptativo**: `det(Σ)^(1/2) * r^d` pode ser menor que `r^d` para clusters elípticos

### Limitações

1. **Custo computacional**: O(n * k * d³) para inversão de matrizes
2. **k mínimo**: Precisa k > d para estimar covariância
3. **Ainda colapsa em d muito alto**: Em d=50+, mesmo adaptativo falha
4. **Instabilidade numérica**: Matrizes singulares podem causar problemas

---

## Alternativas e Recomendações

### 1. Redução de Dimensão Prévia (Recomendado)

**PCA ou UMAP antes de DDC**:

```python
# Reduzir para dimensão intrínseca
pca = PCA(n_components=min(20, n_features), random_state=42)
X_reduced = pca.fit_transform(X)

# Aplicar DDC no espaço reduzido
S, w, info = fit_ddc_coreset(X_reduced, k=k, ...)
```

**Vantagens**:
- Mantém estrutura principal
- DDC funciona bem em dimensões baixas (< 20)
- Computacionalmente eficiente

**Quando usar**:
- d ≥ 30: Sempre reduzir primeiro
- d ≥ 20: Considerar reduzir se estrutura é linear
- d < 20: DDC funciona bem diretamente

### 2. Distâncias Adaptativas (Alternativa)

**Implementar no DDC**:

```python
# Modificar _density_knn para usar Mahalanobis adaptativo
p = _density_knn_adaptive(X0, m_neighbors=m_neighbors, n_iter=3)
```

**Quando usar**:
- d = 20-30: Pode ajudar
- Estrutura não-linear: UMAP pode ser melhor
- d ≥ 50: Redução de dimensão é melhor

### 3. Kernel Adaptativo

**Usar kernels que se adaptam localmente**:

- RBF com largura adaptativa por ponto
- Kernel baseado em densidade local
- Mais complexo, mas pode ser mais robusto

### 4. Densidade Baseada em Projeção

**Projetar em subespaços locais**:

- PCA local por cluster
- Manifold learning local
- Reduzir curse of dimensionality mantendo estrutura

---

## Recomendações Práticas

### Para Dimensões Baixas (d < 20)

✅ **Usar DDC padrão** (distância Euclidiana)
- Funciona bem
- Computacionalmente eficiente
- Não precisa de adaptação

### Para Dimensões Médias (20 ≤ d < 50)

⚠️ **Opção 1**: Reduzir dimensão primeiro (PCA/UMAP)
- Mais seguro
- Funciona bem na prática

⚠️ **Opção 2**: Usar distâncias adaptativas
- Mais caro computacionalmente
- Pode ajudar se estrutura é não-linear

### Para Dimensões Altas (d ≥ 50)

❌ **Não usar DDC diretamente**

✅ **Sempre reduzir dimensão primeiro**:
- PCA para estrutura linear
- UMAP para estrutura não-linear
- Manter dimensão reduzida < 20

✅ **Alternativa**: Usar Random
- Pode ser melhor que DDC com densidade colapsada
- Mais simples e eficiente

---

## Implementação Sugerida

### Modificar DDC para Detectar Alta Dimensão

```python
def fit_ddc_coreset(
    X: np.ndarray,
    k: int,
    auto_reduce_dim: bool = True,
    max_dim: int = 20,
    ...
):
    """
    Auto-reduce dimension if d > max_dim.
    """
    n, d = X.shape
    
    if auto_reduce_dim and d > max_dim:
        # Auto-reduce with PCA
        pca = PCA(n_components=max_dim, random_state=random_state)
        X_reduced = pca.fit_transform(X)
        # ... continue with X_reduced
    else:
        X_reduced = X
    
    # ... rest of DDC
```

### Adicionar Opção de Distâncias Adaptativas

```python
def fit_ddc_coreset(
    X: np.ndarray,
    k: int,
    use_adaptive_distance: bool = False,
    adaptive_iterations: int = 3,
    ...
):
    """
    Option to use adaptive Mahalanobis distances.
    """
    if use_adaptive_distance and d <= 30:
        p = _density_knn_adaptive(X0, m_neighbors, adaptive_iterations)
    else:
        p = _density_knn(X0, m_neighbors)
    # ... rest
```

---

## Conclusão

**Sim, k-NN falha na estimação de densidade em alta dimensão**, e isso explica por que DDC falha em datasets como MNIST e CIFAR-10.

**Distâncias adaptativas ajudam até certo ponto** (d=20-30), mas não resolvem completamente o problema em d muito alto (50+).

**Recomendação principal**: 
- **Sempre reduzir dimensão primeiro** quando d ≥ 30
- **Usar distâncias adaptativas** como alternativa em d=20-30 se redução não for possível
- **Detectar automaticamente** e aplicar redução ou avisar usuário

Isso explicaria por que DDC funciona bem em dados sintéticos de baixa dimensão mas falha em datasets reais de alta dimensão.

