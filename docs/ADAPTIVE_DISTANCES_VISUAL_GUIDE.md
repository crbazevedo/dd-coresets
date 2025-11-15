# Guia Visual: Distâncias Adaptativas

## Comparação Visual: Euclidiana vs Mahalanobis

### Cenário: Cluster Elíptico em 2D

```
Euclidiana (padrão):
─────────────────────

    ●●●●●●●●●
    ●●●●●●●●●
    ●●●●●●●●●
    ●●●●●●●●●
    
Distância: sqrt((x1-x2)² + (y1-y2)²)
Volume: π * r² (círculo)
Problema: Trata x e y igualmente


Mahalanobis (adaptativa):
─────────────────────────

    ●●●●●●●●●
    ●●●●●●●●●
    ●●●●●●●●●
    ●●●●●●●●●
    
Distância: sqrt((x1-x2)²/σ_x² + (y1-y2)²/σ_y²)
Volume: π * sqrt(σ_x * σ_y) * r² (elipse)
Vantagem: Adapta-se à forma do cluster
```

---

## Passo a Passo da Implementação

### 1. Inicialização (Euclidiana)

```
Para cada ponto x_i:
  1. Encontrar k vizinhos mais próximos (Euclidiana)
  2. Calcular r_k = distância ao k-ésimo vizinho
  3. Densidade inicial: p_i = k / (r_k^d)
```

**Exemplo**:
- Ponto `x_i` = (0, 0)
- k = 5, d = 2
- 5º vizinho está a distância r_5 = 1.5
- Densidade: p_i = 5 / (1.5²) = 2.22

---

### 2. Estimação de Covariância Local

```
Para cada ponto x_i:
  1. Pegar k vizinhos encontrados na etapa 1
  2. Calcular média local: μ_local = mean(vizinhos)
  3. Calcular covariância local: Σ_local = cov(vizinhos)
```

**Exemplo**:
- Vizinhos de `x_i`: [(1, 0.1), (2, 0.2), (3, 0.1), (4, 0.3), (5, 0.2)]
- Média local: μ_local = (3.0, 0.18)
- Covariância local:
  ```
  Σ_local = [[2.5, 0.05],
             [0.05, 0.01]]
  ```
- Determinante: det(Σ) = 2.5 * 0.01 - 0.05² = 0.02475

---

### 3. Distância Mahalanobis

```
Para cada ponto x_j:
  1. Diferença: diff = x_j - μ_local
  2. Transformar: diff_transformed = Σ^(-1) @ diff
  3. Distância: d_M = sqrt(diff^T @ diff_transformed)
```

**Exemplo**:
- Ponto `x_j` = (4, 0.3)
- μ_local = (3.0, 0.18)
- diff = (1.0, 0.12)
- Σ^(-1) = [[0.404, -2.02],
            [-2.02, 102.0]]
- diff_transformed = (0.404*1.0 - 2.02*0.12, -2.02*1.0 + 102.0*0.12)
                   = (0.162, 10.24)
- d_M = sqrt(0.162² + 10.24²) = 10.24

**Interpretação**: 
- Diferença em y (0.12) é amplificada porque variância em y é baixa (0.01)
- Diferença em x (1.0) é comprimida porque variância em x é alta (2.5)

---

### 4. Densidade Adaptativa

```
Para cada ponto x_i:
  1. Encontrar k-ésimo vizinho Mahalanobis: r_k_M
  2. Calcular determinante: det(Σ)
  3. Densidade: p_i = k / (r_k_M^d * sqrt(det(Σ)))
```

**Exemplo**:
- k = 5, d = 2
- r_5_M = 2.0 (5º vizinho Mahalanobis)
- det(Σ) = 0.02475
- Densidade: p_i = 5 / (2.0² * sqrt(0.02475))
              = 5 / (4.0 * 0.157)
              = 5 / 0.628
              = 7.96

**Comparação com Euclidiana**:
- Euclidiana: p_i = 5 / (1.5²) = 2.22
- Adaptativa: p_i = 7.96
- **3.6x maior!** (melhor discriminação)

---

## Por Que Funciona: Exemplo Numérico

### Cenário: Cluster Elíptico em 50D

**Cluster**:
- 10 dimensões com variância alta (σ² = 10)
- 40 dimensões com variância baixa (σ² = 0.1)

**Covariância Local**:
```
Σ = diag([10, 10, ..., 10, 0.1, 0.1, ..., 0.1])
   └─ 10 vezes ─┘ └─ 40 vezes ─┘
```

**Determinante**:
```
det(Σ) = 10^10 * 0.1^40
       = 10^10 * 10^-40
       = 10^-30
```

**Volume Adaptativo**:
```
V_adaptive = sqrt(10^-30) * r^50
           = 10^-15 * r^50
```

**Volume Euclidiano**:
```
V_euclidean = r^50
```

**Razão**:
```
V_adaptive / V_euclidean = 10^-15
```

**Densidade Adaptativa**:
```
p_adaptive = k / (10^-15 * r^50)
           = k * 10^15 / r^50
```

**Densidade Euclidiana**:
```
p_euclidean = k / r^50
```

**Melhoria**: 
- Adaptativa é **10^15 vezes maior** que Euclidiana!
- Mantém discriminação mesmo em 50D

**Mas**: Se todas as distâncias r são similares, ambas colapsam.

---

## Algoritmo Completo (Pseudocódigo)

```
function estimate_density_adaptive(X, k, n_iter):
    n, d = shape(X)
    
    # 1. Inicialização com Euclidiana
    nn = NearestNeighbors(k+1, metric='euclidean')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    rk_euclidean = distances[:, k]
    densities = k / (rk_euclidean^d)
    
    # 2. Refinamento iterativo
    for iteration in 1..n_iter:
        new_densities = zeros(n)
        
        for each point x_i:
            # 2a. Obter vizinhos
            neighbors = X[indices[i, 1:k+1]]
            
            # 2b. Estimar covariância local
            μ_local = mean(neighbors)
            Σ_local = cov(neighbors) + ε*I  # Regularização
            
            # 2c. Computar distâncias Mahalanobis
            inv_Σ = inv(Σ_local)
            for each point x_j:
                diff = x_j - μ_local
                d_M[j] = sqrt(diff^T @ inv_Σ @ diff)
            
            # 2d. Encontrar k-ésimo vizinho Mahalanobis
            rk_M = kth_smallest(d_M)
            
            # 2e. Calcular densidade adaptativa
            det_Σ = det(Σ_local)
            new_densities[i] = k / (rk_M^d * sqrt(det_Σ))
        
        densities = new_densities
    
    return normalize(densities)
```

---

## Interpretação Geométrica

### Espaço Euclidiano

```
        ●
       ●●●
      ●●●●●
     ●●●●●●●
    ●●●●●●●●●
    
Todas as direções são iguais
Volume = r^d (esfera)
```

### Espaço Mahalanobis (Adaptado)

```
    ●●●●●●●●●
    ●●●●●●●●●
    ●●●●●●●●●
    ●●●●●●●●●
    
Direções principais são diferentes
Volume = sqrt(det(Σ)) * r^d (elipsóide)
```

**Transformação**:
- Espaço original → Espaço transformado onde covariância é identidade
- Distâncias refletem forma local do cluster
- Volume adaptativo é menor para clusters elípticos

---

## Custo Computacional

### Complexidade

| Operação | Custo | Frequência | Total |
|----------|-------|------------|-------|
| k-NN Euclidiano | O(n log n * d) | 1x | O(n log n * d) |
| Covariância local | O(k * d²) | n * n_iter | O(n * k * d² * n_iter) |
| Inversão de matriz | O(d³) | n * n_iter | O(n * d³ * n_iter) |
| Distâncias Mahalanobis | O(n * d²) | n * n_iter | O(n² * d² * n_iter) |

**Total**: O(n² * d² * n_iter) ou O(n * d³ * n_iter) dependendo do que domina

**Na prática**:
- n = 10,000, d = 20, n_iter = 3
- ~10⁹ operações (segundos a minutos)

---

## Quando Ajuda vs Quando Não Ajuda

### ✅ Ajuda Quando:

1. **Clusters elípticos** (anisotrópicos)
   ```
   Cluster alongado:
   ●●●●●●●●●●●●●●●●●●●
   ```
   - det(Σ) pequeno → volume adaptativo menor
   - Melhor discriminação

2. **Dimensão média** (d = 20-30)
   - Ainda há estrutura para capturar
   - Não colapsou completamente

3. **k > d**
   - Suficientes vizinhos para estimar covariância
   - Matriz não-singular

### ❌ Não Ajuda Quando:

1. **Clusters esféricos** (isotrópicos)
   ```
   Cluster redondo:
       ●●●
      ●●●●●
       ●●●
   ```
   - det(Σ) ≈ constante → volume similar
   - Pouca vantagem sobre Euclidiana

2. **Dimensão muito alta** (d ≥ 50)
   - Mesmo adaptativo colapsa
   - Reduzir dimensão primeiro é melhor

3. **Dados aleatórios**
   - Sem estrutura local
   - Covariância não tem significado

---

## Comparação com Alternativas

### 1. PCA Global + DDC

```python
# Reduzir dimensão globalmente
pca = PCA(n_components=20)
X_reduced = pca.fit_transform(X)
S, w = fit_ddc_coreset(X_reduced, k)
```

**Vantagem**: Mais simples, mais estável  
**Desvantagem**: Perde estrutura não-linear

### 2. UMAP + DDC

```python
# Reduzir dimensão preservando estrutura não-linear
umap = UMAP(n_components=20)
X_reduced = umap.fit_transform(X)
S, w = fit_ddc_coreset(X_reduced, k)
```

**Vantagem**: Preserva estrutura não-linear  
**Desvantagem**: Mais caro, menos interpretável

### 3. Distâncias Adaptativas

```python
# Adaptar localmente sem reduzir dimensão
S, w = fit_ddc_coreset(X, k, use_adaptive_distance=True)
```

**Vantagem**: Mantém todas as dimensões, adapta localmente  
**Desvantagem**: Mais caro, ainda colapsa em d muito alto

---

## Resumo Executivo

### O Que São?

Distâncias adaptativas usam **Mahalanobis distance** com **covariância local** estimada dos k vizinhos de cada ponto.

### Por Que Ajudam?

1. **Capturam anisotropia**: Clusters elípticos têm volume adaptativo menor
2. **Reduzem curse of dimensionality**: Para clusters estruturados
3. **Mantêm discriminação**: Em d=20-30 onde Euclidiana falha

### Limitações?

1. **Ainda colapsa em d≥50**: Limite fundamental
2. **Custo alto**: O(n * d³ * n_iter)
3. **Requer k > d**: Para estimar covariância

### Quando Usar?

- **d = 20-30**: Considerar usar
- **d < 20**: Desnecessário (Euclidiana funciona)
- **d ≥ 50**: Reduzir dimensão primeiro

---

## Referências

- Código: `experiments/ddc_advantage/investigate_high_dim_density.py`
- Análise: `docs/HIGH_DIM_DENSITY_ANALYSIS.md`
- Explicação detalhada: `docs/ADAPTIVE_DISTANCES_EXPLAINED.md`

