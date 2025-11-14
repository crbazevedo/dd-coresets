# Heurísticas para Seleção de Parâmetros do DDC

Este documento fornece heurísticas práticas para selecionar parâmetros do DDC.

## 1. Seleção de k

### Regras Gerais

1. **k deve ser pelo menos 2-3x o número de clusters**
2. **k/n < 0.05** para manter vantagem do DDC
3. **k mínimo**: 2-3 pontos por cluster
4. **k máximo**: 5% de n (k/n = 0.05)

### Heurística por Caso de Uso

| Caso de Uso | k mínimo | k recomendado | k máximo |
|-------------|----------|---------------|----------|
| Cobertura garantida | 3× clusters | 1% de n | 5% de n |
| Balanceado | 2× clusters | 1-2% de n | 5% de n |
| Geral | 2× clusters | 1% de n | 5% de n |

### Exemplos

**Small k, 4 clusters**:
- n=20,000, n_clusters=4
- k recomendado: 200 (k/n=0.010)
- k range: 12 - 1000

**Two Moons**:
- n=5,000, n_clusters=2
- k recomendado: 50 (k/n=0.010)
- k range: 4 - 250

## 2. Seleção de alpha (Balance Densidade-Diversidade)

### Regras

- **alpha baixo (0.05-0.1)**: Mais diversidade, melhor para clusters bem separados
- **alpha médio (0.15-0.2)**: Balance, uso geral
- **alpha alto (0.3+)**: Mais densidade, melhor para dados simples

### Heurística

```python
if n_clusters > 8 or clusters_well_separated:
    alpha = 0.1  # More diversity
elif data_complexity == 'high':
    alpha = 0.1  # More diversity for complex data
else:
    alpha = 0.15-0.2  # Balanced
```

## 3. Seleção de gamma (Peso de Diversidade)

### Regras

- **gamma baixo (1.0-1.5)**: Menos ênfase em diversidade
- **gamma médio (1.5-2.0)**: Balance
- **gamma alto (2.0-2.5)**: Mais diversidade

### Heurística

```python
if k < n_clusters * 2:
    gamma = 2.0-2.5  # More diversity for small k
elif clusters_well_separated:
    gamma = 2.0  # Diversity helps
else:
    gamma = 1.5-1.8  # Balanced
```

## 4. Seleção de m_neighbors (Vizinhos para Densidade)

### Regras

- **m_neighbors baixo (8-16)**: Dados simples, clusters claros
- **m_neighbors médio (16-20)**: Uso geral
- **m_neighbors alto (20-32)**: Dados complexos, estruturas não-lineares

### Heurística

```python
if data_complexity == 'low':
    m_neighbors = 16
elif data_complexity == 'medium':
    m_neighbors = 20
else:
    m_neighbors = 24-32  # More neighbors for complex data
```

## 5. Seleção de refine_iters (Iterações de Refinamento)

### Regras

- **refine_iters = 1**: Dados simples, k grande
- **refine_iters = 2**: Uso geral (recomendado)
- **refine_iters = 3+**: Dados complexos, k pequeno

### Heurística

```python
if data_complexity == 'high' or k < n_clusters * 3:
    refine_iters = 3
elif k > n_clusters * 10:
    refine_iters = 1  # Large k, less refinement needed
else:
    refine_iters = 2  # Default
```

## 6. Quick Parameter Tuning

### Estratégia: Tune em Amostra Pequena

1. **Sample dataset**: Use 5k-10k pontos para tuning
2. **Estimate clusters**: Use K-means elbow method
3. **Estimate complexity**: Teste normalidade, multimodalidade
4. **Test parameter grid**: Teste 3-5 combinações
5. **Select best**: Escolha baseado em composite score
6. **Apply to full data**: Use parâmetros encontrados no dataset completo

### Código de Exemplo

```python
from experiments.ddc_advantage.parameter_heuristics import quick_parameter_tuning

# Quick tuning on sample
best_params = quick_parameter_tuning(X, k_target=1000, n_sample=5000)

# Apply to full data
S, w, info = fit_ddc_coreset(
    X, k=1000, n0=None,
    alpha=best_params['alpha'],
    gamma=best_params['gamma'],
    m_neighbors=best_params['m_neighbors'],
    refine_iters=best_params['refine_iters'],
    reweight_full=True,
)
```

## 7. Workflow Recomendado

### Passo a Passo

1. **Analisar dataset**:
   - Estimar número de clusters
   - Estimar complexidade
   - Verificar estrutura

2. **Selecionar k**:
   - Usar heurística baseada em n e n_clusters
   - Garantir k/n < 0.05
   - Ajustar baseado em caso de uso

3. **Selecionar parâmetros**:
   - Usar heurísticas baseadas em complexidade
   - OU fazer quick tuning em amostra pequena

4. **Validar**:
   - Comparar com Random baseline
   - Verificar métricas (cov, W1, KS)
   - Ajustar se necessário

## 8. Tabela de Referência Rápida

| Característica | alpha | gamma | m_neighbors | refine_iters |
|----------------|-------|-------|-------------|--------------|
| Clusters bem separados | 0.1 | 2.0 | 16 | 2 |
| Dados simples | 0.15 | 1.8 | 16 | 2 |
| Dados complexos | 0.1 | 2.0 | 24 | 3 |
| k muito pequeno | 0.05-0.1 | 2.0-2.5 | 16-20 | 3 |
| k grande | 0.15-0.2 | 1.5-1.8 | 16-20 | 1-2 |
| Multimodal | 0.1 | 2.0 | 20 | 2 |
| Não-convexo | 0.1 | 2.0 | 24 | 2-3 |

