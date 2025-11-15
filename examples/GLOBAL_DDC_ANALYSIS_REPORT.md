# Análise de Parâmetros: Global DDC

## Resumo Executivo

Análise completa do efeito dos parâmetros do Global DDC nas métricas de distribuição conjunta (covariância, correlação, MMD). **180 combinações** de parâmetros foram testadas.

**Data**: 2025-11-12  
**Dataset**: Synthetic (30,000 samples, 10 features)  
**k_reps**: 1000

## Melhores Parâmetros Encontrados

| Parâmetro | Valor Ótimo | Valor Padrão | Mudança |
|-----------|-------------|--------------|---------|
| **alpha** | **0.1** | 0.3 | -66% (mais diversidade) |
| **gamma** | **2.0** | 1.0 | +100% (kernel mais suave) |
| **m_neighbors** | **16** | 32 | -50% (menos neighbors) |
| **refine_iters** | **2** | 1 | +100% (mais refinamento) |

## Métricas com Parâmetros Otimizados

| Métrica | Valor |
|---------|-------|
| **Mean Error (L2)** | 0.0406 |
| **Covariance Error (Frobenius)** | 0.3189 |
| **Correlation Error (Frobenius)** | 0.1669 |
| **MMD** | 0.0489 |
| **Composite Score** | 0.4023 |

## Análise do Efeito de Cada Parâmetro

### 1. Alpha (Trade-off Density-Diversity)

**Efeito**: O parâmetro mais importante!

| Alpha | Mean Composite Score | Cov Error | Corr Error |
|-------|---------------------|-----------|------------|
| **0.1** | **0.652** | 0.32 | 0.17 |
| 0.2 | 2.543 | 2.16 | 0.77 |
| 0.3 | 2.969 | 2.62 | 0.92 |
| 0.4 | 3.050 | 2.70 | 0.95 |
| 0.5 | 3.056 | 2.71 | 0.96 |

**Conclusão**: 
- Alpha menor (mais diversidade) = **MUITO MELHOR** para distribuição conjunta
- Diferença entre alpha=0.1 e alpha=0.5: **4.7x melhor**
- Alpha=0.1 favorece cobertura espacial ampla, essencial para preservar covariância global

### 2. Gamma (Kernel Scale)

**Efeito**: Importante, mas menos crítico que alpha

| Gamma | Mean Composite Score | Cov Error | Corr Error |
|-------|---------------------|-----------|------------|
| **2.0** | **2.386** | 1.99 | 0.80 |
| 1.5 | 2.413 | 2.02 | 0.80 |
| 1.0 | 2.460 | 2.07 | 0.81 |
| 0.5 | 2.558 | 2.16 | 0.84 |

**Conclusão**:
- Gamma maior (kernel mais suave) = melhor para distribuição conjunta
- Diferença entre gamma=2.0 e gamma=0.5: **7% melhor**
- Kernel mais suave distribui pesos de forma mais uniforme, melhorando aproximação da covariância

### 3. m_neighbors (Densidade Local)

**Efeito**: Pequeno, mas consistente

| m_neighbors | Mean Composite Score | Cov Error | Corr Error |
|-------------|---------------------|-----------|------------|
| **16** | **2.432** | 2.04 | 0.80 |
| 32 | 2.451 | 2.06 | 0.81 |
| 64 | 2.480 | 2.09 | 0.82 |

**Conclusão**:
- Menos neighbors (estimativa mais local) = ligeiramente melhor
- Diferença pequena (~2%), mas consistente
- Estimativa mais local pode capturar melhor variações locais na densidade

### 4. refine_iters (Refinamento)

**Efeito**: Mínimo

| refine_iters | Mean Composite Score | Cov Error | Corr Error |
|--------------|---------------------|-----------|------------|
| **3** | **2.435** | 2.04 | 0.80 |
| 2 | 2.444 | 2.05 | 0.80 |
| 1 | 2.483 | 2.08 | 0.81 |

**Conclusão**:
- Mais iterações = ligeiramente melhor, mas diferença muito pequena (~2%)
- Refine_iters=2 é um bom compromisso entre qualidade e tempo

## Interação Alpha vs Gamma

A interação entre alpha e gamma é crucial:

| Alpha | Gamma=0.5 | Gamma=1.0 | Gamma=1.5 | Gamma=2.0 |
|-------|-----------|-----------|-----------|-----------|
| **0.1** | 0.917 | **0.664** | **0.545** | **0.482** |
| 0.2 | 2.624 | 2.549 | 2.511 | 2.489 |
| 0.3 | 3.026 | 2.974 | 2.946 | 2.931 |
| 0.4 | 3.107 | 3.055 | 3.027 | 3.012 |
| 0.5 | 3.114 | 3.061 | 3.033 | 3.017 |

**Observação**: 
- Com alpha=0.1, gamma=2.0 produz o melhor resultado (0.482)
- Com alpha alto (>0.3), o efeito de gamma é menor
- **Alpha é o fator dominante**, mas gamma otimiza ainda mais quando alpha é baixo

## Comparação: Padrão vs Otimizado

### Parâmetros Padrão (alpha=0.3, gamma=1.0, m_neighbors=32, refine_iters=1)

- Covariance error: ~2.62
- Correlation error: ~0.92
- Composite score: ~2.97

### Parâmetros Otimizados (alpha=0.1, gamma=2.0, m_neighbors=16, refine_iters=2)

- Covariance error: **0.32** (88% melhor!)
- Correlation error: **0.17** (82% melhor!)
- Composite score: **0.40** (87% melhor!)

**Melhoria total**: ~87% de redução no composite score!

## Insights Principais

1. **Alpha é o parâmetro mais crítico**: 
   - Reduzir alpha de 0.3 para 0.1 melhora métricas em ~4.7x
   - Favorecer diversidade sobre densidade é essencial para preservar covariância global

2. **Gamma otimiza quando alpha é baixo**:
   - Com alpha=0.1, aumentar gamma de 1.0 para 2.0 melhora em ~27%
   - Kernel mais suave distribui pesos melhor

3. **m_neighbors tem efeito pequeno mas consistente**:
   - Menos neighbors (16) é ligeiramente melhor que mais (32, 64)
   - Estimativa mais local pode ser mais precisa

4. **refine_iters tem efeito mínimo**:
   - Diferença entre 1 e 3 iterações é <2%
   - Refine_iters=2 é suficiente

## Recomendações

### Para Global DDC (unsupervised):

**Parâmetros recomendados**:
- `alpha = 0.1` (favorecer diversidade)
- `gamma = 2.0` (kernel suave)
- `m_neighbors = 16` (estimativa local)
- `refine_iters = 2` (refinamento adequado)

### Comparação com Label-Aware DDC:

| Método | Alpha | Gamma | Cov Error | Corr Error |
|--------|-------|-------|-----------|------------|
| **Global DDC (otimizado)** | 0.1 | 2.0 | 0.32 | 0.17 |
| **Label-aware DDC (otimizado)** | 0.2 | 1.5 | 1.78 | 0.61 |

**Observação**: 
- Global DDC otimizado tem métricas **melhores** que Label-aware DDC
- Mas Global DDC **distorce proporções de classe** (não apropriado para classificação)
- Label-aware DDC preserva classes mas tem métricas piores (trade-off)

## Limitações

1. **Dataset específico**: Resultados podem variar com outros datasets
2. **Sem downstream task**: Não avaliamos impacto em performance de modelo
3. **Grid search limitado**: Pode haver combinações ainda melhores fora do grid testado

## Arquivos Gerados

- `examples/global_ddc_optimization_results.csv` - Todos os 180 resultados
- `examples/global_ddc_best_parameters.json` - Parâmetros ótimos em JSON

## Próximos Passos

1. Validar em outros datasets
2. Testar impacto em downstream tasks
3. Explorar valores intermediários (ex: alpha=0.15, gamma=1.75)
4. Comparar com Label-aware DDC em termos de trade-offs

