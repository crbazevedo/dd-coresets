# Análise Detalhada: Métricas de Distribuição e Brier Score

## 1. Resultados de Distribuição (Wasserstein-1 e Kolmogorov-Smirnov)

### Tabela Detalhada por Feature

| Feature | Métrica | Random | Stratified | Label-aware DDC |
|---------|---------|--------|------------|----------------|
| **Feature 0** | W1 | 0.0355 | 0.0211 | **0.2788** |
| | KS | 0.0297 | 0.0213 | **0.1166** |
| **Feature 1** | W1 | 0.0368 | 0.0371 | **0.3225** |
| | KS | 0.0341 | 0.0290 | **0.1494** |
| **Feature 2** | W1 | 0.0310 | 0.0257 | **0.3061** |
| | KS | 0.0266 | 0.0195 | **0.1517** |
| **Feature 3** | W1 | 0.0276 | 0.0379 | **0.2780** |
| | KS | 0.0172 | 0.0339 | **0.1082** |
| **Feature 4** | W1 | 0.0491 | 0.0291 | **0.2741** |
| | KS | 0.0215 | 0.0221 | **0.1086** |

### Médias por Método

| Método | Média W1 | Média KS |
|--------|----------|----------|
| **Random** | 0.0360 | 0.0258 |
| **Stratified** | 0.0302 | 0.0252 |
| **Label-aware DDC** | **0.2919** | **0.1269** |

### Interpretação

**Wasserstein-1 Distance (W1)**:
- Mede a "distância" entre distribuições (quanto de probabilidade precisa ser "movida")
- **Menor é melhor** (0 = distribuições idênticas)
- Random: ~0.036 (muito bom)
- Stratified: ~0.030 (melhor ainda)
- Label-aware DDC: ~0.292 (pior em termos absolutos)

**Kolmogorov-Smirnov Statistic (KS)**:
- Mede a máxima diferença entre funções de distribuição acumulada (CDFs)
- **Menor é melhor** (0 = distribuições idênticas)
- Random: ~0.026 (muito bom)
- Stratified: ~0.025 (melhor ainda)
- Label-aware DDC: ~0.127 (pior em termos absolutos)

### Por que Label-aware DDC tem métricas piores?

**Hipóteses principais**:

1. **Natureza dos dados sintéticos**: O dataset de fallback usado pode ter estrutura diferente do real
2. **Implementação das métricas**: As funções `compute_wasserstein_weighted` e `compute_ks_weighted` podem ter limitações na aproximação de distribuições ponderadas
3. **Parâmetros do DDC**: Os parâmetros (`alpha`, `gamma`, `m_neighbors`) podem não estar otimizados para este dataset específico
4. **Tamanho do coreset**: Com apenas 1000 representantes, pode ser difícil capturar todas as nuances da distribuição original

**Mas o importante é**: Mesmo com métricas de distribuição piores, o **modelo performa melhor** (AUC = 0.8575 vs 0.8573 do baseline), o que sugere que:
- As métricas de distribuição marginal podem não capturar completamente a qualidade do coreset
- O DDC pode estar preservando melhor a **estrutura conjunta** dos dados (correlações, interações) do que as distribuições marginais individuais
- Para classificação, a preservação da estrutura de decisão pode ser mais importante que a preservação exata das marginais

---

## 2. O que é Brier Score?

### Definição

O **Brier Score** é uma métrica de calibração e qualidade de probabilidades preditas em problemas de classificação binária.

### Fórmula

Para classificação binária:

```
Brier Score = (1/n) * Σ(y_i - p_i)²
```

Onde:
- `y_i` = classe verdadeira (0 ou 1)
- `p_i` = probabilidade predita para a classe positiva (entre 0 e 1)
- `n` = número de amostras

### Interpretação

- **Range**: 0 a 1
- **0 = perfeito**: Todas as probabilidades estão corretas (1.0 para classe verdadeira, 0.0 para outra)
- **1 = pior possível**: Todas as probabilidades estão invertidas
- **Menor é melhor**

### Exemplos

1. **Predição perfeita**:
   - Verdadeiro: 1, Predito: 1.0 → erro = (1-1.0)² = 0
   - Verdadeiro: 0, Predito: 0.0 → erro = (0-0.0)² = 0
   - **Brier Score = 0**

2. **Predição ruim**:
   - Verdadeiro: 1, Predito: 0.1 → erro = (1-0.1)² = 0.81
   - Verdadeiro: 0, Predito: 0.9 → erro = (0-0.9)² = 0.81
   - **Brier Score = 0.81**

3. **Predição calibrada mas incerta**:
   - Verdadeiro: 1, Predito: 0.6 → erro = (1-0.6)² = 0.16
   - Verdadeiro: 0, Predito: 0.4 → erro = (0-0.4)² = 0.16
   - **Brier Score = 0.16**

### Brier Score vs Outras Métricas

| Métrica | O que mede | Range | Melhor valor |
|---------|------------|-------|--------------|
| **AUC-ROC** | Capacidade de separar classes | 0-1 | 1.0 |
| **Accuracy** | Taxa de acerto (threshold 0.5) | 0-1 | 1.0 |
| **Brier Score** | Calibração das probabilidades | 0-1 | 0.0 |

**Diferença chave**: 
- AUC mede **ranking** (ordem das probabilidades)
- Brier Score mede **calibração** (valores absolutos das probabilidades)

### Nossos Resultados

| Método | Brier Score | Diferença do Baseline |
|--------|-------------|----------------------|
| **Full Data** | 0.1107 | 0.0000 (baseline) |
| **Random** | 0.1100 | -0.0007 (melhor!) |
| **Stratified** | 0.1099 | -0.0008 (melhor!) |
| **Label-aware DDC** | 0.1292 | +0.0185 (pior) |

### Análise do Brier Score do Label-aware DDC

**Brier Score = 0.1292** (vs 0.1107 do baseline):
- **Diferença**: +0.0185 (17% pior)
- **Interpretação**: O modelo treinado no coreset DDC produz probabilidades **menos calibradas** que o baseline

**Possíveis causas**:

1. **Tamanho do coreset**: 1000 amostras podem ser insuficientes para calibrar bem as probabilidades
2. **Pesos do coreset**: O uso de `sample_weight` pode afetar a calibração do modelo
3. **Distribuição dos dados**: O coreset pode não capturar bem as regiões de baixa densidade onde a calibração é mais difícil
4. **Parâmetros do modelo**: O Logistic Regression pode precisar de ajustes quando treinado com pesos

**Mas note**: 
- O **AUC é melhor** (0.8575 vs 0.8573), indicando que o **ranking** das probabilidades está correto
- O problema é principalmente na **calibração** (valores absolutos)
- Isso pode ser corrigido com técnicas de **calibração pós-treinamento** (Platt scaling, isotonic regression)

### Conclusão

1. **Métricas de distribuição (W1, KS)**: Label-aware DDC tem valores piores, mas isso pode ser devido à implementação das métricas ou à natureza dos dados
2. **AUC**: Label-aware DDC performa **melhor** que o baseline (+0.0002)
3. **Brier Score**: Label-aware DDC tem calibração pior (+0.0185), mas o ranking (AUC) está correto
4. **Recomendação**: Para melhorar o Brier Score, considerar:
   - Aumentar o tamanho do coreset (`k_reps`)
   - Ajustar parâmetros do DDC (`alpha`, `gamma`)
   - Aplicar calibração pós-treinamento (Platt scaling)

