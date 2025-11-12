# Comparação: Global DDC Otimizado vs Baselines

## Resumo Executivo

Comparação completa das métricas de distribuição conjunta entre Global DDC (padrão e otimizado) e baselines (Random, Stratified).

**Dataset**: Adult Census Income (34,189 amostras de treino, 6 features numéricas)  
**k_reps**: 1000  
**Data**: 2025-11-12

## Resultados Completos

| Método | Mean Error | Cov Error | Corr Error | MMD | Composite Score |
|--------|------------|-----------|------------|-----|-----------------|
| **Random** | 0.0792 | **0.3285** | 0.2083 | **0.0590** | **0.4326** |
| **Stratified** | 0.0896 | 0.6056 | **0.2020** | 0.0524 | 0.7065 |
| **Global DDC (default)** | 0.3919 | 1.6513 | 0.2342 | 0.3210 | 1.7684 |
| **Global DDC (optimized)** | 0.4381 | 1.6353 | 0.1483 | 0.1260 | 1.7095 |

**Ranking** (menor composite score é melhor):
1. **Random** (0.4326) - MELHOR
2. **Stratified** (0.7065) - 63% pior que Random
3. **Global DDC (optimized)** (1.7095) - 295% pior que Random
4. **Global DDC (default)** (1.7684) - 309% pior que Random

## Comparação Relativa (vs Random)

### Covariance Error

| Método | Valor | vs Random |
|--------|-------|-----------|
| Random | 0.3285 | Baseline |
| Stratified | 0.6056 | **+84% pior** |
| Global DDC (optimized) | 1.6353 | **+398% pior** |
| Global DDC (default) | 1.6513 | **+403% pior** |

### Correlation Error

| Método | Valor | vs Random |
|--------|-------|-----------|
| Random | 0.2083 | Baseline |
| Stratified | 0.2020 | **-3% melhor** |
| Global DDC (optimized) | 0.1483 | **-29% melhor** |
| Global DDC (default) | 0.2342 | **+12% pior** |

### Composite Score

| Método | Valor | vs Random |
|--------|-------|-----------|
| Random | 0.4326 | Baseline |
| Stratified | 0.7065 | **+63% pior** |
| Global DDC (optimized) | 1.7095 | **+295% pior** |
| Global DDC (default) | 1.7684 | **+309% pior** |

## Observações Importantes

### 1. Random é o Melhor Baseline

**Surpreendente**: Random sampling produz as **melhores métricas de distribuição conjunta**!

**Possíveis razões**:
- Random sampling preserva naturalmente a estrutura estatística do dataset completo
- Não há viés de seleção que distorça covariâncias
- Com 1000 amostras de um dataset grande, random sampling é estatisticamente representativo

### 2. Global DDC Tem Covariância Pior

**Problema identificado**: Global DDC (mesmo otimizado) tem **covariância error muito maior** que Random:
- Random: 0.33
- Global DDC (optimized): 1.64 (**5x pior**)

**Possíveis causas**:
- DDC seleciona pontos baseado em densidade-diversidade, o que pode **distorcer a covariância global**
- A seleção greedy pode favorecer certas regiões do espaço, alterando a estrutura de covariância
- Os pesos podem não compensar completamente a distorção espacial

### 3. Global DDC Tem Correlação Melhor

**Paradoxo interessante**: Global DDC (optimized) tem **correlação error menor** que Random:
- Random: 0.21
- Global DDC (optimized): 0.15 (**29% melhor**)

**Possível explicação**:
- Correlação é normalizada (não depende da escala)
- DDC pode preservar melhor a **estrutura de correlação relativa** entre features
- Mas falha em preservar as **variâncias absolutas** (covariância)

### 4. Otimização Melhora, Mas Não Resolve

**Comparação Default vs Optimized**:
- Cov error: 1.65 → 1.64 (melhoria mínima, <1%)
- Corr error: 0.23 → 0.15 (**35% melhor**)
- Composite: 1.77 → 1.71 (**3% melhor**)

**Conclusão**: A otimização melhora principalmente a **correlação**, mas não resolve o problema fundamental da **covariância**.

## Análise de Trade-offs

### Por que Random é Melhor?

1. **Sem viés de seleção**: Random sampling não introduz distorções sistemáticas
2. **Estatisticamente representativo**: Com k=1000 de n=34k, random sampling é teoricamente ótimo para preservar estatísticas de segunda ordem
3. **Simplicidade**: Não há complexidade adicional que possa introduzir erros

### Por que Global DDC Falha em Covariância?

1. **Seleção baseada em densidade**: Pode favorecer regiões de alta densidade, distorcendo a distribuição global
2. **Seleção baseada em diversidade**: Pode selecionar pontos extremos que não representam bem a covariância média
3. **Trade-off**: DDC otimiza para preservar distribuições marginais e cobertura espacial, mas não otimiza explicitamente para covariância

### Quando Global DDC Pode Ser Útil?

Apesar de ter métricas piores, Global DDC pode ser útil quando:

1. **Cobertura espacial é importante**: DDC garante melhor cobertura do espaço de features
2. **Outliers são importantes**: DDC pode capturar melhor pontos extremos
3. **Distribuições marginais complexas**: DDC pode preservar melhor distribuições não-Gaussianas
4. **Downstream tasks específicas**: Algumas tarefas podem se beneficiar da estrutura selecionada por DDC mesmo com covariância pior

## Comparação com Label-Aware DDC

Para contexto, comparando com resultados anteriores de Label-aware DDC:

| Método | Cov Error | Corr Error | Preserva Classes? |
|--------|-----------|------------|-------------------|
| Random | 0.33 | 0.21 | Sim (aproximadamente) |
| Stratified | 0.61 | 0.20 | Sim (exatamente) |
| Global DDC (optimized) | 1.64 | 0.15 | **Não** (distorce classes) |
| Label-aware DDC (optimized) | 1.78 | 0.61 | Sim (exatamente) |

**Observação**: 
- Global DDC tem **correlação melhor** que Label-aware DDC (0.15 vs 0.61)
- Mas Global DDC **distorce classes** (não apropriado para classificação)
- Label-aware DDC preserva classes mas tem métricas piores (trade-off aceitável para classificação)

## Conclusões

1. **Random sampling é difícil de superar** para preservar covariância global
2. **Global DDC otimizado melhora correlação** mas não resolve problema de covariância
3. **Trade-off fundamental**: DDC prioriza cobertura espacial sobre covariância exata
4. **Para classificação**: Label-aware DDC é preferível (preserva classes) mesmo com métricas piores
5. **Para EDA/unsupervised**: Random pode ser suficiente, mas DDC oferece melhor cobertura espacial

## Recomendações

### Quando Usar Cada Método

**Use Random quando**:
- Você precisa preservar covariância global exata
- O dataset é grande o suficiente para random sampling ser representativo
- Simplicidade é importante

**Use Stratified quando**:
- Você precisa preservar proporções de classe
- Covariância precisa ser preservada dentro de cada classe
- Classificação é a tarefa principal

**Use Global DDC quando**:
- Você precisa de melhor cobertura espacial
- Distribuições marginais complexas são importantes
- Outliers e pontos extremos são relevantes
- Tarefa é puramente unsupervised

**Use Label-aware DDC quando**:
- Você precisa preservar classes E distribuição conjunta
- Classificação é a tarefa principal
- Trade-off entre métricas e preservação de classes é aceitável

## Arquivos Gerados

- `examples/global_ddc_vs_baselines_comparison.csv` - Tabela completa de comparação
- `examples/GLOBAL_DDC_VS_BASELINES_COMPARISON.md` - Este relatório

