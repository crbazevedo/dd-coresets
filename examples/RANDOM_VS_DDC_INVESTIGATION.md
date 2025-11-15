# Investigação: Por que Random às vezes supera Global DDC?

## Resumo Executivo

Testamos Global DDC vs Random sampling em 6 datasets diferentes. **Random venceu em 4/6 casos**, incluindo o Adult Census Income. DDC venceu apenas em datasets com estrutura de clusters bem definida.

**Resultado principal**: Random sampling é estatisticamente superior para preservar covariância global quando:
- Há correlações fortes entre features (anisotropic)
- Alta dimensionalidade com poucas features informativas (sparse)
- Datasets reais com estrutura complexa
- Datasets grandes (n >> k)

DDC funciona melhor quando:
- Estrutura de clusters bem definida (Gaussian mixtures)
- Distribuições isotrópicas simples

## Resultados Detalhados

| Dataset | n_samples | n_features | Winner | Random Score | DDC Opt Score | Diferença |
|---------|-----------|------------|--------|--------------|---------------|-----------|
| **Gaussian (isotropic)** | 20,000 | 10 | **DDC** | 0.5525 | 0.3766 | **-32%** |
| **Gaussian Mixture (4)** | 20,000 | 10 | **DDC** | 0.3066 | 0.0818 | **-73%** |
| Anisotropic Gaussian | 20,000 | 10 | Random | 0.4297 | 0.4987 | +16% |
| High-dim Sparse | 20,000 | 20 | Random | 0.9527 | 2.6588 | +179% |
| **Adult Census** | 48,842 | 6 | Random | 0.3802 | 1.4699 | +287% |
| Large Gaussian Mixture | 100,000 | 10 | Random | 0.1350 | 0.1549 | +15% |

**Score = Cov Error + 0.5 × Corr Error** (menor é melhor)

## Análise por Dataset

### 1. Gaussian (isotropic) - DDC VENCE ✅

**Características**: Distribuição simples, isotrópica, sem correlações.

**Resultados**:
- Random: Cov=0.384, Corr=0.336
- DDC optimized: Cov=0.297 (-23%), Corr=0.160 (-52%)

**Por que DDC vence?**
- Estrutura simples permite que DDC selecione representantes que preservam bem a distribuição
- Sem correlações complexas, a seleção baseada em densidade-diversidade funciona bem

### 2. Gaussian Mixture (4) - DDC VENCE ✅

**Características**: 4 clusters bem separados, estrutura multimodal clara.

**Resultados**:
- Random: Cov=0.230, Corr=0.152
- DDC optimized: Cov=0.063 (-73%), Corr=0.038 (-75%)

**Por que DDC vence?**
- **Estrutura de clusters bem definida** permite que DDC capture melhor a diversidade espacial
- DDC seleciona representantes de cada cluster, preservando melhor a estrutura multimodal
- Random pode sub-amostrar alguns clusters

### 3. Anisotropic Gaussian - RANDOM VENCE ❌

**Características**: Correlações fortes entre features (primeiras 5 features altamente correlacionadas).

**Resultados**:
- Random: Cov=0.298, Corr=0.263
- DDC optimized: Cov=0.432 (+45%), Corr=0.134 (-49%)

**Por que Random vence?**
- **Correlações fortes** são melhor preservadas por random sampling estatístico
- DDC seleciona pontos baseado em densidade-diversidade, o que pode **distorcer a estrutura de covariância** quando há correlações
- DDC melhora correlação relativa mas piora covariância absoluta

### 4. High-dim Sparse - RANDOM VENCE ❌

**Características**: 20 features, apenas 5 informativas, alta dimensionalidade esparsa.

**Resultados**:
- Random: Cov=0.648, Corr=0.609
- DDC optimized: Cov=2.321 (+258%), Corr=0.675 (+11%)

**Por que Random vence?**
- **Alta dimensionalidade esparsa** dificulta seleção baseada em densidade
- Poucas features informativas: random sampling preserva melhor a estrutura estatística global
- DDC pode estar selecionando pontos em subespaços que não representam bem a covariância global

### 5. Adult Census Income - RANDOM VENCE ❌

**Características**: Dataset real, 6 features numéricas, estrutura complexa não-Gaussiana.

**Resultados**:
- Random: Cov=0.299, Corr=0.162
- DDC optimized: Cov=1.400 (+368%), Corr=0.140 (-14%)

**Por que Random vence?**
- **Estrutura complexa não-Gaussiana** não é bem capturada por seleção baseada em densidade-diversidade
- Features podem ter correlações não-lineares que DDC não preserva bem
- Random sampling é estatisticamente não-viesado para preservar covariância global

### 6. Large Gaussian Mixture - RANDOM VENCE ❌

**Características**: 100,000 amostras, 4 clusters, n >> k (100x).

**Resultados**:
- Random: Cov=0.109, Corr=0.052
- DDC optimized: Cov=0.117 (+8%), Corr=0.075 (+43%)

**Por que Random vence?**
- **n >> k**: Com 100k amostras e k=1000, random sampling é estatisticamente muito representativo
- Lei dos grandes números: random sampling converge para a distribuição verdadeira
- DDC pode introduzir viés desnecessário quando há dados suficientes

## Padrões Identificados

### Quando Random é Melhor:

1. **Correlações fortes entre features**
   - Random preserva melhor estrutura de covariância quando há dependências lineares
   - DDC pode distorcer covariância ao selecionar pontos baseado em densidade

2. **Alta dimensionalidade esparsa**
   - Random é estatisticamente superior em espaços de alta dimensão
   - DDC pode falhar em capturar estrutura global em subespaços esparsos

3. **Datasets reais com estrutura complexa**
   - Estruturas não-Gaussianas não são bem capturadas por DDC
   - Random é não-viesado para preservar estatísticas de segunda ordem

4. **n >> k (muitos dados)**
   - Com dados suficientes, random sampling converge para distribuição verdadeira
   - DDC pode introduzir viés desnecessário

### Quando DDC é Melhor:

1. **Estrutura de clusters bem definida**
   - DDC captura melhor diversidade espacial em datasets multimodais
   - Seleção baseada em densidade-diversidade preserva melhor clusters separados

2. **Distribuições isotrópicas simples**
   - Sem correlações complexas, DDC pode selecionar representantes mais eficientes
   - Melhor cobertura espacial que random

## Explicação Teórica

### Por que Random preserva melhor covariância?

**Teorema Central do Limite**: Com n grande e k razoável, random sampling produz uma amostra estatisticamente representativa que converge para a distribuição verdadeira.

**Não-viés**: Random sampling é **não-viesado** para estimar covariância:
```
E[cov(X_random)] = cov(X_full)
```

**DDC introduz viés**: A seleção baseada em densidade-diversidade pode introduzir viés sistemático:
```
E[cov(X_ddc)] ≠ cov(X_full)
```

O viés ocorre porque:
1. DDC seleciona pontos em regiões de alta densidade
2. Isso pode distorcer a covariância global se a densidade não é uniforme
3. A diversidade pode selecionar pontos extremos que não representam bem a covariância média

### Por que DDC funciona bem em clusters?

1. **Cobertura espacial**: DDC garante representantes de cada cluster
2. **Diversidade**: Seleção baseada em diversidade captura melhor estrutura multimodal
3. **Densidade**: Seleção baseada em densidade captura melhor o centro de cada cluster

## Implicações Práticas

### Quando usar Random:

✅ **Use Random quando**:
- Você precisa preservar covariância global exata
- O dataset é grande (n >> k)
- Há correlações fortes entre features
- Alta dimensionalidade esparsa
- Dataset real com estrutura complexa não-Gaussiana

### Quando usar DDC:

✅ **Use DDC quando**:
- Estrutura de clusters bem definida
- Você precisa de melhor cobertura espacial
- Distribuições marginais complexas são importantes
- Outliers e pontos extremos são relevantes
- Tarefa é puramente unsupervised com estrutura multimodal

### Para Adult Census Income especificamente:

**Conclusão**: Random é melhor porque:
1. Dataset real com estrutura complexa não-Gaussiana
2. Features podem ter correlações não-lineares
3. n=48k é grande o suficiente para random sampling ser estatisticamente representativo
4. DDC introduz viés ao selecionar pontos baseado em densidade-diversidade

## Recomendações

1. **Para preservar covariância global**: Use Random ou Stratified
2. **Para cobertura espacial e clusters**: Use DDC
3. **Para classificação**: Use Label-aware DDC (preserva classes)
4. **Para EDA/unsupervised**: Avalie trade-off entre covariância (Random) e cobertura (DDC)

## Próximos Passos

1. Investigar se DDC pode ser modificado para preservar melhor covariância
2. Testar DDC com diferentes métricas de distância
3. Comparar DDC com outros métodos de coreset (k-means, k-medoids)
4. Investigar se o problema está na seleção ou no reweighting

## Arquivos Gerados

- `examples/random_vs_ddc_investigation.csv` - Resultados completos
- `examples/RANDOM_VS_DDC_INVESTIGATION.md` - Este relatório

