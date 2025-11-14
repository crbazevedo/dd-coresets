# Análise de Falhas do DDC

Este relatório investiga por que DDC falha em certos cenários.

## 1. Distribuições Skewed/Heavy-tailed

### Problema Identificado

DDC falha em preservar caudas de distribuições skewed/heavy-tailed.

### Causa Provável

1. **Seleção baseada em densidade**: DDC seleciona pontos em regiões de alta densidade
2. **Caudas têm baixa densidade**: Outliers e caudas são sub-representados
3. **Diversidade não compensa**: Mesmo com diversidade, DDC não captura bem regiões esparsas
4. **Pesos não corrigem**: Os pesos não conseguem compensar a falta de pontos nas caudas

### Solução Proposta

1. **Ajustar alpha**: Reduzir alpha para dar mais peso à diversidade
2. **Aumentar m_neighbors**: Melhorar estimativa de densidade local
3. **Pré-processamento**: Transformar para distribuição mais simétrica
4. **Híbrido**: Combinar DDC com amostragem de caudas

## 2. Datasets Reais Complexos (MNIST, Fashion-MNIST)

### Problema Identificado

DDC tem covariância muito pior (-75% a -80%) em datasets de imagens.

### Causa Provável

1. **Estrutura não-Gaussiana complexa**: Distribuições muito diferentes de Gaussian
2. **Alta dimensionalidade**: Mesmo após PCA, estrutura é complexa
3. **Múltiplas escalas**: Diferentes dígitos têm diferentes variâncias
4. **Correlações não-lineares**: DDC não captura bem dependências não-lineares
5. **Viés de seleção**: Seleção baseada em densidade distorce covariância global

### Solução Proposta

1. **Label-aware DDC**: Aplicar DDC por classe separadamente
2. **Pré-clustering**: Aplicar DDC dentro de clusters pré-definidos
3. **Parâmetros específicos**: Ajustar alpha, gamma para estrutura específica
4. **Usar Random**: Para datasets muito complexos, Random pode ser melhor

## 3. Anéis Concêntricos

### Problema Identificado

DDC falha em estruturas de anéis concêntricos (-77% covariância, -78% W1).

### Causa Provável

1. **Estrutura muito específica**: Anéis são estruturas não-convexas muito específicas
2. **Densidade uniforme**: Dentro de cada anel, densidade é uniforme
3. **Seleção concentrada**: DDC pode concentrar seleção em alguns anéis
4. **Geometria não-linear**: DDC não captura bem geometria circular

### Solução Proposta

1. **Aumentar diversidade**: Reduzir alpha, aumentar gamma
2. **Estratificação por anel**: Aplicar DDC por anel separadamente
3. **Métrica de distância adaptada**: Usar distância angular para anéis
4. **Usar Random**: Para estruturas muito específicas, Random pode ser melhor

## Recomendações Gerais

### Quando DDC Falha

1. **Distribuições com caudas importantes**: Use Random ou híbrido
2. **Datasets muito complexos**: Use Random ou Label-aware DDC
3. **Estruturas muito específicas**: Teste primeiro, pode precisar de Random

### Estratégias de Mitigação

1. **Pré-análise**: Verificar estrutura antes de escolher método
2. **Híbrido**: Combinar DDC com Random para caudas
3. **Label-aware**: Usar DDC por classe/cluster quando aplicável
4. **Parâmetros adaptativos**: Ajustar parâmetros baseado em características do dataset

