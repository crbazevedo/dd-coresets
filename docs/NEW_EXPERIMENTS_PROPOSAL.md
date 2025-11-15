
# Proposta: Novas Categorias de Experimentos para DDC

Baseado nas fortalezas identificadas do DDC, propomos as seguintes novas categorias:

## Categoria 7: Hierarchical Clusters

### 7.1 Nested Clusters (Clusters dentro de Clusters)
- **Objetivo**: Demonstrar DDC em estrutura hierárquica
- **Dataset**: Clusters grandes contendo sub-clusters
- **Hipótese**: DDC captura melhor estrutura hierárquica que Random
- **Métricas**: Cobertura por nível hierárquico, preservação de estrutura

### 7.2 Varying Cluster Separability
- **Objetivo**: Testar DDC com diferentes níveis de separação
- **Dataset**: Gaussian mixtures com separação variável (0.5σ a 5σ)
- **Hipótese**: DDC mantém vantagem mesmo com clusters menos separados
- **Métricas**: Cov error, W1, cobertura por cluster

## Categoria 8: Time Series / Sequential Data

### 8.1 Periodic Patterns
- **Objetivo**: Demonstrar DDC em dados temporais com padrões periódicos
- **Dataset**: Time series com múltiplos períodos (sine waves, seasonal patterns)
- **Hipótese**: DDC preserva melhor padrões temporais que Random
- **Métricas**: Preservação de frequências, cobertura temporal

### 8.2 Regime Changes
- **Objetivo**: Dados com mudanças de regime
- **Dataset**: Time series com múltiplos regimes (diferentes distribuições)
- **Hipótese**: DDC garante representação de todos os regimes
- **Métricas**: Cobertura por regime, preservação de transições

## Categoria 9: High-Dimensional Low-Intrinsic-Dimension

### 9.1 Manifolds em Alta Dimensão
- **Objetivo**: DDC em dados de alta dimensão mas baixa dimensão intrínseca
- **Dataset**: Dados em manifold (Swiss Roll em 10D, S-Curve em 10D)
- **Hipótese**: DDC funciona bem quando dimensão intrínseca é baixa
- **Métricas**: Preservação de geometria local, cobertura do manifold

### 9.2 Sparse High-Dimensional
- **Objetivo**: Dados esparsos em alta dimensão mas com estrutura
- **Dataset**: Dados esparsos com clusters bem definidos
- **Hipótese**: DDC pode funcionar se estrutura é clara apesar de esparsidade
- **Métricas**: Cobertura espacial, preservação de estrutura

## Categoria 10: Imbalanced but Structured

### 10.1 Rare but Important Clusters
- **Objetivo**: Clusters raros mas importantes (1% do dataset)
- **Dataset**: Mixture com 1 cluster raro mas crítico
- **Hipótese**: DDC garante representação de clusters raros
- **Métricas**: Presença de cluster raro, cobertura garantida

### 10.2 Long-Tail Distributions
- **Objetivo**: Distribuições com long tail (poucos pontos muito importantes)
- **Dataset**: Power-law distributions, Zipf distributions
- **Hipótese**: DDC pode capturar melhor estrutura de long tail
- **Métricas**: Cobertura de tail, preservação de estrutura

## Categoria 11: Multi-Scale Structures

### 11.1 Clusters de Diferentes Escalas
- **Objetivo**: Clusters com tamanhos muito diferentes mas todos importantes
- **Dataset**: Mixture com clusters de tamanhos 1:10:100
- **Hipótese**: DDC garante cobertura mesmo de clusters muito pequenos
- **Métricas**: Cobertura por cluster, preservação de proporções

### 11.2 Features com Diferentes Escalas
- **Objetivo**: Features com variâncias muito diferentes
- **Dataset**: Features com std variando de 0.1 a 10
- **Hipótese**: DDC adapta-se melhor a múltiplas escalas
- **Métricas**: Preservação por feature, normalização adequada

## Categoria 12: Real Datasets com Estrutura Clara

### 12.1 Image Datasets com Clusters Naturais
- **Objetivo**: Datasets de imagens com classes bem definidas
- **Dataset**: CIFAR-10 (10 classes), Caltech-101 (101 classes)
- **Hipótese**: DDC funciona melhor quando classes são bem separadas
- **Métricas**: Cobertura por classe, preservação de estrutura

### 12.2 Text Datasets com Tópicos
- **Objetivo**: Datasets de texto com tópicos claros
- **Dataset**: 20 Newsgroups, Reuters (após embedding)
- **Hipótese**: DDC preserva melhor estrutura de tópicos
- **Métricas**: Cobertura por tópico, preservação de distribuições

### 12.3 Genomic Data
- **Objetivo**: Dados genômicos com populações distintas
- **Dataset**: Dados de expressão gênica com populações diferentes
- **Hipótese**: DDC preserva melhor estrutura populacional
- **Métricas**: Cobertura por população, preservação de variância

## Priorização

### Alta Prioridade (Baseado em Fortalezas do DDC):

1. **Categoria 7.1: Nested Clusters** - Estrutura hierárquica é comum
2. **Categoria 10.1: Rare but Important Clusters** - Caso de uso prático importante
3. **Categoria 11.1: Multi-Scale Clusters** - Extensão de clusters desbalanceados
4. **Categoria 12.1: CIFAR-10** - Dataset real com estrutura clara (vs MNIST complexo)

### Média Prioridade:

5. **Categoria 8.1: Periodic Patterns** - Caso de uso específico
6. **Categoria 9.1: High-Dim Low-Intrinsic** - Validação de hipótese
7. **Categoria 12.2: Text Datasets** - Caso de uso prático

### Baixa Prioridade (Exploratório):

8. **Categoria 7.2: Varying Separability** - Validação de limites
9. **Categoria 8.2: Regime Changes** - Caso específico
10. **Categoria 9.2: Sparse High-Dim** - Pode não funcionar bem
11. **Categoria 10.2: Long-Tail** - Similar a skewed (pode falhar)
12. **Categoria 11.2: Multi-Scale Features** - Pode ser resolvido com scaling
13. **Categoria 12.3: Genomic** - Requer dataset específico

## Implementação Sugerida

Criar scripts para categorias de alta prioridade:
- `nested_clusters.py`
- `rare_clusters.py`
- `multi_scale_clusters.py`
- `cifar10_experiment.py`

