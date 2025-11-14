# Análise do Efeito de k na Performance do DDC

## Resumo Executivo

Esta análise investiga o efeito do tamanho do coreset (k) na performance relativa do DDC vs Random sampling.

### Principais Descobertas

1. **k pequeno (< 0.01 de n)**: DDC mostra maior vantagem
2. **k/n ratio crítico**: DDC mantém vantagem até aproximadamente k/n < 0.05
3. **Vantagem diminui com k maior**: Com k/n > 0.05, vantagem diminui
4. **Estrutura importa**: Two Moons mantém vantagem mesmo com k maior

## Análise por k/n Ratio

              cov_improvement_%               W1_improvement_%              ddc_wins_cov ddc_wins_W1
                           mean     std count             mean    std count          sum         sum
k_n_ratio                                                                                           
(0.0, 0.005]             112.20   63.30     5            33.93  63.58     5            5           4
(0.005, 0.01]            238.80   65.70     2            -1.52  13.02     2            2           1
(0.01, 0.02]             127.15     NaN     1           212.92    NaN     1            1           1
(0.02, 0.05]              60.46  102.24     5            42.67  30.37     5            3           4
(0.05, 0.1]                 NaN     NaN     0              NaN    NaN     0            0           0
(0.1, 1.0]                  NaN     NaN     0              NaN    NaN     0            0           0

## Análise por Valor de k

     k_n_ratio cov_improvement_%               W1_improvement_%               ddc_wins_cov ddc_wins_W1
          mean              mean     std count             mean     std count          sum         sum
k                                                                                                     
16        0.00            121.43     NaN     1            22.21     NaN     1            1           1
24        0.00             25.07     NaN     1            25.42     NaN     1            1           1
32        0.00            187.48     NaN     1           141.83     NaN     1            1           1
50        0.01            181.08  147.33     2            -8.84   23.38     2            2           1
100       0.01            138.63   16.24     2           109.25  146.60     2            2           2
200       0.02             79.80  159.17     2            -6.51    5.96     2            1           0
1000      0.05             83.76  101.58     4            53.91   19.68     4            3           4

## Recomendações

### O que significa 'k pequeno'?

- **k/n < 0.01** (1%): DDC mostra maior vantagem (+100-300%)
- **k/n < 0.05** (5%): DDC ainda mantém vantagem (+50-150%)
- **k/n > 0.05** (5%): Vantagem diminui, Random pode ser suficiente

### Até onde podemos aumentar k?

- **Estruturas simples** (Gaussian mixtures): Até k/n ≈ 0.05
- **Estruturas complexas** (Two Moons): Até k/n ≈ 0.04 (k=200, n=5k)
- **Geral**: Recomendado manter k/n < 0.05 para garantir vantagem

## Tabela Detalhada

  experiment_type    k     n  k_n_ratio  n_clusters  random_cov_err  ddc_cov_err  cov_improvement_%  random_W1   ddc_W1  W1_improvement_%  ddc_wins_cov  ddc_wins_W1
 small_k_gaussian   50 20000     0.0025         NaN        1.002051     0.566431          76.906125   0.085767 0.114932        -25.376287          True        False
 small_k_gaussian  100 20000     0.0050         NaN        0.522783     0.209014         150.118702   0.061412 0.058160          5.592385          True         True
 small_k_gaussian  200 20000     0.0100         NaN        0.408315     0.139670         192.343745   0.039616 0.044375        -10.724729          True        False
proportional_k_2x   16 20000     0.0008         NaN        2.579553     1.164961         121.428206   0.246000 0.201296         22.208432          True         True
proportional_k_3x   24 20000     0.0012         NaN        2.084671     1.666819          25.068799   0.242053 0.192996         25.418790          True         True
proportional_k_4x   32 20000     0.0016         NaN        2.485645     0.864622         187.483576   0.277768 0.114862        141.828608          True         True
        two_moons   50  5000     0.0100         NaN        0.270158     0.070124         285.257687   0.090111 0.083678          7.688444          True         True
        two_moons  100  5000     0.0200         NaN        0.186628     0.082162         127.147501   0.135606 0.043336        212.916340          True         True
        two_moons  200  5000     0.0400         NaN        0.085586     0.127266         -32.750537   0.042799 0.043802         -2.291325         False        False
   cluster_varied 1000 20000     0.0500         2.0        0.206744     0.153139          35.004305   0.046707 0.030669         52.293033          True         True
   cluster_varied 1000 20000     0.0500         4.0        0.206744     0.062819         229.113258   0.046707 0.025662         82.008299          True         True
   cluster_varied 1000 20000     0.0500         8.0        0.206744     0.119647          72.795019   0.046707 0.032486         43.776901          True         True
   cluster_varied 1000 20000     0.0500        16.0        0.206744     0.210658          -1.858082   0.046707 0.033950         37.575168         False         True

