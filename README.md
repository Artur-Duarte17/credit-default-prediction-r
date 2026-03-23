# Credit Default Prediction in R

## Visao geral

Este repositorio implementa um pipeline completo de credit scoring em R para prever inadimplencia em cartao de credito. O objetivo nao e apenas maximizar uma metrica estatistica isolada, mas construir um processo reproduzivel, comparavel entre modelos e metodologicamente defensavel para selecao de variaveis, calibracao de threshold, avaliacao final em teste e interpretabilidade.

O desenho atual segue um protocolo em duas camadas:

- Fase exploratoria: mais barata, usada para filtrar cenarios promissores sem enfraquecer a comparacao.
- Fase de confirmacao: mais rigorosa, aplicada apenas aos finalistas que sobreviveram a exploracao.

Essa divisao reduz custo computacional pela diminuicao inteligente de cenarios, e nao por usar menos rigor na etapa decisiva.

## Problema de negocio

O problema central e identificar clientes com maior risco de default antes da concessao ou renovacao de credito. Em pratica:

- Falso negativo: o modelo classifica um inadimplente como bom pagador.
- Falso positivo: o modelo barra um bom pagador por excesso de cautela.

Em credito, falso negativo costuma ser o erro mais caro, porque implica exposicao real a perda. Por isso a analise final destaca sensibilidade, sem abandonar ROC, PRAUC, F1, especificidade e metricas de negocio.

## Dataset e variavel alvo

- Fonte de dados local: `dados/default_of_credit_card_clients.xls`
- Base limpa derivada: `dados/dados_limpos.csv`
- Variavel alvo no pipeline: `Class`
- Convencao da classe positiva: `Deve`
- Convencao da classe negativa: `Pago`

O script `01_dados.R` renomeia as colunas, remove `ID` e recodifica a resposta para a ordem `Deve`, `Pago`, o que padroniza a classe positiva em todo o pipeline.

## Estrategia metodologica

### Fase exploratoria

A fase exploratoria usa validacao cruzada estratificada mais barata:

- `CV_FOLDS_EXPLORATORIO <- 5`
- `CV_REPEATS_EXPLORATORIO <- 1`

Objetivo:

- mapear saturacao de variaveis em GLM e XGBoost;
- reduzir o espaco de busca para RF, SVM radial, NNET e avNNet;
- selecionar subconjuntos e configuracoes candidatas para confirmacao.

Politica por modelo:

- GLM e XGBoost continuam como guias principais da exploracao Top-N completa.
- RF, SVM radial, NNET e avNNet nao rodam Top-1 ate Top-N inteiro por padrao.
- Esses modelos passam a usar apenas `TopN` candidatos guiados por evidencia.

Como os `TopN` candidatos sao escolhidos:

1. Ler os melhores `TopN` ja encontrados por GLM e XGBoost, se os arquivos de curva existirem.
2. Formar a uniao de:
   `melhor TopN do GLM`, `melhor TopN do XGBoost`, e vizinhos `-2`, `-1`, `+1`, `+2`.
3. Manter apenas valores validos e unicos.
4. Se os arquivos ainda nao existirem, usar fallback explicito `c(10, 13, 14)`.

Essa regra reduz custo de RF/SVM/redes sem perder o papel analitico de GLM e XGBoost como detectores de saturacao.

Grids exploratorios padrao:

- `SVM_Radial`: 3 valores de `sigma` e `C = c(0.5, 1, 2)`
- `NNET`: `size = c(3, 5)` e `decay = c(0.001, 0.01)`
- `avNNet`: mesma grade enxuta da `NNET`
- `RF`: grade pequena baseada em `mtry`
- `XGBoost`: grade controlada, sem early stopping

### Fase de confirmacao

A confirmacao usa validacao cruzada estratificada mais rigorosa:

- `CV_FOLDS_CONFIRMACAO <- 5`
- `CV_REPEATS_CONFIRMACAO <- 2`

Objetivo:

- reavaliar apenas os finalistas sem balanceamento;
- testar `SMOTENC` apenas nos melhores subconjuntos confirmados;
- levar para calibracao de threshold apenas um cenario final confirmado por modelo.

Na confirmacao:

- os mesmos folds sao reutilizados entre modelos sempre que o conjunto de treino e o mesmo;
- modelos caret e XGBoost customizado passam a usar a mesma logica de validacao;
- `SMOTENC` continua restrito ao treino de cada fold ou ao treino final, nunca ao conjunto de teste.

## Como evitar data leakage

O pipeline foi estruturado para evitar vazamento de informacao em pontos criticos:

- O conjunto de teste final fica totalmente intocado ate `08_avaliacao_teste.R`.
- Threshold e calibrado em `07_threshold_calibracao.R` usando predicoes OOF do treino, nunca o teste.
- `SMOTENC` e aplicado apenas dentro do treino de cada fold ou no treino final.
- Na validacao com XGBoost customizado, as transformacoes sao ajustadas no treino do fold e aplicadas no fold de validacao sem reusar informacao futura.
- A comparacao entre modelos usa os mesmos folds estratificados por fase sempre que os modelos enxergam o mesmo conjunto de treino.

## Sensibilidade e especificidade

- Sensibilidade: proporcao de inadimplentes reais (`Deve`) corretamente identificados pelo modelo.
- Especificidade: proporcao de bons pagadores (`Pago`) corretamente preservados pelo modelo.

Leitura pratica:

- alta sensibilidade reduz falso negativo;
- alta especificidade reduz falso positivo.

Em credito, a sensibilidade recebe prioridade analitica porque deixar passar um inadimplente costuma gerar perda financeira mais grave do que rejeitar indevidamente um bom pagador. Ainda assim, a decisao final continua acompanhando ROC, PRAUC, F1, GMean, acuracia e metricas de negocio.

## Pipeline completo

### 1. Preparacao e selecao de variaveis

1. `00_setup.R`
   Carrega pacotes, cria pastas e define seeds, splits, grids e parametros de validacao.
2. `01_dados.R`
   Le a base bruta, limpa colunas e define a resposta `Class`.
3. `02_preprocessamento.R`
   Ajusta categorias, tipos e salva splits estratificados 70/30 e 80/20.
4. `03_selecao_variaveis.R`
   Gera o ranking inicial de variaveis com Elastic Net no treino.

### 2. Exploracao Top-N

5. `04_topn_base.R`
   Curva Top-N completa para GLM em fase exploratoria.
6. `04B_topn_xgboost.R`
   Curva Top-N completa para XGBoost em fase exploratoria.
7. `04C_topn_modelos_caret.R`
   Curvas Top-N candidatas para RF, SVM radial, NNET e avNNet.

### 3. Benchmark sem balanceamento

8. `05A_glm_rf_subconjuntos.R`
   Exploracao e confirmacao sem balanceamento para GLM e RF.
9. `05B_xgboost_subconjuntos.R`
   Exploracao e confirmacao sem balanceamento para XGBoost.
10. `05C_svm_subconjuntos.R`
    Exploracao e confirmacao sem balanceamento para SVM radial.
11. `05D_redes_neurais_subconjuntos.R`
    Exploracao e confirmacao sem balanceamento para NNET e avNNet.

### 4. Confirmacao com SMOTENC

12. `06_rf_balanceamento_smotenc.R`
    Compara RF com e sem `SMOTENC` no melhor subconjunto confirmado do modelo.
13. `06B_xgb_balanceamento_smotenc.R`
    Compara XGBoost com e sem `SMOTENC` no melhor subconjunto confirmado.
14. `06C_modelos_caret_balanceamento_smotenc.R`
    Compara SVM radial, NNET e avNNet com e sem `SMOTENC` nos respectivos finalistas.

### 5. Calibracao, teste e interpretabilidade

15. `07_threshold_calibracao.R`
    Calibra threshold apenas nos finalistas confirmados usando predicoes OOF.
16. `08_avaliacao_teste.R`
    Treina os finalistas no treino completo e avalia no teste intocado.
17. `09_shap.R`
    Gera interpretabilidade SHAP para o modelo vencedor e compara com o ranking Elastic Net.
18. `10_resumo_final.R`
    Consolida benchmark, teste, negocio e interpretabilidade.

## Ordem sugerida de execucao

Ordem recomendada em uma rodada completa:

1. `00_setup.R`
2. `01_dados.R`
3. `02_preprocessamento.R`
4. `03_selecao_variaveis.R`
5. `04_topn_base.R`
6. `04B_topn_xgboost.R`
7. `04C_topn_modelos_caret.R`
8. `05A_glm_rf_subconjuntos.R`
9. `05B_xgboost_subconjuntos.R`
10. `05C_svm_subconjuntos.R`
11. `05D_redes_neurais_subconjuntos.R`
12. `06_rf_balanceamento_smotenc.R`
13. `06B_xgb_balanceamento_smotenc.R`
14. `06C_modelos_caret_balanceamento_smotenc.R`
15. `07_threshold_calibracao.R`
16. `08_avaliacao_teste.R`
17. `09_shap.R`
18. `10_resumo_final.R`

## Splits de treino e teste

- Split padrao do projeto: `SPLIT_TREINO_PADRAO <- 0.70`
- Split alternativo salvo para analise de robustez: `0.80`

O split 70/30 permanece como padrao operacional do pipeline. O split 80/20 e mantido como opcao de robustez, sem substituir o fluxo principal.

## Outputs gerados

### `objetos/`

Armazena artefatos `RDS` usados entre scripts:

- bases limpas e preprocessadas;
- splits de treino e teste;
- curvas Top-N;
- tabelas de benchmark exploratorio e confirmado;
- configuracoes finais;
- resultados de threshold;
- tabelas finais de teste;
- objetos auxiliares de interpretabilidade.

### `resultados/`

Armazena tabelas `CSV` para consumo analitico e auditoria:

- rankings de variaveis;
- curvas e tabelas Top-N;
- benchmarks sem balanceamento;
- comparacoes com `SMOTENC`;
- thresholds calibrados;
- resultados finais de teste;
- tabelas SHAP e comparacao com Elastic Net.

### `figuras/`

Armazena graficos exportados:

- curvas ROC;
- comparacoes por subconjunto e modelo;
- comparacoes de threshold;
- desempenho final em teste;
- visualizacoes SHAP.

## Nota metodologica sobre SMOTENC

`SMOTENC` foi mantido porque a base mistura variaveis numericas e categoricas. A implementacao usa `recipes` e `themis::step_smotenc()` com `skip = TRUE`, de modo que:

- o rebalanceamento acontece apenas no treino de cada fold;
- o fold de validacao permanece na distribuicao original;
- o conjunto de teste final nao recebe oversampling.

Esse detalhe e essencial para manter uma estimativa honesta de desempenho.

## Nota metodologica sobre interpretabilidade com SHAP

O script `09_shap.R` interpreta apenas o modelo vencedor apos a avaliacao final em teste. A comparacao SHAP vs Elastic Net tem papeis distintos:

- Elastic Net fornece um ranking inicial linear para triagem de variaveis;
- SHAP explica a distribuicao de contribuicoes no modelo final, inclusive nao linearidades e interacoes.

Por isso divergencias entre o ranking SHAP e o ranking do Elastic Net sao esperadas e fazem parte da leitura analitica, nao de um erro do pipeline.

## Resumo do protocolo atual

- GLM e XGBoost exploram Top-N completo e guiam a reducao de cenarios.
- RF, SVM radial, NNET e avNNet usam TopN candidatos por padrao.
- A comparacao entre modelos reutiliza folds compartilhados por fase.
- A confirmacao forte fica concentrada apenas nos finalistas.
- Threshold e calibrado em OOF, nao no teste.
- O teste final permanece intocado ate a etapa final.
