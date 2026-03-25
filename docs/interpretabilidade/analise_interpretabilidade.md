# Analise SHAP vs Elastic Net

Cenario interpretado: NNET_Top19_SemBalanceamento_Youden
Modelo: NNET

Leitura analitica:
- O ranking Elastic Net reflete principalmente relacoes lineares e marginais.
- O ranking SHAP do modelo vencedor incorpora nao linearidades, interacoes e redistribuicao de importancia entre variaveis correlacionadas.
- Diferencas grandes de posicao sao esperadas quando o modelo final aprende regras mais complexas do que o modelo linear usado no ranking inicial.
- PAY_4: Elastic Net em 19 e SHAP em 7.
- EDUCATION: Elastic Net em 5 e SHAP em 15.
- PAY_AMT1: Elastic Net em 4 e SHAP em 11.
