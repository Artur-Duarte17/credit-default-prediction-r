# ==============================================================================
# 09_shap.R
# Responsabilidade: gerar SHAP para o modelo vencedor e comparar com Elastic Net.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar dados e identificar modelo vencedor
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(ler_rds_base("treino.rds"))
teste <- garantir_ordem_classe(ler_rds_base("teste.rds"))
tabela_teste <- ler_rds_saida(
  "final",
  "tabela_teste_final.rds",
  subpastas = "teste",
  legados = caminho_objeto_legado("tabela_teste_final.rds")
)
config_modelos <- ler_rds_saida(
  "final",
  "config_modelos_finais.rds",
  subpastas = "threshold",
  legados = caminho_objeto_legado("config_modelos_finais.rds")
)
ranking_variaveis <- ler_rds_base("ranking_variaveis_enet.rds")

melhor_cenario_teste <- tabela_teste %>%
  dplyr::slice(1)

cenario_base <- sub("_Youden$", "", melhor_cenario_teste$Cenario[1])
config_modelo <- config_modelos %>%
  dplyr::filter(Cenario == cenario_base) %>%
  dplyr::slice(1)

vars_modelo <- parse_variaveis(config_modelo$Variaveis[1])

print(melhor_cenario_teste)
print(config_modelo)
print(vars_modelo)

# ------------------------------------------------------------------------------
# BLOCO 2 - Treinar novamente o modelo vencedor
# ------------------------------------------------------------------------------
ajuste_final <- treinar_prever_modelo_final(
  config_modelo = config_modelo,
  treino_df = treino,
  teste_df = teste
)

if (config_modelo$Modelo[1] == "XGBoost") {
  x_treino_bg <- ajuste_final$x_treino
  x_teste_full <- ajuste_final$x_teste
  coluna_dependencia <- if ("PAY_0" %in% colnames(x_treino_bg)) "PAY_0" else colnames(x_treino_bg)[1]

  pred_wrapper_modelo <- function(object, newdata) {
    newdata <- as.matrix(newdata)
    newdata <- alinhar_colunas_matriz(x_treino_bg, newdata)
    predict(object, newdata = newdata)
  }
} else {
  x_treino_bg <- ajuste_final$treino[, vars_modelo, drop = FALSE]
  x_teste_full <- ajuste_final$teste[, vars_modelo, drop = FALSE]
  coluna_dependencia <- if ("PAY_0" %in% names(x_treino_bg)) "PAY_0" else names(x_treino_bg)[1]

  pred_wrapper_modelo <- function(object, newdata) {
    newdata <- as.data.frame(newdata)
    predict(object, newdata = newdata, type = "prob")[, "Deve"]
  }
}

# ------------------------------------------------------------------------------
# BLOCO 3 - Definir amostra para SHAP
# ------------------------------------------------------------------------------
set.seed(SEED_PROJETO)
n_amostra_shap <- min(200, nrow(x_teste_full))
ids_amostra <- sample(seq_len(nrow(x_teste_full)), size = n_amostra_shap)

x_teste_shap <- x_teste_full[ids_amostra, , drop = FALSE]
y_teste_shap <- ajuste_final$teste$Class[ids_amostra]
threshold_final <- melhor_cenario_teste$Threshold[1]

# ------------------------------------------------------------------------------
# BLOCO 4 - Calcular SHAP
# ------------------------------------------------------------------------------
set.seed(SEED_PROJETO)
shap_raw <- fastshap::explain(
  object = ajuste_final$model,
  X = x_treino_bg,
  pred_wrapper = pred_wrapper_modelo,
  newdata = x_teste_shap,
  nsim = 30,
  adjust = TRUE,
  shap_only = FALSE
)

sv <- shapviz::shapviz(shap_raw)
matriz_shap <- shap_raw$shapley_values

# ------------------------------------------------------------------------------
# BLOCO 5 - Importancia global e agregacao
# ------------------------------------------------------------------------------
tabela_shap <- tibble::tibble(
  Variavel = colnames(matriz_shap),
  Importancia_SHAP = colMeans(abs(matriz_shap))
) %>%
  dplyr::arrange(desc(Importancia_SHAP)) %>%
  dplyr::mutate(Posicao = dplyr::row_number())

if (config_modelo$Modelo[1] == "XGBoost") {
  tabela_shap_agregada <- tabela_shap %>%
    dplyr::mutate(
      Variavel_Original = purrr::map_chr(
        Variavel,
        mapear_variavel_original,
        nomes_originais = vars_modelo
      )
    ) %>%
    dplyr::group_by(Variavel_Original) %>%
    dplyr::summarise(
      Importancia_SHAP = sum(Importancia_SHAP),
      .groups = "drop"
    ) %>%
    dplyr::arrange(desc(Importancia_SHAP)) %>%
    dplyr::mutate(Posicao = dplyr::row_number()) %>%
    dplyr::rename(Variavel = Variavel_Original)
} else {
  tabela_shap_agregada <- tabela_shap
}

top10_shap <- tabela_shap_agregada %>%
  dplyr::slice(1:10)

tabela_shap_vs_enet <- criar_tabela_shap_vs_enet(
  tabela_shap = tabela_shap_agregada,
  ranking_enet = ranking_variaveis
) %>%
  dplyr::arrange(dplyr::desc(abs(Delta_Posicao)), dplyr::coalesce(Posicao_SHAP, Posicao_ElasticNet))

top_diferencas <- tabela_shap_vs_enet %>%
  dplyr::filter(!is.na(Posicao_SHAP), !is.na(Posicao_ElasticNet)) %>%
  dplyr::slice(1:10)

metadata_shap <- tibble::tibble(
  Cenario = melhor_cenario_teste$Cenario,
  Modelo = config_modelo$Modelo,
  Usa_SMOTENC = config_modelo$Usa_SMOTENC,
  Threshold = threshold_final
)

print(top10_shap)
print(top_diferencas)

# ------------------------------------------------------------------------------
# BLOCO 6 - Graficos SHAP
# ------------------------------------------------------------------------------
grafico_shap_importancia <- shapviz::sv_importance(
  sv,
  kind = "bar",
  max_display = 10
)

grafico_shap_bee <- shapviz::sv_importance(
  sv,
  kind = "bee",
  max_display = 10
)

grafico_dependencia <- shapviz::sv_dependence(
  sv,
  v = coluna_dependencia
)

classe_prevista <- ifelse(pred_wrapper_modelo(ajuste_final$model, x_teste_shap) >= threshold_final, "Deve", "Pago")
id_caso_deve <- which(classe_prevista == "Deve")[1]

if (!is.na(id_caso_deve)) {
  grafico_waterfall <- shapviz::sv_waterfall(
    sv,
    row_id = id_caso_deve,
    max_display = 10
  )
} else {
  grafico_waterfall <- NULL
}

print(grafico_shap_importancia)
print(grafico_shap_bee)
print(grafico_dependencia)

# ------------------------------------------------------------------------------
# BLOCO 7 - Escrever interpretacao resumida
# ------------------------------------------------------------------------------
linhas_interpretacao <- c(
  "# Analise SHAP vs Elastic Net",
  "",
  paste("Cenario interpretado:", melhor_cenario_teste$Cenario[1]),
  paste("Modelo:", config_modelo$Modelo[1]),
  "",
  "Leitura analitica:",
  "- O ranking Elastic Net reflete principalmente relacoes lineares e marginais.",
  "- O ranking SHAP do modelo vencedor incorpora nao linearidades, interacoes e redistribuicao de importancia entre variaveis correlacionadas.",
  "- Diferencas grandes de posicao sao esperadas quando o modelo final aprende regras mais complexas do que o modelo linear usado no ranking inicial."
)

if (nrow(top_diferencas) > 0) {
  for (j in seq_len(min(3, nrow(top_diferencas)))) {
    linhas_interpretacao <- c(
      linhas_interpretacao,
      paste0(
        "- ",
        top_diferencas$Variavel[j],
        ": Elastic Net em ",
        top_diferencas$Posicao_ElasticNet[j],
        " e SHAP em ",
        top_diferencas$Posicao_SHAP[j],
        "."
      )
    )
  }
}

salvar_texto_saida(
  linhas_interpretacao,
  "analise_interpretabilidade.md",
  subpastas = "interpretabilidade"
)

# ------------------------------------------------------------------------------
# BLOCO 8 - Salvar resultados
# ------------------------------------------------------------------------------
salvar_rds_saida(tabela_shap_agregada, "interpretabilidade", "tabela_shap_final.rds", subpastas = "shap")
salvar_csv_saida(tabela_shap_agregada, "interpretabilidade", "tabela_shap_final.csv", subpastas = "shap")
salvar_csv_saida(top10_shap, "interpretabilidade", "top10_shap_final.csv", subpastas = "shap")

salvar_rds_saida(
  tabela_shap_vs_enet,
  "interpretabilidade",
  "tabela_shap_vs_elastic_net.rds",
  subpastas = "shap"
)
salvar_csv_saida(
  tabela_shap_vs_enet,
  "interpretabilidade",
  "tabela_shap_vs_elastic_net.csv",
  subpastas = "shap"
)

salvar_rds_saida(
  metadata_shap,
  "interpretabilidade",
  "metadata_shap_modelo.rds",
  subpastas = "shap"
)
salvar_csv_saida(
  metadata_shap,
  "interpretabilidade",
  "metadata_shap_modelo.csv",
  subpastas = "shap"
)

salvar_figura_saida(
  plot = grafico_shap_importancia,
  fase = "interpretabilidade",
  arquivo = "shap_importancia_principal.png",
  subpastas = "shap",
  classificacao = "principal",
  width = 8,
  height = 5
)

salvar_figura_saida(
  plot = grafico_shap_bee,
  fase = "interpretabilidade",
  arquivo = "shap_beeswarm_suplementar.png",
  subpastas = "shap",
  classificacao = "suplementar",
  width = 9,
  height = 6
)

salvar_figura_saida(
  plot = grafico_dependencia,
  fase = "interpretabilidade",
  arquivo = "shap_dependencia_pay0_suplementar.png",
  subpastas = "shap",
  classificacao = "suplementar",
  width = 8,
  height = 5
)

if (!is.null(grafico_waterfall)) {
  salvar_figura_saida(
    plot = grafico_waterfall,
    fase = "interpretabilidade",
    arquivo = "shap_waterfall_exemplo_suplementar.png",
    subpastas = "shap",
    classificacao = "suplementar",
    width = 9,
    height = 5
  )
}

message("09_shap.R concluido com sucesso.")
