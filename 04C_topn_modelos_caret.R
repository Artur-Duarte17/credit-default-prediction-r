# ==============================================================================
# 04C_topn_modelos_caret.R
# Responsabilidade: curvas Top-N exploratorias para RF, SVM radial, NNET e avNNet.
# Por padrao, esta etapa usa apenas TopN candidatos guiados por GLM e XGBoost.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar treino e ranking
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(ler_rds_base("treino.rds"))
ranking_variaveis <- ler_rds_base("ranking_variaveis_enet.rds")

ordem_variaveis <- ranking_variaveis$Variavel_Original
modelos_topn <- c("RF", "SVM_Radial", "NNET", "avNNet")

if (isTRUE(RODAR_TOPN_COMPLETO_MODELOS_CARET)) {
  topn_utilizados <- tibble::tibble(
    TopN = seq_along(ordem_variaveis),
    Origem = "Override_TopN_Completo",
    MelhorTopN_GLM = NA_integer_,
    MelhorTopN_XGBoost = NA_integer_
  )
} else {
  topn_utilizados <- obter_topn_candidatos(ordem_variaveis = ordem_variaveis)
}

subconjuntos_candidatos <- obter_subconjuntos_candidatos(
  ordem_variaveis = ordem_variaveis,
  topn_candidatos = topn_utilizados$TopN
)

folds_exploratorios <- criar_folds_estratificados(
  y = treino$Class,
  fase = "exploratorio"
)

print(topn_utilizados)

# ------------------------------------------------------------------------------
# BLOCO 2 - Rodar curvas Top-N candidatas
# ------------------------------------------------------------------------------
resultados_modelos <- vector("list", length(modelos_topn))
names(resultados_modelos) <- modelos_topn

for (modelo_atual in modelos_topn) {
  resultados_topn <- vector("list", length(subconjuntos_candidatos))

  for (idx_sub in seq_along(subconjuntos_candidatos)) {
    nome_sub <- names(subconjuntos_candidatos)[idx_sub]
    vars_k <- subconjuntos_candidatos[[idx_sub]]
    topn_k <- as.integer(gsub("^Top", "", nome_sub))
    dados_k <- treino[, c(vars_k, "Class"), drop = FALSE]
    formula_k <- montar_formula(vars_k)

    cat("\n=============================\n")
    cat("Modelo:", modelo_atual, "-", nome_sub, "\n")
    cat("Variaveis:", paste(vars_k, collapse = ", "), "\n")
    cat("=============================\n")

    modelo_ajustado <- treinar_modelo_caret(
      modelo = modelo_atual,
      formula_modelo = formula_k,
      dados_sub = dados_k,
      fase_validacao = "exploratorio",
      folds_cv = folds_exploratorios
    )

    resultados_topn[[idx_sub]] <- extrair_melhor_resultado_caret(
      modelo_ajustado,
      metadata = list(
        TopN = topn_k,
        Modelo = modelo_atual,
        Variaveis = paste(vars_k, collapse = ", "),
        Usa_SMOTENC = FALSE
      )
    ) %>%
      adicionar_contexto_validacao(
        fase = "exploratorio",
        folds_cv = folds_exploratorios
      ) %>%
      dplyr::left_join(topn_utilizados, by = "TopN") %>%
      dplyr::select(
        TopN, Modelo,
        ROC, Sens, Spec, Precision, F1, GMean,
        dplyr::everything()
      )
  }

  resultados_modelos[[modelo_atual]] <- dplyr::bind_rows(resultados_topn)
}

curva_topn_modelos <- dplyr::bind_rows(resultados_modelos) %>%
  dplyr::arrange(Modelo, TopN)

top10_topn_modelos <- curva_topn_modelos %>%
  dplyr::group_by(Modelo) %>%
  dplyr::group_modify(~ ordenar_resultados_modelagem(.x) %>% dplyr::slice_head(n = min(10, nrow(.x)))) %>%
  dplyr::ungroup()

print(top10_topn_modelos)

# ------------------------------------------------------------------------------
# BLOCO 3 - Graficos
# ------------------------------------------------------------------------------
grafico_topn_candidatos <- ggplot2::ggplot(
  curva_topn_modelos,
  ggplot2::aes(x = TopN, y = ROC, color = Modelo)
) +
  ggplot2::geom_point(size = 2.3) +
  ggplot2::facet_wrap(~ Modelo, scales = "free_x") +
  ggplot2::labs(
    title = "Top-N candidatos por modelo",
    subtitle = "RF, SVM radial e redes com reducao guiada por GLM e XGBoost",
    x = "Quantidade de variaveis",
    y = "ROC"
  ) +
  ggplot2::theme_minimal()

print(grafico_topn_candidatos)

# ------------------------------------------------------------------------------
# BLOCO 4 - Salvar resultados
# ------------------------------------------------------------------------------
salvar_rds_saida(curva_topn_modelos, "exploratorio", "curva_topn_modelos_caret.rds", subpastas = "topn")
salvar_rds_saida(topn_utilizados, "exploratorio", "topn_candidatos_modelos_caret.rds", subpastas = "topn")

salvar_csv_saida(curva_topn_modelos, "exploratorio", "curva_topn_modelos_caret.csv", subpastas = "topn")
salvar_csv_saida(top10_topn_modelos, "exploratorio", "top10_topn_modelos_caret.csv", subpastas = "topn")
salvar_csv_saida(topn_utilizados, "exploratorio", "topn_candidatos_modelos_caret.csv", subpastas = "topn")

salvar_figura_saida(
  plot = grafico_topn_candidatos,
  fase = "exploratorio",
  arquivo = "roc_topn_modelos_caret_suplementar.png",
  subpastas = "topn",
  classificacao = "suplementar",
  width = 10,
  height = 6
)

message("04C_topn_modelos_caret.R concluido com sucesso.")
