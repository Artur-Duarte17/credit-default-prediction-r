# ==============================================================================
# 04C_topn_modelos_caret.R
# Responsabilidade: curvas Top-N para RF, SVM radial, NNET e avNNet.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar treino e ranking
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(readRDS("objetos/treino.rds"))
ranking_variaveis <- readRDS("objetos/ranking_variaveis_enet.rds")

ordem_variaveis <- ranking_variaveis$Variavel_Original
modelos_topn <- c("RF", "SVM_Radial", "NNET", "avNNet")

# ------------------------------------------------------------------------------
# BLOCO 2 - Rodar curvas Top-N
# ------------------------------------------------------------------------------
resultados_modelos <- vector("list", length(modelos_topn))
names(resultados_modelos) <- modelos_topn

for (modelo_atual in modelos_topn) {
  resultados_topn <- vector("list", length(ordem_variaveis))

  for (k in seq_along(ordem_variaveis)) {
    vars_k <- ordem_variaveis[seq_len(k)]
    dados_k <- treino[, c(vars_k, "Class"), drop = FALSE]
    formula_k <- montar_formula(vars_k)

    cat("\n=============================\n")
    cat("Modelo:", modelo_atual, "- Top-", k, "\n")
    cat("Variaveis:", paste(vars_k, collapse = ", "), "\n")
    cat("=============================\n")

    modelo_ajustado <- treinar_modelo_caret(
      modelo = modelo_atual,
      formula_modelo = formula_k,
      dados_sub = dados_k
    )

    resultados_topn[[k]] <- extrair_melhor_resultado_caret(
      modelo_ajustado,
      metadata = list(
        TopN = k,
        Modelo = modelo_atual,
        Variaveis = paste(vars_k, collapse = ", "),
        Usa_SMOTENC = FALSE
      )
    ) %>%
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
  dplyr::slice_max(order_by = ROC, n = 10, with_ties = FALSE) %>%
  dplyr::arrange(Modelo, dplyr::desc(ROC)) %>%
  dplyr::ungroup()

print(top10_topn_modelos)

# ------------------------------------------------------------------------------
# BLOCO 3 - Graficos
# ------------------------------------------------------------------------------
grafico_curvas_topn <- ggplot2::ggplot(
  curva_topn_modelos,
  ggplot2::aes(x = TopN, y = ROC, color = Modelo)
) +
  ggplot2::geom_line(linewidth = 1) +
  ggplot2::geom_point(size = 1.7) +
  ggplot2::labs(
    title = "Curvas Top-N por modelo",
    subtitle = "RF, SVM radial e redes neurais",
    x = "Quantidade de variaveis",
    y = "ROC"
  ) +
  ggplot2::theme_minimal()

grafico_top10_modelos <- ggplot2::ggplot(
  top10_topn_modelos,
  ggplot2::aes(
    x = ROC,
    y = reorder(paste0(Modelo, " / Top-", TopN), ROC),
    fill = Modelo
  )
) +
  ggplot2::geom_col() +
  ggplot2::labs(
    title = "Top 10 subconjuntos por modelo",
    x = "ROC",
    y = NULL
  ) +
  ggplot2::theme_minimal()

print(grafico_curvas_topn)
print(grafico_top10_modelos)

# ------------------------------------------------------------------------------
# BLOCO 4 - Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(curva_topn_modelos, "objetos/curva_topn_modelos_caret.rds")
readr::write_csv(curva_topn_modelos, "resultados/curva_topn_modelos_caret.csv")
readr::write_csv(top10_topn_modelos, "resultados/top10_topn_modelos_caret.csv")

ggplot2::ggsave(
  filename = "figuras/curva_roc_topn_modelos_caret.png",
  plot = grafico_curvas_topn,
  width = 10,
  height = 6
)

ggplot2::ggsave(
  filename = "figuras/top10_melhores_subconjuntos_modelos_caret.png",
  plot = grafico_top10_modelos,
  width = 10,
  height = 6
)

message("04C_topn_modelos_caret.R concluido com sucesso.")
