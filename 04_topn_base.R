# ==============================================================================
# 04_topn_base.R
# Responsabilidade: curva Top-N exploratoria para GLM.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar treino e ranking
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(ler_rds_base("treino.rds"))
ranking_variaveis <- ler_rds_base("ranking_variaveis_enet.rds")

ordem_variaveis <- ranking_variaveis$Variavel_Original
resultados_topn <- vector("list", length(ordem_variaveis))
folds_exploratorios <- criar_folds_estratificados(
  y = treino$Class,
  fase = "exploratorio"
)

# ------------------------------------------------------------------------------
# BLOCO 2 - Loop Top-1 ate Top-N
# ------------------------------------------------------------------------------
for (k in seq_along(ordem_variaveis)) {
  vars_k <- ordem_variaveis[seq_len(k)]
  dados_k <- treino[, c(vars_k, "Class"), drop = FALSE]
  formula_k <- montar_formula(vars_k)

  cat("\n=============================\n")
  cat("Rodando GLM Top-", k, "\n", sep = "")
  cat("Variaveis:", paste(vars_k, collapse = ", "), "\n")
  cat("=============================\n")

  modelo_glm_k <- treinar_modelo_caret(
    modelo = "GLM",
    formula_modelo = formula_k,
    dados_sub = dados_k,
    fase_validacao = "exploratorio",
    folds_cv = folds_exploratorios
  )

  resultados_topn[[k]] <- extrair_melhor_resultado_caret(
    modelo_glm_k,
    metadata = list(
      TopN = k,
      Modelo = "GLM",
      Variaveis = paste(vars_k, collapse = ", "),
      Usa_SMOTENC = FALSE
    )
  ) %>%
    adicionar_contexto_validacao(
      fase = "exploratorio",
      folds_cv = folds_exploratorios
    ) %>%
    dplyr::select(
      TopN, Modelo,
      ROC, Sens, Spec, Precision, F1, GMean,
      dplyr::everything()
    )
}

# ------------------------------------------------------------------------------
# BLOCO 3 - Consolidar curva e top 10
# ------------------------------------------------------------------------------
curva_topn <- dplyr::bind_rows(resultados_topn) %>%
  dplyr::arrange(TopN)

tabela_topn <- curva_topn %>%
  dplyr::arrange(desc(ROC), desc(F1), desc(GMean))

top10_topn <- tabela_topn %>%
  dplyr::slice(1:10)

melhor_topn <- curva_topn %>%
  dplyr::slice_max(order_by = ROC, n = 1, with_ties = FALSE)

print(melhor_topn)
print(top10_topn)

# ------------------------------------------------------------------------------
# BLOCO 4 - Graficos
# ------------------------------------------------------------------------------
top10_plot <- top10_topn %>%
  dplyr::mutate(
    Subconjunto = paste0("Top-", TopN),
    Subconjunto = reorder(Subconjunto, ROC)
  )

grafico_melhores_topn <- ggplot2::ggplot(top10_plot, ggplot2::aes(x = ROC, y = Subconjunto)) +
  ggplot2::geom_segment(
    ggplot2::aes(
      x = min(top10_plot$ROC) - 0.001,
      xend = ROC,
      y = Subconjunto,
      yend = Subconjunto
    ),
    color = "#a8ddb5",
    linewidth = 1
  ) +
  ggplot2::geom_point(color = "#006d2c", size = 4) +
  ggplot2::geom_text(
    ggplot2::aes(label = round(ROC, 5)),
    hjust = -0.3,
    size = 3.5,
    show.legend = FALSE
  ) +
  ggplot2::labs(
    title = "Top 10 Melhores Subconjuntos por ROC - GLM",
    x = "Area sob a curva ROC",
    y = NULL
  ) +
  ggplot2::xlim(min(top10_plot$ROC) - 0.001, max(top10_plot$ROC) + 0.001) +
  ggplot2::theme_minimal()

grafico_roc_topn <- ggplot2::ggplot(curva_topn, ggplot2::aes(x = TopN, y = ROC)) +
  ggplot2::geom_line(color = "#2ca25f", linewidth = 1) +
  ggplot2::geom_point(color = "#2ca25f", size = 2) +
  ggplot2::geom_point(
    data = melhor_topn,
    ggplot2::aes(x = TopN, y = ROC),
    color = "#00441b",
    size = 4
  ) +
  ggplot2::labs(
    title = "Desempenho por quantidade de variaveis - GLM",
    subtitle = "Curva de saturacao Top-N",
    x = "Quantidade de variaveis",
    y = "ROC"
  ) +
  ggplot2::theme_minimal()

print(grafico_melhores_topn)
print(grafico_roc_topn)

# ------------------------------------------------------------------------------
# BLOCO 5 - Salvar resultados
# ------------------------------------------------------------------------------
salvar_rds_saida(curva_topn, "exploratorio", "curva_topn_glm.rds", subpastas = "topn")
salvar_rds_saida(melhor_topn, "exploratorio", "melhor_topn_glm.rds", subpastas = "topn")
salvar_csv_saida(curva_topn, "exploratorio", "curva_topn_glm.csv", subpastas = "topn")
salvar_csv_saida(top10_topn, "exploratorio", "top10_topn_glm.csv", subpastas = "topn")

salvar_figura_saida(
  plot = grafico_roc_topn,
  fase = "exploratorio",
  arquivo = "roc_topn_glm_principal.png",
  subpastas = "topn",
  classificacao = "principal",
  width = 8,
  height = 5
)

salvar_figura_saida(
  plot = grafico_melhores_topn,
  fase = "exploratorio",
  arquivo = "top10_topn_glm_suplementar.png",
  subpastas = "topn",
  classificacao = "suplementar",
  width = 8,
  height = 5
)

message("04_topn_base.R concluido com sucesso.")
