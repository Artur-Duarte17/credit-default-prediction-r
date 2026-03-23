# ==============================================================================
# 04B_topn_xgboost.R
# Responsabilidade: curva Top-N exploratoria para XGBoost.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar treino e ranking
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(ler_rds_base("treino.rds"))
ranking_variaveis <- ler_rds_base("ranking_variaveis_enet.rds")

ordem_variaveis <- ranking_variaveis$Variavel_Original
n_total_variaveis <- length(ordem_variaveis)

set.seed(SEED_PROJETO)
folds_cv <- criar_folds_estratificados(
  y = treino$Class,
  fase = "exploratorio"
)
grid_xgb <- obter_grid_modelo_fase(
  modelo = "XGBoost",
  dados_sub = treino[, c(ordem_variaveis[1], "Class"), drop = FALSE],
  fase = "exploratorio"
)
resultados_topn <- vector("list", n_total_variaveis)

# ------------------------------------------------------------------------------
# BLOCO 2 - Loop Top-1 ate Top-N
# ------------------------------------------------------------------------------
for (k in seq_along(ordem_variaveis)) {
  vars_k <- ordem_variaveis[seq_len(k)]
  dados_k <- treino[, c(vars_k, "Class"), drop = FALSE]
  formula_k <- montar_formula(vars_k)

  cat("\n=============================\n")
  cat("Rodando XGBoost Top-", k, "\n", sep = "")
  cat("Variaveis:", paste(vars_k, collapse = ", "), "\n")
  cat("=============================\n")

  tabela_xgb_k <- avaliar_xgb_cv(
    dados = dados_k,
    folds = folds_cv,
    grid_xgb = grid_xgb,
    aplicar_smotenc = FALSE,
    formula_modelo = formula_k
  )

  resultados_topn[[k]] <- extrair_melhor_resultado_xgb(
    tabela_xgb_k,
    metadata = list(
      TopN = k,
      Modelo = "XGBoost",
      Variaveis = paste(vars_k, collapse = ", "),
      Usa_SMOTENC = FALSE
    )
  ) %>%
    adicionar_contexto_validacao(
      fase = "exploratorio",
      folds_cv = folds_cv
    ) %>%
    dplyr::select(
      TopN, Modelo,
      ROC, Sens, Spec, Precision, F1, GMean,
      dplyr::everything()
    )
  print(resultados_topn[[k]])
}

# ------------------------------------------------------------------------------
# BLOCO 3 - Consolidar resultados
# ------------------------------------------------------------------------------
curva_topn_xgb <- dplyr::bind_rows(resultados_topn) %>%
  dplyr::arrange(TopN)

tabela_topn_xgb <- curva_topn_xgb %>%
  dplyr::arrange(desc(ROC), desc(F1), desc(GMean))

top10_topn_xgb <- tabela_topn_xgb %>%
  dplyr::slice(1:10)

melhor_topn_xgb <- curva_topn_xgb %>%
  dplyr::slice_max(order_by = ROC, n = 1, with_ties = FALSE)

print(melhor_topn_xgb)

# ------------------------------------------------------------------------------
# BLOCO 4 - Graficos
# ------------------------------------------------------------------------------
top10_plot <- top10_topn_xgb %>%
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
    color = "#9ecae1",
    linewidth = 1
  ) +
  ggplot2::geom_point(color = "#08519c", size = 4) +
  ggplot2::geom_text(
    ggplot2::aes(label = round(ROC, 5)),
    hjust = -0.3,
    size = 3.5,
    show.legend = FALSE
  ) +
  ggplot2::labs(
    title = "Top 10 Melhores Subconjuntos por ROC - XGBoost",
    x = "Area sob a curva ROC",
    y = NULL
  ) +
  ggplot2::xlim(min(top10_plot$ROC) - 0.001, max(top10_plot$ROC) + 0.001) +
  ggplot2::theme_minimal()

grafico_roc_topn <- ggplot2::ggplot(curva_topn_xgb, ggplot2::aes(x = TopN, y = ROC)) +
  ggplot2::geom_line(color = "#3182bd", linewidth = 1) +
  ggplot2::geom_point(color = "#3182bd", size = 2) +
  ggplot2::geom_point(
    data = melhor_topn_xgb,
    ggplot2::aes(x = TopN, y = ROC),
    color = "#08306b",
    size = 4
  ) +
  ggplot2::labs(
    title = "Desempenho por quantidade de variaveis - XGBoost",
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
salvar_rds_saida(curva_topn_xgb, "exploratorio", "curva_topn_xgboost.rds", subpastas = "topn")
salvar_rds_saida(melhor_topn_xgb, "exploratorio", "melhor_topn_xgboost.rds", subpastas = "topn")

salvar_csv_saida(curva_topn_xgb, "exploratorio", "curva_topn_xgboost.csv", subpastas = "topn")
salvar_csv_saida(top10_topn_xgb, "exploratorio", "top10_topn_xgboost.csv", subpastas = "topn")

salvar_figura_saida(
  plot = grafico_roc_topn,
  fase = "exploratorio",
  arquivo = "roc_topn_xgboost_principal.png",
  subpastas = "topn",
  classificacao = "principal",
  width = 8,
  height = 5
)

salvar_figura_saida(
  plot = grafico_melhores_topn,
  fase = "exploratorio",
  arquivo = "top10_topn_xgboost_suplementar.png",
  subpastas = "topn",
  classificacao = "suplementar",
  width = 8,
  height = 5
)

message("04B_topn_xgboost.R concluido com sucesso.")
