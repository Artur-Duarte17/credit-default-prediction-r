# ==============================================================================
# 05B_xgboost_subconjuntos.R
# Responsabilidade: benchmark de XGBoost nos subconjuntos Top-10/13/14.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar dados e ranking
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(readRDS("objetos/treino.rds"))
ranking_variaveis <- readRDS("objetos/ranking_variaveis_enet.rds")

ordem_variaveis <- ranking_variaveis$Variavel_Original
subconjuntos <- obter_subconjuntos_fixos(ordem_variaveis)
grid_xgb <- grid_xgb_padrao()

print(subconjuntos)

# ------------------------------------------------------------------------------
# BLOCO 2 - Treinar XGBoost
# ------------------------------------------------------------------------------
resultados_xgb <- list()
contador <- 1

for (nome_sub in names(subconjuntos)) {
  vars_sub <- subconjuntos[[nome_sub]]
  dados_sub <- treino[, c(vars_sub, "Class"), drop = FALSE]
  formula_sub <- montar_formula(vars_sub)

  set.seed(SEED_PROJETO)
  folds_cv <- caret::createFolds(dados_sub$Class, k = CV_FOLDS_PADRAO, returnTrain = FALSE)

  cat("\n====================================================\n")
  cat("Subconjunto:", nome_sub, "\n")
  cat("Variaveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")

  tabela_xgb_sub <- avaliar_xgb_cv(
    dados = dados_sub,
    folds = folds_cv,
    grid_xgb = grid_xgb,
    aplicar_smotenc = FALSE,
    formula_modelo = formula_sub
  )

  resultados_xgb[[contador]] <- tabela_xgb_sub %>%
    dplyr::arrange(desc(ROC), desc(F1), desc(GMean)) %>%
    dplyr::slice(1) %>%
    dplyr::mutate(
      Subconjunto = nome_sub,
      Modelo = "XGBoost",
      Variaveis = paste(vars_sub, collapse = ", "),
      Base_Treino = "TreinoCompleto",
      Usa_SMOTENC = FALSE
    )
  contador <- contador + 1
}

# ------------------------------------------------------------------------------
# BLOCO 3 - Consolidar resultados
# ------------------------------------------------------------------------------
tabela_xgb <- dplyr::bind_rows(resultados_xgb) %>%
  dplyr::select(
    Subconjunto, Modelo,
    ROC, Sens, Spec, Precision, F1, GMean,
    dplyr::everything()
  ) %>%
  dplyr::arrange(desc(ROC), desc(F1), desc(GMean))

if (file.exists("objetos/tabela_benchmark_glm_rf_sem_balanceamento.rds")) {
  tabela_glm_rf <- readRDS("objetos/tabela_benchmark_glm_rf_sem_balanceamento.rds")
  tabela_benchmark_completa <- dplyr::bind_rows(tabela_glm_rf, tabela_xgb) %>%
    dplyr::arrange(desc(ROC), desc(F1), desc(GMean))
} else {
  tabela_benchmark_completa <- tabela_xgb
}

print(tabela_benchmark_completa)

# ------------------------------------------------------------------------------
# BLOCO 4 - Graficos
# ------------------------------------------------------------------------------
grafico_roc_modelos <- ggplot2::ggplot(
  tabela_benchmark_completa,
  ggplot2::aes(x = Subconjunto, y = ROC, color = Modelo, group = Modelo)
) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 3) +
  ggplot2::geom_text(
    ggplot2::aes(label = round(ROC, 4)),
    vjust = -0.8,
    size = 3
  ) +
  ggplot2::labs(
    title = "ROC por modelo e subconjunto",
    x = "Subconjunto",
    y = "ROC"
  ) +
  ggplot2::theme_minimal()

grafico_f1_modelos <- ggplot2::ggplot(
  tabela_benchmark_completa,
  ggplot2::aes(x = Subconjunto, y = F1, color = Modelo, group = Modelo)
) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 3) +
  ggplot2::geom_text(
    ggplot2::aes(label = round(F1, 4)),
    vjust = -0.8,
    size = 3
  ) +
  ggplot2::labs(
    title = "F1 por modelo e subconjunto",
    x = "Subconjunto",
    y = "F1"
  ) +
  ggplot2::theme_minimal()

print(grafico_roc_modelos)
print(grafico_f1_modelos)

# ------------------------------------------------------------------------------
# BLOCO 5 - Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(tabela_xgb, "objetos/tabela_xgboost_subconjuntos_sem_balanceamento.rds")
readr::write_csv(tabela_xgb, "resultados/tabela_xgboost_subconjuntos_sem_balanceamento.csv")

saveRDS(
  tabela_benchmark_completa,
  "objetos/tabela_benchmark_glm_rf_xgb_sem_balanceamento.rds"
)
readr::write_csv(
  tabela_benchmark_completa,
  "resultados/tabela_benchmark_glm_rf_xgb_sem_balanceamento.csv"
)

ggplot2::ggsave(
  filename = "figuras/roc_glm_rf_xgb_subconjuntos_sem_balanceamento.png",
  plot = grafico_roc_modelos,
  width = 8,
  height = 5
)

ggplot2::ggsave(
  filename = "figuras/f1_glm_rf_xgb_subconjuntos_sem_balanceamento.png",
  plot = grafico_f1_modelos,
  width = 8,
  height = 5
)

message("05B_xgboost_subconjuntos.R concluido com sucesso.")
