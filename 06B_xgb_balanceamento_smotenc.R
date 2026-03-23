# ==============================================================================
# 06B_xgb_balanceamento_smotenc.R
# Responsabilidade: comparar XGBoost sem balanceamento vs com SMOTENC.
# O passo step_smotenc() fica sempre com skip = TRUE.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar dados e ranking
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(readRDS("objetos/treino.rds"))
ranking_variaveis <- readRDS("objetos/ranking_variaveis_enet.rds")

ordem_variaveis <- ranking_variaveis$Variavel_Original
subconjuntos <- obter_subconjuntos_fixos(ordem_variaveis, tamanhos = c(13, 14))
grid_xgb <- grid_xgb_padrao()

resultados <- list()
contador <- 1

# ------------------------------------------------------------------------------
# BLOCO 2 - Loop por subconjunto
# ------------------------------------------------------------------------------
for (nome_sub in names(subconjuntos)) {
  vars_sub <- subconjuntos[[nome_sub]]
  dados_sub <- treino[, c(vars_sub, "Class"), drop = FALSE]
  formula_sub <- montar_formula(vars_sub)

  set.seed(SEED_PROJETO)
  folds_xgb <- caret::createFolds(dados_sub$Class, k = CV_FOLDS_PADRAO, returnTrain = FALSE)

  cat("\n====================================================\n")
  cat("Subconjunto:", nome_sub, "\n")
  cat("Variaveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")

  tabela_base <- avaliar_xgb_cv(
    dados = dados_sub,
    folds = folds_xgb,
    grid_xgb = grid_xgb,
    aplicar_smotenc = FALSE,
    formula_modelo = formula_sub
  )

  resultados[[contador]] <- tabela_base %>%
    dplyr::arrange(desc(ROC), desc(F1), desc(GMean)) %>%
    dplyr::slice(1) %>%
    dplyr::mutate(
      Subconjunto = nome_sub,
      Modelo = "XGBoost",
      Cenario = "Sem_balanceamento",
      Variaveis = paste(vars_sub, collapse = ", "),
      Usa_SMOTENC = FALSE
    )
  contador <- contador + 1

  tabela_smotenc <- avaliar_xgb_cv(
    dados = dados_sub,
    folds = folds_xgb,
    grid_xgb = grid_xgb,
    aplicar_smotenc = TRUE,
    formula_modelo = formula_sub
  )

  resultados[[contador]] <- tabela_smotenc %>%
    dplyr::arrange(desc(ROC), desc(F1), desc(GMean)) %>%
    dplyr::slice(1) %>%
    dplyr::mutate(
      Subconjunto = nome_sub,
      Modelo = "XGBoost",
      Cenario = "Com_SMOTENC",
      Variaveis = paste(vars_sub, collapse = ", "),
      Usa_SMOTENC = TRUE
    )
  contador <- contador + 1
}

# ------------------------------------------------------------------------------
# BLOCO 3 - Consolidar resultados
# ------------------------------------------------------------------------------
tabela_xgb_balanceamento <- dplyr::bind_rows(resultados) %>%
  dplyr::select(
    Subconjunto, Modelo, Cenario,
    ROC, Sens, Spec, Precision, F1, GMean,
    dplyr::everything()
  ) %>%
  dplyr::arrange(Subconjunto, desc(ROC), desc(F1), desc(GMean))

if (file.exists("objetos/tabela_rf_balanceamento_smotenc.rds")) {
  tabela_rf <- readRDS("objetos/tabela_rf_balanceamento_smotenc.rds")
  tabela_balanceamento_completa <- dplyr::bind_rows(tabela_rf, tabela_xgb_balanceamento) %>%
    dplyr::arrange(Modelo, Subconjunto, desc(ROC), desc(F1), desc(GMean))
} else {
  tabela_balanceamento_completa <- tabela_xgb_balanceamento
}

print(tabela_balanceamento_completa)

# ------------------------------------------------------------------------------
# BLOCO 4 - Graficos
# ------------------------------------------------------------------------------
grafico_roc_balanceamento <- ggplot2::ggplot(
  tabela_balanceamento_completa,
  ggplot2::aes(x = Subconjunto, y = ROC, color = Cenario, group = Cenario)
) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 3) +
  ggplot2::geom_text(
    ggplot2::aes(label = round(ROC, 4)),
    vjust = -0.8,
    size = 3
  ) +
  ggplot2::facet_wrap(~ Modelo) +
  ggplot2::labs(
    title = "ROC: comparacao de balanceamento por modelo",
    x = "Subconjunto",
    y = "ROC"
  ) +
  ggplot2::theme_minimal()

grafico_f1_balanceamento <- ggplot2::ggplot(
  tabela_balanceamento_completa,
  ggplot2::aes(x = Subconjunto, y = F1, color = Cenario, group = Cenario)
) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 3) +
  ggplot2::geom_text(
    ggplot2::aes(label = round(F1, 4)),
    vjust = -0.8,
    size = 3
  ) +
  ggplot2::facet_wrap(~ Modelo) +
  ggplot2::labs(
    title = "F1: comparacao de balanceamento por modelo",
    x = "Subconjunto",
    y = "F1"
  ) +
  ggplot2::theme_minimal()

print(grafico_roc_balanceamento)
print(grafico_f1_balanceamento)

# ------------------------------------------------------------------------------
# BLOCO 5 - Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(tabela_xgb_balanceamento, "objetos/tabela_xgb_balanceamento_smotenc.rds")
readr::write_csv(tabela_xgb_balanceamento, "resultados/tabela_xgb_balanceamento_smotenc.csv")

saveRDS(tabela_balanceamento_completa, "objetos/tabela_rf_xgb_balanceamento_smotenc.rds")
readr::write_csv(tabela_balanceamento_completa, "resultados/tabela_rf_xgb_balanceamento_smotenc.csv")

ggplot2::ggsave(
  filename = "figuras/roc_rf_xgb_balanceamento_smotenc.png",
  plot = grafico_roc_balanceamento,
  width = 10,
  height = 5
)

ggplot2::ggsave(
  filename = "figuras/f1_rf_xgb_balanceamento_smotenc.png",
  plot = grafico_f1_balanceamento,
  width = 10,
  height = 5
)

message("06B_xgb_balanceamento_smotenc.R concluido com sucesso.")
