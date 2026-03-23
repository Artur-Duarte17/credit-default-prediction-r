# ==============================================================================
# 06B_xgb_balanceamento_smotenc.R
# Responsabilidade: fase de confirmacao para XGBoost, comparando sem
# balanceamento versus com SMOTENC apenas no melhor subconjunto confirmado.
# O passo step_smotenc() fica sempre com skip = TRUE.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar dados e finalista confirmado
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(readRDS("objetos/treino.rds"))
tabela_xgb_confirmada <- readRDS("objetos/tabela_xgboost_subconjuntos_sem_balanceamento.rds") %>%
  dplyr::filter(Modelo == "XGBoost")

finalistas_xgb <- selecionar_finalistas_modelagem(
  tabela = tabela_xgb_confirmada,
  n_por_modelo = N_FINALISTAS_BALANCEAMENTO_POR_MODELO
)

folds_confirmacao <- criar_folds_estratificados(
  y = treino$Class,
  fase = "confirmacao"
)
grid_xgb_confirmacao <- obter_grid_modelo_fase(
  modelo = "XGBoost",
  dados_sub = treino[, c(setdiff(names(treino), "Class")[1], "Class"), drop = FALSE],
  fase = "confirmacao"
)

print(finalistas_xgb)

resultados <- list()
contador <- 1

# ------------------------------------------------------------------------------
# BLOCO 2 - Confirmacao do balanceamento no subconjunto finalista
# ------------------------------------------------------------------------------
for (i in seq_len(nrow(finalistas_xgb))) {
  config_atual <- finalistas_xgb[i, , drop = FALSE]
  vars_sub <- parse_variaveis(config_atual$Variaveis[1])
  dados_sub <- treino[, c(vars_sub, "Class"), drop = FALSE]
  formula_sub <- montar_formula(vars_sub)

  cat("\n====================================================\n")
  cat("Confirmacao de balanceamento -", config_atual$Subconjunto[1], "\n")
  cat("Variaveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")

  tabela_base <- avaliar_xgb_cv(
    dados = dados_sub,
    folds = folds_confirmacao,
    grid_xgb = grid_xgb_confirmacao,
    aplicar_smotenc = FALSE,
    formula_modelo = formula_sub
  )

  resultados[[contador]] <- extrair_melhor_resultado_xgb(
    tabela_base,
    metadata = list(
      Subconjunto = config_atual$Subconjunto[1],
      TopN = config_atual$TopN[1],
      Modelo = "XGBoost",
      Cenario = "Sem_balanceamento",
      Variaveis = config_atual$Variaveis[1],
      Usa_SMOTENC = FALSE
    )
  ) %>%
    adicionar_contexto_validacao(
      fase = "confirmacao",
      folds_cv = folds_confirmacao
    )
  contador <- contador + 1

  tabela_smotenc <- avaliar_xgb_cv(
    dados = dados_sub,
    folds = folds_confirmacao,
    grid_xgb = grid_xgb_confirmacao,
    aplicar_smotenc = TRUE,
    formula_modelo = formula_sub
  )

  resultados[[contador]] <- extrair_melhor_resultado_xgb(
    tabela_smotenc,
    metadata = list(
      Subconjunto = config_atual$Subconjunto[1],
      TopN = config_atual$TopN[1],
      Modelo = "XGBoost",
      Cenario = "Com_SMOTENC",
      Variaveis = config_atual$Variaveis[1],
      Usa_SMOTENC = TRUE
    )
  ) %>%
    adicionar_contexto_validacao(
      fase = "confirmacao",
      folds_cv = folds_confirmacao
    )
  contador <- contador + 1
}

# ------------------------------------------------------------------------------
# BLOCO 3 - Consolidar resultados
# ------------------------------------------------------------------------------
tabela_xgb_balanceamento <- dplyr::bind_rows(resultados) %>%
  dplyr::select(
    Subconjunto, TopN, Modelo, Cenario,
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
    title = "ROC: confirmacao do balanceamento por modelo",
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
    title = "F1: confirmacao do balanceamento por modelo",
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
