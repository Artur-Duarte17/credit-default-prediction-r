# ==============================================================================
# 06_rf_balanceamento_smotenc.R
# Responsabilidade: fase de confirmacao para RF, comparando sem balanceamento
# versus com SMOTENC apenas no melhor subconjunto confirmado do modelo.
# O passo step_smotenc() fica sempre com skip = TRUE.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar dados e finalista confirmado
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(readRDS("objetos/treino.rds"))
tabela_rf_confirmada <- readRDS("objetos/tabela_benchmark_glm_rf_sem_balanceamento.rds") %>%
  dplyr::filter(Modelo == "RF")

finalistas_rf <- selecionar_finalistas_modelagem(
  tabela = tabela_rf_confirmada,
  n_por_modelo = N_FINALISTAS_BALANCEAMENTO_POR_MODELO
)

folds_confirmacao <- criar_folds_estratificados(
  y = treino$Class,
  fase = "confirmacao"
)

print(finalistas_rf)

resultados <- list()
contador <- 1

# ------------------------------------------------------------------------------
# BLOCO 2 - Confirmacao do balanceamento no subconjunto finalista
# ------------------------------------------------------------------------------
for (i in seq_len(nrow(finalistas_rf))) {
  config_atual <- finalistas_rf[i, , drop = FALSE]
  vars_sub <- parse_variaveis(config_atual$Variaveis[1])
  dados_sub <- treino[, c(vars_sub, "Class"), drop = FALSE]
  formula_sub <- montar_formula(vars_sub)

  cat("\n====================================================\n")
  cat("Confirmacao de balanceamento -", config_atual$Subconjunto[1], "\n")
  cat("Variaveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")

  modelo_rf_base <- treinar_modelo_caret(
    modelo = "RF",
    formula_modelo = formula_sub,
    dados_sub = dados_sub,
    usar_smotenc = FALSE,
    fase_validacao = "confirmacao",
    folds_cv = folds_confirmacao
  )

  resultados[[contador]] <- extrair_melhor_resultado_caret(
    modelo_rf_base,
    metadata = list(
      Subconjunto = config_atual$Subconjunto[1],
      TopN = config_atual$TopN[1],
      Modelo = "RF",
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

  modelo_rf_smotenc <- treinar_modelo_caret(
    modelo = "RF",
    formula_modelo = formula_sub,
    dados_sub = dados_sub,
    usar_smotenc = TRUE,
    fase_validacao = "confirmacao",
    folds_cv = folds_confirmacao
  )

  resultados[[contador]] <- extrair_melhor_resultado_caret(
    modelo_rf_smotenc,
    metadata = list(
      Subconjunto = config_atual$Subconjunto[1],
      TopN = config_atual$TopN[1],
      Modelo = "RF",
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
tabela_balanceamento <- dplyr::bind_rows(resultados) %>%
  dplyr::select(
    Subconjunto, TopN, Modelo, Cenario,
    ROC, Sens, Spec, Precision, F1, GMean,
    dplyr::everything()
  ) %>%
  dplyr::arrange(Subconjunto, desc(ROC), desc(F1), desc(GMean))

print(tabela_balanceamento)

# ------------------------------------------------------------------------------
# BLOCO 4 - Graficos
# ------------------------------------------------------------------------------
grafico_roc_balanceamento <- ggplot2::ggplot(
  tabela_balanceamento,
  ggplot2::aes(x = Subconjunto, y = ROC, color = Cenario, group = Cenario)
) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 3) +
  ggplot2::geom_text(
    ggplot2::aes(label = round(ROC, 4)),
    vjust = -0.8,
    size = 3
  ) +
  ggplot2::labs(
    title = "RF: ROC confirmado sem balanceamento vs com SMOTENC",
    x = "Subconjunto",
    y = "ROC"
  ) +
  ggplot2::theme_minimal()

grafico_f1_balanceamento <- ggplot2::ggplot(
  tabela_balanceamento,
  ggplot2::aes(x = Subconjunto, y = F1, color = Cenario, group = Cenario)
) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 3) +
  ggplot2::geom_text(
    ggplot2::aes(label = round(F1, 4)),
    vjust = -0.8,
    size = 3
  ) +
  ggplot2::labs(
    title = "RF: F1 confirmado sem balanceamento vs com SMOTENC",
    x = "Subconjunto",
    y = "F1"
  ) +
  ggplot2::theme_minimal()

print(grafico_roc_balanceamento)
print(grafico_f1_balanceamento)

# ------------------------------------------------------------------------------
# BLOCO 5 - Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(tabela_balanceamento, "objetos/tabela_rf_balanceamento_smotenc.rds")
readr::write_csv(tabela_balanceamento, "resultados/tabela_rf_balanceamento_smotenc.csv")

ggplot2::ggsave(
  filename = "figuras/roc_rf_balanceamento_smotenc.png",
  plot = grafico_roc_balanceamento,
  width = 8,
  height = 5
)

ggplot2::ggsave(
  filename = "figuras/f1_rf_balanceamento_smotenc.png",
  plot = grafico_f1_balanceamento,
  width = 8,
  height = 5
)

message("06_rf_balanceamento_smotenc.R concluido com sucesso.")
