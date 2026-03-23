# ==============================================================================
# 06_rf_balanceamento_smotenc.R
# Responsabilidade: comparar RF sem balanceamento vs com SMOTENC.
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

resultados <- list()
contador <- 1

# ------------------------------------------------------------------------------
# BLOCO 2 - Loop por subconjunto
# ------------------------------------------------------------------------------
for (nome_sub in names(subconjuntos)) {
  vars_sub <- subconjuntos[[nome_sub]]
  dados_sub <- treino[, c(vars_sub, "Class"), drop = FALSE]
  formula_sub <- montar_formula(vars_sub)

  cat("\n====================================================\n")
  cat("Subconjunto:", nome_sub, "\n")
  cat("Variaveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")

  modelo_rf_base <- treinar_modelo_caret(
    modelo = "RF",
    formula_modelo = formula_sub,
    dados_sub = dados_sub,
    usar_smotenc = FALSE
  )

  resultados[[contador]] <- extrair_melhor_resultado_caret(
    modelo_rf_base,
    metadata = list(
      Subconjunto = nome_sub,
      Modelo = "RF",
      Cenario = "Sem_balanceamento",
      Variaveis = paste(vars_sub, collapse = ", "),
      Usa_SMOTENC = FALSE
    )
  )
  contador <- contador + 1

  modelo_rf_smotenc <- treinar_modelo_caret(
    modelo = "RF",
    formula_modelo = formula_sub,
    dados_sub = dados_sub,
    usar_smotenc = TRUE
  )

  resultados[[contador]] <- extrair_melhor_resultado_caret(
    modelo_rf_smotenc,
    metadata = list(
      Subconjunto = nome_sub,
      Modelo = "RF",
      Cenario = "Com_SMOTENC",
      Variaveis = paste(vars_sub, collapse = ", "),
      Usa_SMOTENC = TRUE
    )
  )
  contador <- contador + 1
}

# ------------------------------------------------------------------------------
# BLOCO 3 - Consolidar resultados
# ------------------------------------------------------------------------------
tabela_balanceamento <- dplyr::bind_rows(resultados) %>%
  dplyr::select(
    Subconjunto, Modelo, Cenario,
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
    title = "RF: ROC sem balanceamento vs com SMOTENC",
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
    title = "RF: F1 sem balanceamento vs com SMOTENC",
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
