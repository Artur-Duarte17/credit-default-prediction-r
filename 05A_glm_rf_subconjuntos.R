# ==============================================================================
# 05A_glm_rf_subconjuntos.R
# Responsabilidade: benchmark de GLM e RF nos subconjuntos Top-10/13/14.
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

print(subconjuntos)

# ------------------------------------------------------------------------------
# BLOCO 2 - Treinar GLM e RF
# ------------------------------------------------------------------------------
resultados_modelos <- list()
contador <- 1

for (nome_sub in names(subconjuntos)) {
  vars_sub <- subconjuntos[[nome_sub]]
  dados_sub <- treino[, c(vars_sub, "Class"), drop = FALSE]
  formula_sub <- montar_formula(vars_sub)

  cat("\n====================================================\n")
  cat("Subconjunto:", nome_sub, "\n")
  cat("Variaveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")

  modelo_glm <- treinar_modelo_caret(
    modelo = "GLM",
    formula_modelo = formula_sub,
    dados_sub = dados_sub
  )

  resultados_modelos[[contador]] <- extrair_melhor_resultado_caret(
    modelo_glm,
    metadata = list(
      Subconjunto = nome_sub,
      Modelo = "GLM",
      Variaveis = paste(vars_sub, collapse = ", "),
      Base_Treino = "TreinoCompleto",
      Usa_SMOTENC = FALSE
    )
  )
  contador <- contador + 1

  modelo_rf <- treinar_modelo_caret(
    modelo = "RF",
    formula_modelo = formula_sub,
    dados_sub = dados_sub
  )

  resultados_modelos[[contador]] <- extrair_melhor_resultado_caret(
    modelo_rf,
    metadata = list(
      Subconjunto = nome_sub,
      Modelo = "RF",
      Variaveis = paste(vars_sub, collapse = ", "),
      Base_Treino = "TreinoCompleto",
      Usa_SMOTENC = FALSE
    )
  )
  contador <- contador + 1
}

# ------------------------------------------------------------------------------
# BLOCO 3 - Consolidar resultados
# ------------------------------------------------------------------------------
tabela_benchmark <- dplyr::bind_rows(resultados_modelos) %>%
  dplyr::select(
    Subconjunto, Modelo,
    ROC, Sens, Spec, Precision, F1, GMean,
    dplyr::everything()
  ) %>%
  dplyr::arrange(desc(ROC), desc(F1), desc(GMean))

print(tabela_benchmark)

# ------------------------------------------------------------------------------
# BLOCO 4 - Graficos
# ------------------------------------------------------------------------------
grafico_roc_modelos <- ggplot2::ggplot(
  tabela_benchmark,
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
  tabela_benchmark,
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
saveRDS(tabela_benchmark, "objetos/tabela_benchmark_glm_rf_sem_balanceamento.rds")
readr::write_csv(tabela_benchmark, "resultados/tabela_benchmark_glm_rf_sem_balanceamento.csv")

ggplot2::ggsave(
  filename = "figuras/roc_glm_rf_subconjuntos_sem_balanceamento.png",
  plot = grafico_roc_modelos,
  width = 8,
  height = 5
)

ggplot2::ggsave(
  filename = "figuras/f1_glm_rf_subconjuntos_sem_balanceamento.png",
  plot = grafico_f1_modelos,
  width = 8,
  height = 5
)

message("05A_glm_rf_subconjuntos.R concluido com sucesso.")
