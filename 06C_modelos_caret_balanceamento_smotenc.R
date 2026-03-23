# ==============================================================================
# 06C_modelos_caret_balanceamento_smotenc.R
# Responsabilidade: comparar SVM radial, NNET e avNNet com e sem SMOTENC.
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
modelos_balancear <- c("SVM_Radial", "NNET", "avNNet")

resultados <- list()
contador <- 1

# ------------------------------------------------------------------------------
# BLOCO 2 - Loop por subconjunto e modelo
# ------------------------------------------------------------------------------
for (nome_sub in names(subconjuntos)) {
  vars_sub <- subconjuntos[[nome_sub]]
  dados_sub <- treino[, c(vars_sub, "Class"), drop = FALSE]
  formula_sub <- montar_formula(vars_sub)

  cat("\n====================================================\n")
  cat("Subconjunto:", nome_sub, "\n")
  cat("Variaveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")

  for (modelo_atual in modelos_balancear) {
    modelo_base <- treinar_modelo_caret(
      modelo = modelo_atual,
      formula_modelo = formula_sub,
      dados_sub = dados_sub,
      usar_smotenc = FALSE
    )

    resultados[[contador]] <- extrair_melhor_resultado_caret(
      modelo_base,
      metadata = list(
        Subconjunto = nome_sub,
        Modelo = modelo_atual,
        Cenario = "Sem_balanceamento",
        Variaveis = paste(vars_sub, collapse = ", "),
        Usa_SMOTENC = FALSE
      )
    )
    contador <- contador + 1

    modelo_smotenc <- treinar_modelo_caret(
      modelo = modelo_atual,
      formula_modelo = formula_sub,
      dados_sub = dados_sub,
      usar_smotenc = TRUE
    )

    resultados[[contador]] <- extrair_melhor_resultado_caret(
      modelo_smotenc,
      metadata = list(
        Subconjunto = nome_sub,
        Modelo = modelo_atual,
        Cenario = "Com_SMOTENC",
        Variaveis = paste(vars_sub, collapse = ", "),
        Usa_SMOTENC = TRUE
      )
    )
    contador <- contador + 1
  }
}

tabela_balanceamento <- dplyr::bind_rows(resultados) %>%
  dplyr::select(
    Subconjunto, Modelo, Cenario,
    ROC, Sens, Spec, Precision, F1, GMean,
    dplyr::everything()
  ) %>%
  dplyr::arrange(Modelo, Subconjunto, desc(ROC), desc(F1), desc(GMean))

print(tabela_balanceamento)

grafico_roc_balanceamento <- ggplot2::ggplot(
  tabela_balanceamento,
  ggplot2::aes(x = Subconjunto, y = ROC, color = Cenario, group = Cenario)
) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 3) +
  ggplot2::facet_wrap(~ Modelo) +
  ggplot2::labs(
    title = "ROC: SMOTENC nos modelos caret adicionais",
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
  ggplot2::facet_wrap(~ Modelo) +
  ggplot2::labs(
    title = "F1: SMOTENC nos modelos caret adicionais",
    x = "Subconjunto",
    y = "F1"
  ) +
  ggplot2::theme_minimal()

print(grafico_roc_balanceamento)
print(grafico_f1_balanceamento)

saveRDS(tabela_balanceamento, "objetos/tabela_modelos_caret_balanceamento_smotenc.rds")
readr::write_csv(tabela_balanceamento, "resultados/tabela_modelos_caret_balanceamento_smotenc.csv")

ggplot2::ggsave(
  filename = "figuras/roc_modelos_caret_balanceamento_smotenc.png",
  plot = grafico_roc_balanceamento,
  width = 12,
  height = 6
)

ggplot2::ggsave(
  filename = "figuras/f1_modelos_caret_balanceamento_smotenc.png",
  plot = grafico_f1_balanceamento,
  width = 12,
  height = 6
)

message("06C_modelos_caret_balanceamento_smotenc.R concluido com sucesso.")
