# ==============================================================================
# 06C_modelos_caret_balanceamento_smotenc.R
# Responsabilidade: fase de confirmacao para SVM radial, NNET e avNNet,
# comparando sem balanceamento versus com SMOTENC apenas nos subconjuntos
# finalistas de cada modelo. O passo step_smotenc() fica sempre com skip = TRUE.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar dados e finalistas confirmados
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(readRDS("objetos/treino.rds"))
modelos_balancear <- c("SVM_Radial", "NNET", "avNNet")

tabela_sem_balanceamento <- dplyr::bind_rows(
  readRDS("objetos/tabela_svm_subconjuntos_sem_balanceamento.rds"),
  readRDS("objetos/tabela_redes_neurais_subconjuntos_sem_balanceamento.rds")
) %>%
  dplyr::filter(Modelo %in% modelos_balancear)

finalistas_modelos <- selecionar_finalistas_modelagem(
  tabela = tabela_sem_balanceamento,
  n_por_modelo = N_FINALISTAS_BALANCEAMENTO_POR_MODELO
)

folds_confirmacao <- criar_folds_estratificados(
  y = treino$Class,
  fase = "confirmacao"
)

print(finalistas_modelos)

resultados <- list()
contador <- 1

# ------------------------------------------------------------------------------
# BLOCO 2 - Confirmacao do balanceamento nos finalistas
# ------------------------------------------------------------------------------
for (i in seq_len(nrow(finalistas_modelos))) {
  config_atual <- finalistas_modelos[i, , drop = FALSE]
  vars_sub <- parse_variaveis(config_atual$Variaveis[1])
  dados_sub <- treino[, c(vars_sub, "Class"), drop = FALSE]
  formula_sub <- montar_formula(vars_sub)

  cat("\n====================================================\n")
  cat("Confirmacao de balanceamento -", config_atual$Modelo[1], "-", config_atual$Subconjunto[1], "\n")
  cat("Variaveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")

  modelo_base <- treinar_modelo_caret(
    modelo = config_atual$Modelo[1],
    formula_modelo = formula_sub,
    dados_sub = dados_sub,
    usar_smotenc = FALSE,
    fase_validacao = "confirmacao",
    folds_cv = folds_confirmacao
  )

  resultados[[contador]] <- extrair_melhor_resultado_caret(
    modelo_base,
    metadata = list(
      Subconjunto = config_atual$Subconjunto[1],
      TopN = config_atual$TopN[1],
      Modelo = config_atual$Modelo[1],
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

  modelo_smotenc <- treinar_modelo_caret(
    modelo = config_atual$Modelo[1],
    formula_modelo = formula_sub,
    dados_sub = dados_sub,
    usar_smotenc = TRUE,
    fase_validacao = "confirmacao",
    folds_cv = folds_confirmacao
  )

  resultados[[contador]] <- extrair_melhor_resultado_caret(
    modelo_smotenc,
    metadata = list(
      Subconjunto = config_atual$Subconjunto[1],
      TopN = config_atual$TopN[1],
      Modelo = config_atual$Modelo[1],
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

tabela_balanceamento <- dplyr::bind_rows(resultados) %>%
  dplyr::select(
    Subconjunto, TopN, Modelo, Cenario,
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
    title = "ROC: confirmacao do SMOTENC nos modelos caret adicionais",
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
    title = "F1: confirmacao do SMOTENC nos modelos caret adicionais",
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
