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
treino <- garantir_ordem_classe(ler_rds_base("treino.rds"))
tabela_rf_confirmada <- ler_rds_saida(
  "confirmacao",
  "tabela_benchmark_glm_rf_sem_balanceamento.rds",
  subpastas = "benchmark",
  legados = caminho_objeto_legado("tabela_benchmark_glm_rf_sem_balanceamento.rds")
) %>%
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
# BLOCO 4 - Salvar resultados
# ------------------------------------------------------------------------------
salvar_rds_saida(
  tabela_balanceamento,
  "confirmacao",
  "tabela_rf_balanceamento_smotenc.rds",
  subpastas = "balanceamento"
)
salvar_csv_saida(
  tabela_balanceamento,
  "confirmacao",
  "tabela_rf_balanceamento_smotenc.csv",
  subpastas = "balanceamento"
)

message("06_rf_balanceamento_smotenc.R concluido com sucesso.")
