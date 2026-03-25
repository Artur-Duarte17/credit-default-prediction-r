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
treino <- garantir_ordem_classe(ler_rds_base("treino.rds"))
tabela_xgb_confirmada <- ler_rds_saida(
  "confirmacao",
  "tabela_xgboost_subconjuntos_sem_balanceamento.rds",
  subpastas = "benchmark",
  legados = caminho_objeto_legado("tabela_xgboost_subconjuntos_sem_balanceamento.rds")
) %>%
  dplyr::filter(Modelo == "XGBoost")

finalistas_xgb <- selecionar_finalistas_modelagem(
  tabela = tabela_xgb_confirmada,
  n_por_modelo = N_FINALISTAS_BALANCEAMENTO_POR_MODELO
)

folds_confirmacao <- criar_folds_estratificados(
  y = treino$Class,
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
  grid_xgb_atual <- obter_grid_modelo_config(config_atual)

  cat("\n====================================================\n")
  cat("Confirmacao de balanceamento -", config_atual$Subconjunto[1], "\n")
  cat("Variaveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")

  tabela_base <- avaliar_xgb_cv(
    dados = dados_sub,
    folds = folds_confirmacao,
    grid_xgb = grid_xgb_atual,
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
    grid_xgb = grid_xgb_atual,
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

tabela_rf <- ler_rds_saida(
  "confirmacao",
  "tabela_rf_balanceamento_smotenc.rds",
  subpastas = "balanceamento",
  legados = caminho_objeto_legado("tabela_rf_balanceamento_smotenc.rds"),
  obrigatorio = FALSE
)
if (!is.null(tabela_rf)) {
  tabela_balanceamento_completa <- dplyr::bind_rows(tabela_rf, tabela_xgb_balanceamento) %>%
    dplyr::arrange(Modelo, Subconjunto, desc(ROC), desc(F1), desc(GMean))
} else {
  tabela_balanceamento_completa <- tabela_xgb_balanceamento
}

print(tabela_balanceamento_completa)

# ------------------------------------------------------------------------------
# BLOCO 4 - Salvar resultados
# ------------------------------------------------------------------------------
salvar_rds_saida(
  tabela_xgb_balanceamento,
  "confirmacao",
  "tabela_xgb_balanceamento_smotenc.rds",
  subpastas = "balanceamento"
)
salvar_csv_saida(
  tabela_xgb_balanceamento,
  "confirmacao",
  "tabela_xgb_balanceamento_smotenc.csv",
  subpastas = "balanceamento"
)

salvar_rds_saida(
  tabela_balanceamento_completa,
  "confirmacao",
  "tabela_rf_xgb_balanceamento_smotenc.rds",
  subpastas = "balanceamento"
)
salvar_csv_saida(
  tabela_balanceamento_completa,
  "confirmacao",
  "tabela_rf_xgb_balanceamento_smotenc.csv",
  subpastas = "balanceamento"
)

message("06B_xgb_balanceamento_smotenc.R concluido com sucesso.")
