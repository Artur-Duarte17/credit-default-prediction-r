# ==============================================================================
# 07_threshold_calibracao.R
# Responsabilidade: calibrar thresholds via OOF apenas para os finalistas
# confirmados. A selecao do threshold continua fora do teste final.
# Observacao de negocio:
# Em credito, perder um inadimplente (falso negativo) tende a custar mais do que
# barrar um bom pagador (falso positivo). Por isso sensibilidade precisa ser
# acompanhada com destaque nas comparacoes de threshold e na selecao do modelo.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar treino e tabelas confirmadas
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(ler_rds_base("treino.rds"))
tabela_benchmark <- ler_rds_saida(
  "confirmacao",
  "tabela_benchmark_modelos_sem_balanceamento.rds",
  subpastas = "benchmark",
  legados = caminho_objeto_legado("tabela_benchmark_modelos_sem_balanceamento.rds")
)

carregar_balanceamentos <- function() {
  tabelas <- list(
    ler_rds_saida(
      "confirmacao",
      "tabela_rf_balanceamento_smotenc.rds",
      subpastas = "balanceamento",
      legados = caminho_objeto_legado("tabela_rf_balanceamento_smotenc.rds"),
      obrigatorio = FALSE
    ),
    ler_rds_saida(
      "confirmacao",
      "tabela_xgb_balanceamento_smotenc.rds",
      subpastas = "balanceamento",
      legados = caminho_objeto_legado("tabela_xgb_balanceamento_smotenc.rds"),
      obrigatorio = FALSE
    ),
    ler_rds_saida(
      "confirmacao",
      "tabela_modelos_caret_balanceamento_smotenc.rds",
      subpastas = "balanceamento",
      legados = caminho_objeto_legado("tabela_modelos_caret_balanceamento_smotenc.rds"),
      obrigatorio = FALSE
    )
  )

  tabelas <- Filter(Negate(is.null), tabelas)

  if (length(tabelas) == 0) {
    return(NULL)
  }

  dplyr::bind_rows(tabelas)
}

tabela_balanceamento <- carregar_balanceamentos()
folds_confirmacao <- criar_folds_estratificados(
  y = treino$Class,
  fase = "confirmacao"
)

# ------------------------------------------------------------------------------
# BLOCO 2 - Selecionar finalistas confirmados por modelo
# ------------------------------------------------------------------------------
configs_base <- selecionar_melhor_cenario_por_modelo(tabela_benchmark) %>%
  dplyr::mutate(
    Usa_SMOTENC = FALSE,
    Origem_Finalista = "Confirmacao_SemBalanceamento",
    Cenario = paste0(Modelo, "_", Subconjunto, "_SemBalanceamento")
  )

if (!is.null(tabela_balanceamento)) {
  configs_smotenc <- tabela_balanceamento %>%
    dplyr::filter(Cenario == "Com_SMOTENC", is.finite(ROC)) %>%
    selecionar_melhor_cenario_por_modelo() %>%
    dplyr::mutate(
      Usa_SMOTENC = TRUE,
      Origem_Finalista = "Confirmacao_ComSMOTENC",
      Cenario = paste0(Modelo, "_", Subconjunto, "_ComSMOTENC")
    )
} else {
  configs_smotenc <- tibble::tibble()
}

configs_finalistas <- dplyr::bind_rows(configs_base, configs_smotenc) %>%
  selecionar_melhor_cenario_por_modelo() %>%
  dplyr::arrange(Modelo)

print(configs_finalistas)

# ------------------------------------------------------------------------------
# BLOCO 3 - Gerar predicoes OOF nos finalistas
# ------------------------------------------------------------------------------
gerar_predicoes_oof <- function(config_modelo, treino_df, folds_cv) {
  vars_modelo <- parse_variaveis(config_modelo$Variaveis[1])
  dados_sub <- treino_df[, c(vars_modelo, "Class"), drop = FALSE]
  formula_sub <- montar_formula(vars_modelo)

  if (config_modelo$Modelo[1] == "XGBoost") {
    resultado_xgb <- avaliar_xgb_oof(
      dados = dados_sub,
      folds = folds_cv,
      grid_xgb = obter_grid_modelo_config(config_modelo),
      aplicar_smotenc = isTRUE(config_modelo$Usa_SMOTENC[1]),
      formula_modelo = formula_sub
    )

    return(resultado_xgb$pred_melhor)
  }

  modelo_oof <- treinar_modelo_caret(
    modelo = config_modelo$Modelo[1],
    formula_modelo = formula_sub,
    dados_sub = dados_sub,
    tune_grid = obter_grid_modelo_config(config_modelo),
    usar_smotenc = isTRUE(config_modelo$Usa_SMOTENC[1]),
    save_predictions = TRUE,
    fase_validacao = "confirmacao",
    folds_cv = folds_cv
  )

  extrair_predicoes_oof_caret(modelo_oof)
}

# ------------------------------------------------------------------------------
# BLOCO 4 - Calibrar threshold por finalista
# ------------------------------------------------------------------------------
tabela_thresholds <- list()
config_modelos <- list()

for (i in seq_len(nrow(configs_finalistas))) {
  config_atual <- configs_finalistas[i, , drop = FALSE]
  pred_oof <- gerar_predicoes_oof(
    config_modelo = config_atual,
    treino_df = treino,
    folds_cv = folds_confirmacao
  )

  threshold_youden <- encontrar_threshold_youden(
    obs = pred_oof$obs,
    prob_deve = pred_oof$Deve
  )

  tabela_thresholds[[length(tabela_thresholds) + 1]] <- calcular_metricas_prob(
    obs = pred_oof$obs,
    prob_deve = pred_oof$Deve,
    threshold = 0.50,
    cenario = config_atual$Cenario[1],
    regra = "Padrao_0.50"
  )

  metricas_youden <- calcular_metricas_prob(
    obs = pred_oof$obs,
    prob_deve = pred_oof$Deve,
    threshold = threshold_youden$threshold,
    cenario = config_atual$Cenario[1],
    regra = "Youden"
  )

  tabela_thresholds[[length(tabela_thresholds) + 1]] <- metricas_youden

  config_modelos[[length(config_modelos) + 1]] <- config_atual %>%
    dplyr::mutate(
      Threshold_Youden = threshold_youden$threshold,
      ROC_OOF = metricas_youden$ROC,
      PRAUC_OOF = metricas_youden$PRAUC,
      Sens_OOF = metricas_youden$Sens,
      Spec_OOF = metricas_youden$Spec,
      F1_OOF = metricas_youden$F1,
      GMean_OOF = metricas_youden$GMean
    )
}

tabela_thresholds <- dplyr::bind_rows(tabela_thresholds) %>%
  dplyr::arrange(Cenario, RegraThreshold)

config_modelos <- dplyr::bind_rows(config_modelos) %>%
  dplyr::arrange(desc(Sens_OOF), desc(ROC_OOF), desc(F1_OOF), Modelo)

print(tabela_thresholds)
print(config_modelos)

# ------------------------------------------------------------------------------
# BLOCO 5 - Grafico comparativo
# ------------------------------------------------------------------------------
tabela_longa <- tabela_thresholds %>%
  dplyr::select(Cenario, RegraThreshold, Sens, Spec, Precision, F1, GMean) %>%
  tidyr::pivot_longer(
    cols = c(Sens, Spec, Precision, F1, GMean),
    names_to = "Metrica",
    values_to = "Valor"
  )

grafico_metricas_threshold <- ggplot2::ggplot(
  tabela_longa,
  ggplot2::aes(x = Metrica, y = Valor, fill = RegraThreshold)
) +
  ggplot2::geom_col(position = "dodge") +
  ggplot2::facet_wrap(~ Cenario) +
  ggplot2::labs(
    title = "Impacto do threshold padrao vs Youden nos finalistas",
    subtitle = "Sensibilidade merece leitura prioritaria em credito",
    x = "Metrica",
    y = "Valor"
  ) +
  ggplot2::theme_minimal()

print(grafico_metricas_threshold)

# ------------------------------------------------------------------------------
# BLOCO 6 - Salvar resultados
# ------------------------------------------------------------------------------
salvar_rds_saida(tabela_thresholds, "final", "tabela_thresholds_finais.rds", subpastas = "threshold")
salvar_csv_saida(tabela_thresholds, "final", "tabela_thresholds_finais.csv", subpastas = "threshold")

salvar_rds_saida(
  configs_finalistas,
  "final",
  "config_modelos_finalistas_confirmacao.rds",
  subpastas = "threshold"
)
salvar_csv_saida(
  configs_finalistas,
  "final",
  "config_modelos_finalistas_confirmacao.csv",
  subpastas = "threshold"
)

salvar_rds_saida(config_modelos, "final", "config_modelos_finais.rds", subpastas = "threshold")
salvar_csv_saida(config_modelos, "final", "config_modelos_finais.csv", subpastas = "threshold")

salvar_figura_saida(
  plot = grafico_metricas_threshold,
  fase = "final",
  arquivo = "comparacao_thresholds_principal.png",
  subpastas = "threshold",
  classificacao = "principal",
  width = 14,
  height = 8
)

message("07_threshold_calibracao.R concluido com sucesso.")
