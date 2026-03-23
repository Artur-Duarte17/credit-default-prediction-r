# ==============================================================================
# 07_threshold_calibracao.R
# Responsabilidade: calibrar thresholds via OOF para todos os modelos finais.
# Observacao de negocio:
# Em credito, perder um inadimplente (falso negativo) tende a custar mais do que
# barrar um bom pagador (falso positivo). Por isso sensibilidade precisa ser
# acompanhada com destaque nas comparacoes de threshold e na selecao do modelo.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar treino e tabelas de benchmark
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(readRDS("objetos/treino.rds"))
tabela_benchmark <- readRDS("objetos/tabela_benchmark_modelos_sem_balanceamento.rds")

carregar_balanceamentos <- function() {
  caminhos <- c(
    "objetos/tabela_rf_balanceamento_smotenc.rds",
    "objetos/tabela_xgb_balanceamento_smotenc.rds",
    "objetos/tabela_modelos_caret_balanceamento_smotenc.rds"
  )

  existentes <- caminhos[file.exists(caminhos)]

  if (length(existentes) == 0) {
    return(NULL)
  }

  dplyr::bind_rows(lapply(existentes, readRDS))
}

tabela_balanceamento <- carregar_balanceamentos()

# ------------------------------------------------------------------------------
# BLOCO 2 - Montar cenarios candidatos
# ------------------------------------------------------------------------------
configs_base <- tabela_benchmark %>%
  dplyr::group_by(Modelo) %>%
  dplyr::arrange(dplyr::desc(ROC), dplyr::desc(F1), dplyr::desc(GMean), .by_group = TRUE) %>%
  dplyr::slice(1) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(
    Usa_SMOTENC = FALSE,
    Cenario = paste0(Modelo, "_", Subconjunto, "_SemBalanceamento")
  )

if (!is.null(tabela_balanceamento)) {
  configs_smotenc <- tabela_balanceamento %>%
    dplyr::filter(Cenario == "Com_SMOTENC") %>%
    dplyr::group_by(Modelo) %>%
    dplyr::arrange(dplyr::desc(ROC), dplyr::desc(F1), dplyr::desc(GMean), .by_group = TRUE) %>%
    dplyr::slice(1) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      Usa_SMOTENC = TRUE,
      Cenario = paste0(Modelo, "_", Subconjunto, "_ComSMOTENC")
    )
} else {
  configs_smotenc <- tibble::tibble()
}

configs_candidatos <- dplyr::bind_rows(configs_base, configs_smotenc) %>%
  dplyr::mutate(TopN = as.integer(gsub("^Top", "", Subconjunto))) %>%
  dplyr::arrange(Modelo, Usa_SMOTENC)

print(configs_candidatos)

# ------------------------------------------------------------------------------
# BLOCO 3 - Gerar predicoes OOF
# ------------------------------------------------------------------------------
gerar_predicoes_oof <- function(config_modelo, treino_df) {
  vars_modelo <- parse_variaveis(config_modelo$Variaveis[1])
  dados_sub <- treino_df[, c(vars_modelo, "Class"), drop = FALSE]
  formula_sub <- montar_formula(vars_modelo)

  if (config_modelo$Modelo[1] == "XGBoost") {
    set.seed(SEED_PROJETO)
    folds_xgb <- caret::createFolds(dados_sub$Class, k = CV_FOLDS_PADRAO, returnTrain = FALSE)

    resultado_xgb <- avaliar_xgb_oof(
      dados = dados_sub,
      folds = folds_xgb,
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
    cv_repeats = 1,
    save_predictions = TRUE
  )

  extrair_predicoes_oof_caret(modelo_oof)
}

# ------------------------------------------------------------------------------
# BLOCO 4 - Calibrar threshold por cenario
# ------------------------------------------------------------------------------
tabela_thresholds <- list()
config_modelos <- list()

for (i in seq_len(nrow(configs_candidatos))) {
  config_atual <- configs_candidatos[i, , drop = FALSE]
  pred_oof <- gerar_predicoes_oof(config_atual, treino_df = treino)
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
    title = "Impacto do threshold padrao vs Youden",
    subtitle = "Sensibilidade merece leitura prioritaria em credito",
    x = "Metrica",
    y = "Valor"
  ) +
  ggplot2::theme_minimal()

print(grafico_metricas_threshold)

# ------------------------------------------------------------------------------
# BLOCO 6 - Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(tabela_thresholds, "objetos/tabela_thresholds_finais.rds")
readr::write_csv(tabela_thresholds, "resultados/tabela_thresholds_finais.csv")

saveRDS(config_modelos, "objetos/config_modelos_finais.rds")
readr::write_csv(config_modelos, "resultados/config_modelos_finais.csv")

ggplot2::ggsave(
  filename = "figuras/comparacao_thresholds_finais.png",
  plot = grafico_metricas_threshold,
  width = 14,
  height = 8
)

message("07_threshold_calibracao.R concluido com sucesso.")
