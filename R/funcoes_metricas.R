# ==============================================================================
# R/funcoes_metricas.R
# Responsabilidade: concentrar metricas de classificacao e negocio.
# ==============================================================================

construir_confusao_binaria <- function(obs, pred, positivo = "Deve", negativo = "Pago") {
  obs_factor <- factor(obs, levels = c(positivo, negativo))
  pred_factor <- factor(pred, levels = c(positivo, negativo))
  cm <- table(pred_factor, obs_factor)

  list(
    TP = as.numeric(cm[positivo, positivo]),
    FP = as.numeric(cm[positivo, negativo]),
    FN = as.numeric(cm[negativo, positivo]),
    TN = as.numeric(cm[negativo, negativo])
  )
}

calcular_estatisticas_confusao <- function(TP, FP, FN, TN) {
  sens <- ifelse((TP + FN) == 0, NA_real_, TP / (TP + FN))
  spec <- ifelse((TN + FP) == 0, NA_real_, TN / (TN + FP))
  precision <- ifelse((TP + FP) == 0, NA_real_, TP / (TP + FP))
  f1 <- ifelse(
    is.na(precision) || is.na(sens) || (precision + sens) == 0,
    NA_real_,
    2 * precision * sens / (precision + sens)
  )
  gmean <- ifelse(is.na(sens) || is.na(spec), NA_real_, sqrt(sens * spec))
  accuracy <- ifelse((TP + FP + FN + TN) == 0, NA_real_, (TP + TN) / (TP + FP + FN + TN))

  tibble::tibble(
    Sens = sens,
    Spec = spec,
    Precision = precision,
    F1 = f1,
    GMean = gmean,
    Accuracy = accuracy
  )
}

calcular_auc_roc <- function(obs, prob_deve, positivo = "Deve", negativo = "Pago") {
  obs_roc <- factor(obs, levels = c(negativo, positivo))

  tryCatch(
    {
      roc_obj <- pROC::roc(
        response = obs_roc,
        predictor = prob_deve,
        levels = c(negativo, positivo),
        direction = "<",
        quiet = TRUE
      )

      as.numeric(pROC::auc(roc_obj))
    },
    error = function(...) NA_real_
  )
}

calcular_auc_pr <- function(obs, prob_deve, positivo = "Deve", negativo = "Pago") {
  if (!any(obs == positivo) || !any(obs == negativo)) {
    return(NA_real_)
  }

  tryCatch(
    {
      pr_obj <- PRROC::pr.curve(
        scores.class0 = prob_deve[obs == positivo],
        scores.class1 = prob_deve[obs == negativo],
        curve = FALSE
      )

      pr_obj$auc.integral
    },
    error = function(...) NA_real_
  )
}

metricas_binarias <- function(data, lev = NULL, model = NULL) {
  if (is.null(lev)) {
    lev <- levels(data$obs)
  }

  base_metrics <- caret::twoClassSummary(data, lev = lev, model = model)
  conf <- construir_confusao_binaria(
    obs = data$obs,
    pred = data$pred,
    positivo = lev[1],
    negativo = lev[2]
  )
  estat <- calcular_estatisticas_confusao(
    TP = conf$TP,
    FP = conf$FP,
    FN = conf$FN,
    TN = conf$TN
  )

  c(
    ROC = unname(base_metrics["ROC"]),
    Sens = estat$Sens,
    Spec = estat$Spec,
    Precision = estat$Precision,
    F1 = estat$F1,
    GMean = estat$GMean
  )
}

calcular_metricas_prob <- function(
  obs,
  prob_deve,
  threshold = 0.50,
  cenario = NA_character_,
  regra = NA_character_,
  positivo = "Deve",
  negativo = "Pago"
) {
  pred_class <- ifelse(prob_deve >= threshold, positivo, negativo)
  conf <- construir_confusao_binaria(
    obs = obs,
    pred = pred_class,
    positivo = positivo,
    negativo = negativo
  )
  estat <- calcular_estatisticas_confusao(
    TP = conf$TP,
    FP = conf$FP,
    FN = conf$FN,
    TN = conf$TN
  )

  tibble::tibble(
    Cenario = cenario,
    RegraThreshold = regra,
    Threshold = threshold,
    ROC = calcular_auc_roc(obs = obs, prob_deve = prob_deve, positivo = positivo, negativo = negativo),
    PRAUC = calcular_auc_pr(obs = obs, prob_deve = prob_deve, positivo = positivo, negativo = negativo),
    Sens = estat$Sens,
    Spec = estat$Spec,
    Precision = estat$Precision,
    F1 = estat$F1,
    GMean = estat$GMean,
    Accuracy = estat$Accuracy,
    TP = conf$TP,
    FP = conf$FP,
    FN = conf$FN,
    TN = conf$TN
  )
}

encontrar_threshold_youden <- function(obs, prob_deve, positivo = "Deve", negativo = "Pago") {
  obs_roc <- factor(obs, levels = c(negativo, positivo))

  roc_obj <- pROC::roc(
    response = obs_roc,
    predictor = prob_deve,
    levels = c(negativo, positivo),
    direction = "<",
    quiet = TRUE
  )

  coords_best <- pROC::coords(
    roc = roc_obj,
    x = "best",
    best.method = "youden",
    ret = c("threshold", "sensitivity", "specificity"),
    transpose = FALSE
  )

  list(
    roc_obj = roc_obj,
    threshold = as.numeric(coords_best["threshold"]),
    sensitivity = as.numeric(coords_best["sensitivity"]),
    specificity = as.numeric(coords_best["specificity"])
  )
}

adicionar_metricas_negocio <- function(
  tabela_metricas,
  custo_falso_negativo = CUSTO_FALSO_NEGATIVO,
  custo_falso_positivo = CUSTO_FALSO_POSITIVO,
  beneficio_verdadeiro_positivo = BENEFICIO_VERDADEIRO_POSITIVO
) {
  tabela_metricas %>%
    dplyr::mutate(
      Custo_FN = FN * custo_falso_negativo,
      Custo_FP = FP * custo_falso_positivo,
      Beneficio_TP = TP * beneficio_verdadeiro_positivo,
      Custo_Erro_Total = Custo_FN + Custo_FP,
      Resultado_Liquido = Beneficio_TP - Custo_FN - Custo_FP
    )
}

criar_tabela_shap_vs_enet <- function(tabela_shap, ranking_enet) {
  tabela_shap %>%
    dplyr::rename(
      Variavel = Variavel,
      Posicao_SHAP = Posicao,
      Importancia_SHAP = Importancia_SHAP
    ) %>%
    dplyr::full_join(
      ranking_enet %>%
        dplyr::transmute(
          Variavel = Variavel_Original,
          Posicao_ElasticNet = Posicao,
          Importancia_ElasticNet = Importancia
        ),
      by = "Variavel"
    ) %>%
    dplyr::mutate(
      Delta_Posicao = Posicao_ElasticNet - Posicao_SHAP
    ) %>%
    dplyr::arrange(dplyr::coalesce(Posicao_SHAP, Posicao_ElasticNet), Variavel)
}
