# ==============================================================================
# 07_threshold_calibracao.R
# Responsabilidade: escolher threshold corretamente usando predições
# out-of-fold no treino, sem tocar no conjunto de teste.
# Inclui RF e XGBoost com e sem SMOTENC no Top-14.
# ==============================================================================

source("00_setup.R")

# ------------------------------------------------------------------------------
# BLOCO 1 — Carregar dados e ranking
# ------------------------------------------------------------------------------
treino <- readRDS("objetos/treino.rds")
ranking_variaveis <- readRDS("objetos/ranking_variaveis_enet.rds")

vars_top14 <- ranking_variaveis$Variavel_Original[1:14]
dados_sub <- treino[, c(vars_top14, "Class")]
formula_sub <- as.formula(
  paste("Class ~", paste(vars_top14, collapse = " + "))
)

print(vars_top14)

# ------------------------------------------------------------------------------
# BLOCO 2 — Funções auxiliares
# ------------------------------------------------------------------------------
metricas_binarias <- function(data, lev = NULL, model = NULL) {
  base_metrics <- caret::twoClassSummary(data, lev = lev, model = model)

  obs  <- factor(data$obs, levels = lev)
  pred <- factor(data$pred, levels = lev)

  cm <- table(pred, obs)

  TP <- cm[lev[1], lev[1]]
  FP <- cm[lev[1], lev[2]]
  FN <- cm[lev[2], lev[1]]
  TN <- cm[lev[2], lev[2]]

  Precision <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
  Recall    <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
  F1        <- ifelse(is.na(Precision) || is.na(Recall) || (Precision + Recall) == 0,
                      NA,
                      2 * Precision * Recall / (Precision + Recall))
  GMean     <- ifelse(is.na(base_metrics["Sens"]) || is.na(base_metrics["Spec"]),
                      NA,
                      sqrt(base_metrics["Sens"] * base_metrics["Spec"]))

  c(
    ROC = unname(base_metrics["ROC"]),
    Sens = unname(base_metrics["Sens"]),
    Spec = unname(base_metrics["Spec"]),
    Precision = unname(Precision),
    F1 = unname(F1),
    GMean = unname(GMean)
  )
}

controle_oof <- caret::trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = metricas_binarias,
  savePredictions = "final",
  allowParallel = FALSE
)

calcular_metricas_threshold <- function(obs, prob_deve, threshold, cenario, regra) {

  obs_factor <- factor(obs, levels = c("Deve", "Pago"))
  pred_class <- ifelse(prob_deve >= threshold, "Deve", "Pago")
  pred_factor <- factor(pred_class, levels = c("Deve", "Pago"))

  cm <- table(pred_factor, obs_factor)

  TP <- cm["Deve", "Deve"]
  FP <- cm["Deve", "Pago"]
  FN <- cm["Pago", "Deve"]
  TN <- cm["Pago", "Pago"]

  Sens <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
  Spec <- ifelse((TN + FP) == 0, NA, TN / (TN + FP))
  Precision <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
  F1 <- ifelse(is.na(Precision) || is.na(Sens) || (Precision + Sens) == 0,
               NA,
               2 * Precision * Sens / (Precision + Sens))
  GMean <- ifelse(is.na(Sens) || is.na(Spec), NA, sqrt(Sens * Spec))

  obs_roc <- factor(obs, levels = c("Pago", "Deve"))
  roc_obj <- pROC::roc(
    response = obs_roc,
    predictor = prob_deve,
    levels = c("Pago", "Deve"),
    direction = "<",
    quiet = TRUE
  )

  ROC <- as.numeric(pROC::auc(roc_obj))

  tibble::tibble(
    Cenario = cenario,
    RegraThreshold = regra,
    Threshold = threshold,
    ROC = ROC,
    Sens = Sens,
    Spec = Spec,
    Precision = Precision,
    F1 = F1,
    GMean = GMean
  )
}

encontrar_threshold_youden <- function(obs, prob_deve) {
  obs_roc <- factor(obs, levels = c("Pago", "Deve"))

  roc_obj <- pROC::roc(
    response = obs_roc,
    predictor = prob_deve,
    levels = c("Pago", "Deve"),
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

alinhar_colunas_matriz <- function(x_ref, x_new) {
  x_ref <- as.matrix(x_ref)
  x_new <- as.matrix(x_new)

  cols_faltantes <- setdiff(colnames(x_ref), colnames(x_new))
  if (length(cols_faltantes) > 0) {
    matriz_zero <- matrix(
      0,
      nrow = nrow(x_new),
      ncol = length(cols_faltantes),
      dimnames = list(NULL, cols_faltantes)
    )
    x_new <- cbind(x_new, matriz_zero)
  }

  cols_extras <- setdiff(colnames(x_new), colnames(x_ref))
  if (length(cols_extras) > 0) {
    x_new <- x_new[, !colnames(x_new) %in% cols_extras, drop = FALSE]
  }

  x_new[, colnames(x_ref), drop = FALSE]
}

avaliar_xgb_oof <- function(dados, folds, grid_xgb, aplicar_smotenc = FALSE, formula_modelo = NULL) {
  resultados_grid <- vector("list", nrow(grid_xgb))
  predicoes_grid <- vector("list", nrow(grid_xgb))

  for (i in seq_len(nrow(grid_xgb))) {
    params_grid <- grid_xgb[i, ]
    prob_oof <- rep(NA_real_, nrow(dados))
    metricas_folds <- vector("list", length(folds))

    for (j in seq_along(folds)) {
      idx_valid <- folds[[j]]
      treino_fold <- dados[-idx_valid, , drop = FALSE]
      valid_fold  <- dados[idx_valid, , drop = FALSE]

      if (aplicar_smotenc) {
        receita_fold <- recipes::recipe(formula_modelo, data = treino_fold) %>%
          themis::step_smotenc(Class, over_ratio = 1, neighbors = 5, skip = TRUE)

        prep_fold <- recipes::prep(receita_fold, training = treino_fold, retain = TRUE)

        treino_proc <- recipes::juice(prep_fold)
        valid_proc  <- recipes::bake(prep_fold, new_data = valid_fold)
      } else {
        treino_proc <- treino_fold
        valid_proc  <- valid_fold
      }

      x_treino <- model.matrix(Class ~ ., data = treino_proc)[, -1, drop = FALSE]
      x_valid  <- model.matrix(Class ~ ., data = valid_proc)[, -1, drop = FALSE]
      x_valid  <- alinhar_colunas_matriz(x_treino, x_valid)

      dtrain <- xgboost::xgb.DMatrix(
        data = x_treino,
        label = ifelse(treino_proc$Class == "Deve", 1, 0)
      )

      modelo_xgb <- xgboost::xgb.train(
        params = list(
          objective = "binary:logistic",
          eval_metric = "auc",
          max_depth = params_grid$max_depth,
          eta = params_grid$eta,
          gamma = params_grid$gamma,
          colsample_bytree = params_grid$colsample_bytree,
          min_child_weight = params_grid$min_child_weight,
          subsample = params_grid$subsample
        ),
        data = dtrain,
        nrounds = params_grid$nrounds,
        verbose = 0
      )

      prob_valid <- predict(modelo_xgb, newdata = x_valid)
      prob_oof[idx_valid] <- prob_valid

      metricas_folds[[j]] <- calcular_metricas_threshold(
        obs = valid_fold$Class,
        prob_deve = prob_valid,
        threshold = 0.50,
        cenario = "XGBoost_Auxiliar",
        regra = "Padrao_0.50"
      ) %>%
        dplyr::select(ROC, Sens, Spec, Precision, F1, GMean)
    }

    tabela_folds <- dplyr::bind_rows(metricas_folds)

    metricas_oof <- calcular_metricas_threshold(
      obs = dados$Class,
      prob_deve = prob_oof,
      threshold = 0.50,
      cenario = "XGBoost_Auxiliar",
      regra = "Padrao_0.50"
    )

    resultados_grid[[i]] <- dplyr::bind_cols(
      tibble::tibble(GridID = i),
      params_grid,
      metricas_oof %>% dplyr::select(ROC, Sens, Spec, Precision, F1, GMean),
      tibble::tibble(
        ROCSD = stats::sd(tabela_folds$ROC, na.rm = TRUE),
        SensSD = stats::sd(tabela_folds$Sens, na.rm = TRUE),
        SpecSD = stats::sd(tabela_folds$Spec, na.rm = TRUE),
        PrecisionSD = stats::sd(tabela_folds$Precision, na.rm = TRUE),
        F1SD = stats::sd(tabela_folds$F1, na.rm = TRUE),
        GMeanSD = stats::sd(tabela_folds$GMean, na.rm = TRUE)
      )
    )

    predicoes_grid[[i]] <- tibble::tibble(
      obs = dados$Class,
      Deve = prob_oof
    )
  }

  tabela_resultados <- dplyr::bind_rows(resultados_grid) %>%
    dplyr::arrange(desc(ROC), desc(F1), desc(GMean))

  melhor_id <- tabela_resultados$GridID[1]

  list(
    resultados = tabela_resultados,
    melhor_resultado = tabela_resultados %>% dplyr::slice(1),
    pred_melhor = predicoes_grid[[melhor_id]]
  )
}

# ------------------------------------------------------------------------------
# BLOCO 3 — RF sem balanceamento (OOF)
# ------------------------------------------------------------------------------
cat("\nTreinando RF Top-14 sem balanceamento com savePredictions...\n")

set.seed(123)
modelo_rf_base_oof <- caret::train(
  formula_sub,
  data = dados_sub,
  method = "rf",
  metric = "ROC",
  trControl = controle_oof,
  tuneGrid = data.frame(mtry = 3),
  importance = TRUE,
  ntree = 200
)

pred_base <- modelo_rf_base_oof$pred %>%
  dplyr::filter(mtry == 3) %>%
  dplyr::select(obs, Deve)

th_base <- encontrar_threshold_youden(
  obs = pred_base$obs,
  prob_deve = pred_base$Deve
)

cat("Threshold Youden - RF sem balanceamento:", th_base$threshold, "\n")

# ------------------------------------------------------------------------------
# BLOCO 4 — RF com SMOTENC (OOF)
# ------------------------------------------------------------------------------
cat("\nTreinando RF Top-14 com SMOTENC e savePredictions...\n")

receita_smotenc <- recipes::recipe(formula_sub, data = dados_sub) %>%
  themis::step_smotenc(Class, over_ratio = 1, neighbors = 5)

set.seed(123)
modelo_rf_smotenc_oof <- caret::train(
  receita_smotenc,
  data = dados_sub,
  method = "rf",
  metric = "ROC",
  trControl = controle_oof,
  tuneGrid = data.frame(mtry = 2),
  importance = TRUE,
  ntree = 200
)

pred_smotenc <- modelo_rf_smotenc_oof$pred %>%
  dplyr::filter(mtry == 2) %>%
  dplyr::select(obs, Deve)

th_smotenc <- encontrar_threshold_youden(
  obs = pred_smotenc$obs,
  prob_deve = pred_smotenc$Deve
)

cat("Threshold Youden - RF com SMOTENC:", th_smotenc$threshold, "\n")

# ------------------------------------------------------------------------------
# BLOCO 5 — XGBoost sem balanceamento (OOF manual)
# ------------------------------------------------------------------------------
grid_xgb <- expand.grid(
  nrounds = c(100, 150),
  max_depth = c(3, 5),
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

set.seed(123)
folds_xgb <- caret::createFolds(dados_sub$Class, k = 5, returnTrain = FALSE)

cat("\nTreinando XGBoost Top-14 sem balanceamento com OOF manual...\n")

set.seed(123)
resultado_xgb_base <- avaliar_xgb_oof(
  dados = dados_sub,
  folds = folds_xgb,
  grid_xgb = grid_xgb,
  aplicar_smotenc = FALSE,
  formula_modelo = formula_sub
)

pred_xgb_base <- resultado_xgb_base$pred_melhor
th_xgb_base <- encontrar_threshold_youden(
  obs = pred_xgb_base$obs,
  prob_deve = pred_xgb_base$Deve
)

cat("Melhor grid - XGBoost sem balanceamento:\n")
print(
  resultado_xgb_base$melhor_resultado %>%
    dplyr::select(nrounds, max_depth, eta, gamma, colsample_bytree,
                  min_child_weight, subsample, ROC, F1, GMean)
)
cat("Threshold Youden - XGBoost sem balanceamento:", th_xgb_base$threshold, "\n")

# ------------------------------------------------------------------------------
# BLOCO 6 — XGBoost com SMOTENC (OOF manual)
# ------------------------------------------------------------------------------
cat("\nTreinando XGBoost Top-14 com SMOTENC e OOF manual...\n")

set.seed(123)
resultado_xgb_smotenc <- avaliar_xgb_oof(
  dados = dados_sub,
  folds = folds_xgb,
  grid_xgb = grid_xgb,
  aplicar_smotenc = TRUE,
  formula_modelo = formula_sub
)

pred_xgb_smotenc <- resultado_xgb_smotenc$pred_melhor
th_xgb_smotenc <- encontrar_threshold_youden(
  obs = pred_xgb_smotenc$obs,
  prob_deve = pred_xgb_smotenc$Deve
)

cat("Melhor grid - XGBoost com SMOTENC:\n")
print(
  resultado_xgb_smotenc$melhor_resultado %>%
    dplyr::select(nrounds, max_depth, eta, gamma, colsample_bytree,
                  min_child_weight, subsample, ROC, F1, GMean)
)
cat("Threshold Youden - XGBoost com SMOTENC:", th_xgb_smotenc$threshold, "\n")

# ------------------------------------------------------------------------------
# BLOCO 7 — Comparar threshold padrão 0.50 vs Youden
# ------------------------------------------------------------------------------
tabela_thresholds <- dplyr::bind_rows(
  calcular_metricas_threshold(
    obs = pred_base$obs,
    prob_deve = pred_base$Deve,
    threshold = 0.50,
    cenario = "RF_Top14_SemBalanceamento",
    regra = "Padrao_0.50"
  ),
  calcular_metricas_threshold(
    obs = pred_base$obs,
    prob_deve = pred_base$Deve,
    threshold = th_base$threshold,
    cenario = "RF_Top14_SemBalanceamento",
    regra = "Youden"
  ),
  calcular_metricas_threshold(
    obs = pred_smotenc$obs,
    prob_deve = pred_smotenc$Deve,
    threshold = 0.50,
    cenario = "RF_Top14_ComSMOTENC",
    regra = "Padrao_0.50"
  ),
  calcular_metricas_threshold(
    obs = pred_smotenc$obs,
    prob_deve = pred_smotenc$Deve,
    threshold = th_smotenc$threshold,
    cenario = "RF_Top14_ComSMOTENC",
    regra = "Youden"
  ),
  calcular_metricas_threshold(
    obs = pred_xgb_base$obs,
    prob_deve = pred_xgb_base$Deve,
    threshold = 0.50,
    cenario = "XGB_Top14_SemBalanceamento",
    regra = "Padrao_0.50"
  ),
  calcular_metricas_threshold(
    obs = pred_xgb_base$obs,
    prob_deve = pred_xgb_base$Deve,
    threshold = th_xgb_base$threshold,
    cenario = "XGB_Top14_SemBalanceamento",
    regra = "Youden"
  ),
  calcular_metricas_threshold(
    obs = pred_xgb_smotenc$obs,
    prob_deve = pred_xgb_smotenc$Deve,
    threshold = 0.50,
    cenario = "XGB_Top14_ComSMOTENC",
    regra = "Padrao_0.50"
  ),
  calcular_metricas_threshold(
    obs = pred_xgb_smotenc$obs,
    prob_deve = pred_xgb_smotenc$Deve,
    threshold = th_xgb_smotenc$threshold,
    cenario = "XGB_Top14_ComSMOTENC",
    regra = "Youden"
  )
)

print(tabela_thresholds)

config_modelos <- dplyr::bind_rows(
  tibble::tibble(
    Cenario = "RF_Top14_SemBalanceamento",
    Modelo = "RF",
    Usa_SMOTENC = FALSE,
    Threshold_Youden = th_base$threshold,
    ROC_OOF = calcular_metricas_threshold(
      obs = pred_base$obs,
      prob_deve = pred_base$Deve,
      threshold = th_base$threshold,
      cenario = "RF_Top14_SemBalanceamento",
      regra = "Youden"
    )$ROC,
    F1_OOF = calcular_metricas_threshold(
      obs = pred_base$obs,
      prob_deve = pred_base$Deve,
      threshold = th_base$threshold,
      cenario = "RF_Top14_SemBalanceamento",
      regra = "Youden"
    )$F1,
    mtry = 3,
    nrounds = NA_real_,
    max_depth = NA_real_,
    eta = NA_real_,
    gamma = NA_real_,
    colsample_bytree = NA_real_,
    min_child_weight = NA_real_,
    subsample = NA_real_
  ),
  tibble::tibble(
    Cenario = "RF_Top14_ComSMOTENC",
    Modelo = "RF",
    Usa_SMOTENC = TRUE,
    Threshold_Youden = th_smotenc$threshold,
    ROC_OOF = calcular_metricas_threshold(
      obs = pred_smotenc$obs,
      prob_deve = pred_smotenc$Deve,
      threshold = th_smotenc$threshold,
      cenario = "RF_Top14_ComSMOTENC",
      regra = "Youden"
    )$ROC,
    F1_OOF = calcular_metricas_threshold(
      obs = pred_smotenc$obs,
      prob_deve = pred_smotenc$Deve,
      threshold = th_smotenc$threshold,
      cenario = "RF_Top14_ComSMOTENC",
      regra = "Youden"
    )$F1,
    mtry = 2,
    nrounds = NA_real_,
    max_depth = NA_real_,
    eta = NA_real_,
    gamma = NA_real_,
    colsample_bytree = NA_real_,
    min_child_weight = NA_real_,
    subsample = NA_real_
  ),
  resultado_xgb_base$melhor_resultado %>%
    dplyr::transmute(
      Cenario = "XGB_Top14_SemBalanceamento",
      Modelo = "XGBoost",
      Usa_SMOTENC = FALSE,
      Threshold_Youden = th_xgb_base$threshold,
      ROC_OOF = ROC,
      F1_OOF = F1,
      mtry = NA_real_,
      nrounds = nrounds,
      max_depth = max_depth,
      eta = eta,
      gamma = gamma,
      colsample_bytree = colsample_bytree,
      min_child_weight = min_child_weight,
      subsample = subsample
    ),
  resultado_xgb_smotenc$melhor_resultado %>%
    dplyr::transmute(
      Cenario = "XGB_Top14_ComSMOTENC",
      Modelo = "XGBoost",
      Usa_SMOTENC = TRUE,
      Threshold_Youden = th_xgb_smotenc$threshold,
      ROC_OOF = ROC,
      F1_OOF = F1,
      mtry = NA_real_,
      nrounds = nrounds,
      max_depth = max_depth,
      eta = eta,
      gamma = gamma,
      colsample_bytree = colsample_bytree,
      min_child_weight = min_child_weight,
      subsample = subsample
    )
)

print(config_modelos)

# ------------------------------------------------------------------------------
# BLOCO 8 — Gráfico simples das métricas
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
  aes(x = Metrica, y = Valor, fill = RegraThreshold)
) +
  ggplot2::geom_col(position = "dodge") +
  ggplot2::facet_wrap(~ Cenario) +
  ggplot2::labs(
    title = "Impacto do threshold padrão vs Youden",
    x = "Métrica",
    y = "Valor"
  ) +
  ggplot2::theme_minimal()

print(grafico_metricas_threshold)

# ------------------------------------------------------------------------------
# BLOCO 9 — Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(tabela_thresholds, "objetos/tabela_thresholds_top14.rds")
readr::write_csv(tabela_thresholds, "resultados/tabela_thresholds_top14.csv")

saveRDS(config_modelos, "objetos/config_modelos_top14.rds")
readr::write_csv(config_modelos, "resultados/config_modelos_top14.csv")

ggplot2::ggsave(
  filename = "figuras/comparacao_thresholds_top14.png",
  plot = grafico_metricas_threshold,
  width = 12,
  height = 7
)

message("07_threshold_calibracao.R concluído com sucesso.")
