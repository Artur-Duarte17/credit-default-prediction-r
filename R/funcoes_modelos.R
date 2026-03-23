# ==============================================================================
# R/funcoes_modelos.R
# Responsabilidade: grids, controles, treino e avaliacao de modelos.
# ==============================================================================

source("R/funcoes_metricas.R")
source("R/funcoes_preprocessamento.R")

controle_cv_padrao <- function(
  number = CV_FOLDS_PADRAO,
  repeats = CV_REPEATS_PADRAO,
  save_predictions = FALSE
) {
  args <- list(
    method = if (repeats > 1) "repeatedcv" else "cv",
    number = number,
    classProbs = TRUE,
    summaryFunction = metricas_binarias,
    savePredictions = if (save_predictions) "final" else FALSE,
    allowParallel = FALSE
  )

  if (repeats > 1) {
    args$repeats <- repeats
  }

  do.call(caret::trainControl, args)
}

controle_final_sem_cv <- function() {
  caret::trainControl(
    method = "none",
    classProbs = TRUE,
    summaryFunction = metricas_binarias,
    allowParallel = FALSE
  )
}

grid_rf_padrao <- function(p) {
  data.frame(
    mtry = sort(unique(c(
      max(1, floor(sqrt(p)) - 1),
      max(1, floor(sqrt(p))),
      min(p, floor(sqrt(p)) + 1)
    )))
  )
}

grid_svm_padrao <- function(dados_sub, c_values = c(0.25, 0.5, 1, 2, 4)) {
  x_sub <- model.matrix(Class ~ ., data = dados_sub)[, -1, drop = FALSE]

  sigma_bruto <- tryCatch(
    as.numeric(kernlab::sigest(x_sub, frac = 1)),
    error = function(...) rep(NA_real_, 3)
  )

  sigma_vals <- sort(unique(signif(sigma_bruto[is.finite(sigma_bruto) & sigma_bruto > 0], 6)))

  if (length(sigma_vals) < 3) {
    base_sigma <- sigma_vals[1]

    if (!length(base_sigma) || !is.finite(base_sigma) || base_sigma <= 0) {
      base_sigma <- 0.05
    }

    sigma_vals <- sort(unique(signif(base_sigma * c(0.5, 1, 2), 6)))
  }

  if (length(sigma_vals) > 3) {
    sigma_vals <- sigma_vals[c(1, ceiling(length(sigma_vals) / 2), length(sigma_vals))]
  }

  expand.grid(
    sigma = sigma_vals,
    C = c_values
  )
}

grid_xgb_padrao <- function() {
  expand.grid(
    nrounds = c(100, 150, 200),
    max_depth = c(3, 5),
    eta = c(0.05, 0.10),
    gamma = 0,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    subsample = 0.8
  )
}

grid_nnet_padrao <- function() {
  expand.grid(
    size = c(3, 5, 7),
    decay = c(0.001, 0.01, 0.10)
  )
}

grid_avnnet_padrao <- function() {
  expand.grid(
    size = c(3, 5, 7),
    decay = c(0.001, 0.01, 0.10),
    bag = FALSE
  )
}

obter_especificacao_modelo <- function(modelo) {
  switch(
    modelo,
    GLM = list(
      tipo = "caret",
      method = "glm",
      preProcess = NULL,
      train_args = list(family = binomial())
    ),
    RF = list(
      tipo = "caret",
      method = "rf",
      preProcess = NULL,
      train_args = list(importance = TRUE, ntree = 200)
    ),
    SVM_Radial = list(
      tipo = "caret",
      method = "svmRadial",
      preProcess = c("center", "scale"),
      train_args = list()
    ),
    NNET = list(
      tipo = "caret",
      method = "nnet",
      preProcess = c("center", "scale"),
      train_args = list(trace = FALSE, maxit = 200, MaxNWts = 10000)
    ),
    avNNet = list(
      tipo = "caret",
      method = "avNNet",
      preProcess = c("center", "scale"),
      train_args = list(trace = FALSE, maxit = 200, repeats = 1, MaxNWts = 10000)
    ),
    XGBoost = list(
      tipo = "xgb",
      method = "xgb"
    ),
    stop(sprintf("Modelo nao suportado: %s", modelo))
  )
}

obter_grid_modelo_padrao <- function(modelo, dados_sub) {
  p <- ncol(dados_sub) - 1

  switch(
    modelo,
    GLM = NULL,
    RF = grid_rf_padrao(p = p),
    SVM_Radial = grid_svm_padrao(dados_sub = dados_sub),
    NNET = grid_nnet_padrao(),
    avNNet = grid_avnnet_padrao(),
    XGBoost = grid_xgb_padrao(),
    stop(sprintf("Sem grid padrao para o modelo: %s", modelo))
  )
}

obter_grid_modelo_config <- function(config_modelo) {
  modelo <- as.character(config_modelo$Modelo[1])

  switch(
    modelo,
    GLM = NULL,
    RF = data.frame(mtry = as.integer(config_modelo$mtry[1])),
    SVM_Radial = data.frame(
      sigma = as.numeric(config_modelo$sigma[1]),
      C = as.numeric(config_modelo$C[1])
    ),
    NNET = data.frame(
      size = as.integer(config_modelo$size[1]),
      decay = as.numeric(config_modelo$decay[1])
    ),
    avNNet = data.frame(
      size = as.integer(config_modelo$size[1]),
      decay = as.numeric(config_modelo$decay[1]),
      bag = as.logical(config_modelo$bag[1])
    ),
    XGBoost = data.frame(
      nrounds = as.integer(config_modelo$nrounds[1]),
      max_depth = as.integer(config_modelo$max_depth[1]),
      eta = as.numeric(config_modelo$eta[1]),
      gamma = as.numeric(config_modelo$gamma[1]),
      colsample_bytree = as.numeric(config_modelo$colsample_bytree[1]),
      min_child_weight = as.numeric(config_modelo$min_child_weight[1]),
      subsample = as.numeric(config_modelo$subsample[1])
    ),
    stop(sprintf("Sem grid para config do modelo: %s", modelo))
  )
}

treinar_modelo_caret <- function(
  modelo,
  formula_modelo,
  dados_sub,
  tune_grid = NULL,
  usar_smotenc = FALSE,
  cv_number = CV_FOLDS_PADRAO,
  cv_repeats = CV_REPEATS_PADRAO,
  save_predictions = FALSE
) {
  dados_sub <- garantir_ordem_classe(dados_sub)
  spec <- obter_especificacao_modelo(modelo)

  if (is.null(tune_grid)) {
    tune_grid <- obter_grid_modelo_padrao(modelo = modelo, dados_sub = dados_sub)
  }

  objeto_treino <- if (usar_smotenc) {
    preparar_receita_smotenc(formula_modelo = formula_modelo, dados = dados_sub)
  } else {
    formula_modelo
  }

  args <- c(
    list(
      objeto_treino,
      data = dados_sub,
      method = spec$method,
      metric = "ROC",
      trControl = controle_cv_padrao(
        number = cv_number,
        repeats = cv_repeats,
        save_predictions = save_predictions
      )
    ),
    spec$train_args
  )

  if (!is.null(spec$preProcess)) {
    args$preProcess <- spec$preProcess
  }

  if (!is.null(tune_grid)) {
    args$tuneGrid <- tune_grid
  }

  set.seed(SEED_PROJETO)
  do.call(caret::train, args)
}

extrair_melhor_resultado_caret <- function(modelo, metadata = list()) {
  res <- tibble::as_tibble(modelo$results)

  if (!is.null(modelo$bestTune)) {
    for (coluna in names(modelo$bestTune)) {
      res <- res[res[[coluna]] == modelo$bestTune[[coluna]], , drop = FALSE]
    }
  }

  for (nome in names(metadata)) {
    res[[nome]] <- metadata[[nome]]
  }

  res
}

extrair_predicoes_oof_caret <- function(modelo) {
  pred <- tibble::as_tibble(modelo$pred)

  if (!is.null(modelo$bestTune)) {
    for (coluna in names(modelo$bestTune)) {
      pred <- pred[pred[[coluna]] == modelo$bestTune[[coluna]], , drop = FALSE]
    }
  }

  if ("rowIndex" %in% names(pred)) {
    pred <- pred %>%
      dplyr::group_by(rowIndex, obs) %>%
      dplyr::summarise(Deve = mean(Deve, na.rm = TRUE), .groups = "drop")
  } else {
    pred <- pred %>%
      dplyr::mutate(rowIndex = dplyr::row_number()) %>%
      dplyr::select(rowIndex, obs, Deve)
  }

  pred %>%
    dplyr::arrange(rowIndex)
}

avaliar_xgb_cv <- function(dados, folds, grid_xgb, aplicar_smotenc = FALSE, formula_modelo = NULL) {
  resultados <- vector("list", nrow(grid_xgb))

  for (i in seq_len(nrow(grid_xgb))) {
    params_grid <- grid_xgb[i, , drop = FALSE]

    metricas_folds <- lapply(folds, function(idx_valid) {
      treino_fold <- dados[-idx_valid, , drop = FALSE]
      valid_fold <- dados[idx_valid, , drop = FALSE]

      if (aplicar_smotenc) {
        dados_proc <- preparar_treino_teste_modelo(
          treino_df = treino_fold,
          teste_df = valid_fold,
          formula_modelo = formula_modelo,
          usar_smotenc = TRUE
        )

        treino_proc <- dados_proc$treino
        valid_proc <- dados_proc$teste
      } else {
        treino_proc <- treino_fold
        valid_proc <- valid_fold
      }

      x_treino <- model.matrix(Class ~ ., data = treino_proc)[, -1, drop = FALSE]
      x_valid <- model.matrix(Class ~ ., data = valid_proc)[, -1, drop = FALSE]
      x_valid <- alinhar_colunas_matriz(x_treino, x_valid)

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

      calcular_metricas_prob(
        obs = valid_fold$Class,
        prob_deve = prob_valid,
        threshold = 0.50
      )
    })

    tabela_folds <- dplyr::bind_rows(metricas_folds)

    resultados[[i]] <- dplyr::bind_cols(
      params_grid,
      tibble::tibble(
        ROC = mean(tabela_folds$ROC, na.rm = TRUE),
        PRAUC = mean(tabela_folds$PRAUC, na.rm = TRUE),
        Sens = mean(tabela_folds$Sens, na.rm = TRUE),
        Spec = mean(tabela_folds$Spec, na.rm = TRUE),
        Precision = mean(tabela_folds$Precision, na.rm = TRUE),
        F1 = mean(tabela_folds$F1, na.rm = TRUE),
        GMean = mean(tabela_folds$GMean, na.rm = TRUE),
        Accuracy = mean(tabela_folds$Accuracy, na.rm = TRUE),
        ROCSD = stats::sd(tabela_folds$ROC, na.rm = TRUE),
        F1SD = stats::sd(tabela_folds$F1, na.rm = TRUE),
        GMeanSD = stats::sd(tabela_folds$GMean, na.rm = TRUE)
      )
    )
  }

  dplyr::bind_rows(resultados)
}

avaliar_xgb_oof <- function(dados, folds, grid_xgb, aplicar_smotenc = FALSE, formula_modelo = NULL) {
  resultados_grid <- vector("list", nrow(grid_xgb))
  predicoes_grid <- vector("list", nrow(grid_xgb))

  for (i in seq_len(nrow(grid_xgb))) {
    params_grid <- grid_xgb[i, , drop = FALSE]
    prob_oof <- rep(NA_real_, nrow(dados))

    for (j in seq_along(folds)) {
      idx_valid <- folds[[j]]
      treino_fold <- dados[-idx_valid, , drop = FALSE]
      valid_fold <- dados[idx_valid, , drop = FALSE]

      if (aplicar_smotenc) {
        dados_proc <- preparar_treino_teste_modelo(
          treino_df = treino_fold,
          teste_df = valid_fold,
          formula_modelo = formula_modelo,
          usar_smotenc = TRUE
        )

        treino_proc <- dados_proc$treino
        valid_proc <- dados_proc$teste
      } else {
        treino_proc <- treino_fold
        valid_proc <- valid_fold
      }

      x_treino <- model.matrix(Class ~ ., data = treino_proc)[, -1, drop = FALSE]
      x_valid <- model.matrix(Class ~ ., data = valid_proc)[, -1, drop = FALSE]
      x_valid <- alinhar_colunas_matriz(x_treino, x_valid)

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

      prob_oof[idx_valid] <- predict(modelo_xgb, newdata = x_valid)
    }

    metricas_oof <- calcular_metricas_prob(
      obs = dados$Class,
      prob_deve = prob_oof,
      threshold = 0.50
    )

    resultados_grid[[i]] <- dplyr::bind_cols(
      tibble::tibble(GridID = i),
      params_grid,
      metricas_oof %>% dplyr::select(ROC, PRAUC, Sens, Spec, Precision, F1, GMean, Accuracy)
    )

    predicoes_grid[[i]] <- tibble::tibble(
      rowIndex = seq_len(nrow(dados)),
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

treinar_prever_xgb <- function(treino_df, teste_df, config_modelo) {
  x_treino <- model.matrix(Class ~ ., data = treino_df)[, -1, drop = FALSE]
  x_teste <- model.matrix(Class ~ ., data = teste_df)[, -1, drop = FALSE]
  x_teste <- alinhar_colunas_matriz(x_treino, x_teste)

  dtrain <- xgboost::xgb.DMatrix(
    data = x_treino,
    label = ifelse(treino_df$Class == "Deve", 1, 0)
  )

  modelo_xgb <- xgboost::xgb.train(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = as.integer(config_modelo$max_depth[1]),
      eta = as.numeric(config_modelo$eta[1]),
      gamma = as.numeric(config_modelo$gamma[1]),
      colsample_bytree = as.numeric(config_modelo$colsample_bytree[1]),
      min_child_weight = as.numeric(config_modelo$min_child_weight[1]),
      subsample = as.numeric(config_modelo$subsample[1])
    ),
    data = dtrain,
    nrounds = as.integer(config_modelo$nrounds[1]),
    verbose = 0
  )

  list(
    model = modelo_xgb,
    prob = predict(modelo_xgb, newdata = x_teste),
    x_treino = x_treino,
    x_teste = x_teste
  )
}

treinar_prever_modelo_final <- function(config_modelo, treino_df, teste_df) {
  config_modelo <- tibble::as_tibble(config_modelo)
  modelo <- as.character(config_modelo$Modelo[1])
  vars_modelo <- parse_variaveis(config_modelo$Variaveis[1])
  formula_modelo <- montar_formula(vars_modelo)

  treino_sub <- garantir_ordem_classe(treino_df[, c(vars_modelo, "Class"), drop = FALSE])
  teste_sub <- garantir_ordem_classe(teste_df[, c(vars_modelo, "Class"), drop = FALSE])

  if (modelo == "XGBoost") {
    dados_modelo <- preparar_treino_teste_modelo(
      treino_df = treino_sub,
      teste_df = teste_sub,
      formula_modelo = formula_modelo,
      usar_smotenc = isTRUE(config_modelo$Usa_SMOTENC[1])
    )

    ajuste_xgb <- treinar_prever_xgb(
      treino_df = dados_modelo$treino,
      teste_df = dados_modelo$teste,
      config_modelo = config_modelo
    )

    return(list(
      model = ajuste_xgb$model,
      prob = ajuste_xgb$prob,
      treino = dados_modelo$treino,
      teste = dados_modelo$teste,
      formula = formula_modelo,
      x_treino = ajuste_xgb$x_treino,
      x_teste = ajuste_xgb$x_teste
    ))
  }

  spec <- obter_especificacao_modelo(modelo)
  tune_grid <- obter_grid_modelo_config(config_modelo)
  objeto_treino <- if (isTRUE(config_modelo$Usa_SMOTENC[1])) {
    preparar_receita_smotenc(formula_modelo = formula_modelo, dados = treino_sub)
  } else {
    formula_modelo
  }

  args <- c(
    list(
      objeto_treino,
      data = treino_sub,
      method = spec$method,
      metric = "ROC",
      trControl = controle_final_sem_cv()
    ),
    spec$train_args
  )

  if (!is.null(spec$preProcess)) {
    args$preProcess <- spec$preProcess
  }

  if (!is.null(tune_grid)) {
    args$tuneGrid <- tune_grid
  }

  set.seed(SEED_PROJETO)
  modelo_final <- do.call(caret::train, args)

  list(
    model = modelo_final,
    prob = predict(modelo_final, newdata = teste_sub, type = "prob")[, "Deve"],
    treino = treino_sub,
    teste = teste_sub,
    formula = formula_modelo
  )
}
