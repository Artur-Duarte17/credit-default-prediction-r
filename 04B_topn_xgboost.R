# ==============================================================================
# 04B_topn_xgboost.R
# Responsabilidade: testar subconjuntos Top-1 ate Top-N usando XGBoost
# como modelo caixa-preta.
# ==============================================================================

source("00_setup.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar treino e ranking
# ------------------------------------------------------------------------------
treino <- readRDS("objetos/treino.rds")
ranking_variaveis <- readRDS("objetos/ranking_variaveis_enet.rds")

ordem_variaveis <- ranking_variaveis$Variavel_Original
n_total_variaveis <- length(ordem_variaveis)

print(ranking_variaveis)
cat("Total de variaveis ranqueadas:", n_total_variaveis, "\n")

# ------------------------------------------------------------------------------
# BLOCO 2 - Definir folds e grade do XGBoost
# ------------------------------------------------------------------------------
set.seed(123)
folds_cv <- caret::createFolds(treino$Class, k = 5, returnTrain = FALSE)

grid_xgb <- expand.grid(
  nrounds = c(100, 150),
  max_depth = c(3, 5),
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

print(grid_xgb)

# ------------------------------------------------------------------------------
# BLOCO 3 - Funcoes auxiliares
# ------------------------------------------------------------------------------
calcular_metricas_prob <- function(obs, prob_deve) {
  obs_factor <- factor(obs, levels = c("Deve", "Pago"))
  pred_class <- ifelse(prob_deve >= 0.5, "Deve", "Pago")
  pred_factor <- factor(pred_class, levels = c("Deve", "Pago"))

  cm <- table(pred_factor, obs_factor)

  TP <- cm["Deve", "Deve"]
  FP <- cm["Deve", "Pago"]
  FN <- cm["Pago", "Deve"]
  TN <- cm["Pago", "Pago"]

  Sens <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
  Spec <- ifelse((TN + FP) == 0, NA, TN / (TN + FP))
  Precision <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
  F1 <- ifelse(
    is.na(Precision) || is.na(Sens) || (Precision + Sens) == 0,
    NA,
    2 * Precision * Sens / (Precision + Sens)
  )
  GMean <- ifelse(is.na(Sens) || is.na(Spec), NA, sqrt(Sens * Spec))

  roc_obj <- pROC::roc(
    response = factor(obs, levels = c("Pago", "Deve")),
    predictor = prob_deve,
    levels = c("Pago", "Deve"),
    direction = "<",
    quiet = TRUE
  )

  tibble::tibble(
    ROC = as.numeric(pROC::auc(roc_obj)),
    Sens = Sens,
    Spec = Spec,
    Precision = Precision,
    F1 = F1,
    GMean = GMean
  )
}

avaliar_xgb_cv <- function(x, y, grid_xgb, folds) {
  resultados <- vector("list", nrow(grid_xgb))

  for (i in seq_len(nrow(grid_xgb))) {
    params_grid <- grid_xgb[i, ]

    metricas_folds <- lapply(folds, function(idx_valid) {
      idx_train <- setdiff(seq_len(nrow(x)), idx_valid)

      dtrain <- xgboost::xgb.DMatrix(
        data = x[idx_train, , drop = FALSE],
        label = ifelse(y[idx_train] == "Deve", 1, 0)
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

      prob_valid <- predict(
        modelo_xgb,
        newdata = x[idx_valid, , drop = FALSE]
      )

      calcular_metricas_prob(
        obs = y[idx_valid],
        prob_deve = prob_valid
      )
    })

    tabela_folds <- dplyr::bind_rows(metricas_folds)

    resultados[[i]] <- dplyr::bind_cols(
      params_grid,
      tibble::tibble(
        ROC = mean(tabela_folds$ROC, na.rm = TRUE),
        Sens = mean(tabela_folds$Sens, na.rm = TRUE),
        Spec = mean(tabela_folds$Spec, na.rm = TRUE),
        Precision = mean(tabela_folds$Precision, na.rm = TRUE),
        F1 = mean(tabela_folds$F1, na.rm = TRUE),
        GMean = mean(tabela_folds$GMean, na.rm = TRUE),
        ROCSD = stats::sd(tabela_folds$ROC, na.rm = TRUE),
        SensSD = stats::sd(tabela_folds$Sens, na.rm = TRUE),
        SpecSD = stats::sd(tabela_folds$Spec, na.rm = TRUE),
        PrecisionSD = stats::sd(tabela_folds$Precision, na.rm = TRUE),
        F1SD = stats::sd(tabela_folds$F1, na.rm = TRUE),
        GMeanSD = stats::sd(tabela_folds$GMean, na.rm = TRUE)
      )
    )
  }

  dplyr::bind_rows(resultados)
}

# ------------------------------------------------------------------------------
# BLOCO 4 - Loop Top-1 ate Top-N
# ------------------------------------------------------------------------------
resultados_topn <- vector("list", n_total_variaveis)

for (k in seq_along(ordem_variaveis)) {
  vars_k <- ordem_variaveis[1:k]
  dados_k <- treino[, c(vars_k, "Class")]
  x_k <- model.matrix(Class ~ ., data = dados_k)[, -1, drop = FALSE]
  y_k <- dados_k$Class

  cat("\n=============================\n")
  cat("Rodando Top-", k, "\n", sep = "")
  cat("Variaveis:", paste(vars_k, collapse = ", "), "\n")
  cat("=============================\n")

  set.seed(123)
  tabela_xgb_k <- avaliar_xgb_cv(
    x = x_k,
    y = y_k,
    grid_xgb = grid_xgb,
    folds = folds_cv
  )

  res_k <- tabela_xgb_k %>%
    dplyr::arrange(desc(ROC), desc(F1), desc(GMean)) %>%
    dplyr::slice(1) %>%
    dplyr::mutate(
      TopN = k,
      Modelo = "XGBoost",
      Variaveis = paste(vars_k, collapse = ", ")
    ) %>%
    dplyr::select(
      TopN, Modelo,
      ROC, Sens, Spec, Precision, F1, GMean,
      dplyr::everything()
    )

  print(res_k)
  resultados_topn[[k]] <- res_k
}

# ------------------------------------------------------------------------------
# BLOCO 5 - Consolidar resultados
# ------------------------------------------------------------------------------
tabela_topn_xgb <- dplyr::bind_rows(resultados_topn) %>%
  dplyr::arrange(desc(ROC), desc(F1), desc(GMean))

print(tabela_topn_xgb)

top10_topn_xgb <- tabela_topn_xgb %>%
  dplyr::slice(1:10)

print(top10_topn_xgb)

curva_topn_xgb <- tabela_topn_xgb %>%
  dplyr::arrange(TopN)

melhor_topn_xgb <- curva_topn_xgb %>%
  dplyr::slice_max(order_by = ROC, n = 1, with_ties = FALSE)

print(melhor_topn_xgb)

# ------------------------------------------------------------------------------
# BLOCO 6 - Graficos
# ------------------------------------------------------------------------------
top10_plot <- top10_topn_xgb %>%
  dplyr::mutate(
    Subconjunto = paste0("Top-", TopN),
    Subconjunto = reorder(Subconjunto, ROC)
  )

grafico_melhores_topn <- ggplot2::ggplot(top10_plot, aes(x = ROC, y = Subconjunto)) +
  ggplot2::geom_segment(
    aes(
      x = min(top10_plot$ROC) - 0.001,
      xend = ROC,
      y = Subconjunto,
      yend = Subconjunto
    ),
    color = "#9ecae1",
    linewidth = 1
  ) +
  ggplot2::geom_point(color = "#08519c", size = 4) +
  ggplot2::geom_text(
    aes(label = round(ROC, 5)),
    hjust = -0.3,
    size = 3.5
  ) +
  ggplot2::labs(
    title = "Top 10 Melhores Subconjuntos por ROC - XGBoost",
    x = "Area Sob a Curva (ROC)",
    y = NULL
  ) +
  ggplot2::xlim(min(top10_plot$ROC) - 0.001, max(top10_plot$ROC) + 0.001) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", size = 14),
    panel.grid.major.y = ggplot2::element_blank()
  )

print(grafico_melhores_topn)

grafico_roc_topn <- ggplot2::ggplot(curva_topn_xgb, aes(x = TopN, y = ROC)) +
  ggplot2::geom_line(color = "#3182bd", linewidth = 1) +
  ggplot2::geom_point(color = "#3182bd", size = 2) +
  ggplot2::geom_point(
    data = melhor_topn_xgb,
    aes(x = TopN, y = ROC),
    color = "#08306b",
    size = 4
  ) +
  ggplot2::labs(
    title = "Desempenho (ROC) por Quantidade de Variaveis - XGBoost",
    subtitle = "Avaliacao do impacto ao adicionar atributos sucessivos",
    x = "Quantidade de Variaveis (Top-N)",
    y = "Area Sob a Curva (ROC)"
  ) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", size = 14)
  )

print(grafico_roc_topn)

# ------------------------------------------------------------------------------
# BLOCO 7 - Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(curva_topn_xgb, "objetos/curva_topn_xgboost.rds")
saveRDS(melhor_topn_xgb, "objetos/melhor_topn_xgboost.rds")

readr::write_csv(curva_topn_xgb, "resultados/curva_topn_xgboost.csv")
readr::write_csv(top10_topn_xgb, "resultados/top10_topn_xgboost.csv")

ggplot2::ggsave(
  filename = "figuras/curva_roc_topn_xgboost.png",
  plot = grafico_roc_topn,
  width = 8,
  height = 5
)

ggplot2::ggsave(
  filename = "figuras/top10_melhores_subconjuntos_xgboost.png",
  plot = grafico_melhores_topn,
  width = 8,
  height = 5
)

message("04B_topn_xgboost.R concluido com sucesso.")
