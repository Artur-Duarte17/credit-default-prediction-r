# ==============================================================================
# 05B_xgboost_subconjuntos.R
# Responsabilidade: treinar XGBoost nos subconjuntos Top-10, Top-13 e Top-14
# e consolidar a comparação com o benchmark já existente de GLM/RF.
# ==============================================================================

source("00_setup.R")

# ------------------------------------------------------------------------------
# BLOCO 1 — Carregar dados e ranking
# ------------------------------------------------------------------------------
treino <- readRDS("objetos/treino.rds")
ranking_variaveis <- readRDS("objetos/ranking_variaveis_enet.rds")

ordem_variaveis <- ranking_variaveis$Variavel_Original

subconjuntos <- list(
  Top10 = ordem_variaveis[1:10],
  Top13 = ordem_variaveis[1:13],
  Top14 = ordem_variaveis[1:14]
)

print(subconjuntos)

# ------------------------------------------------------------------------------
# BLOCO 2 — Funções auxiliares
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
  F1 <- ifelse(is.na(Precision) || is.na(Sens) || (Precision + Sens) == 0,
               NA,
               2 * Precision * Sens / (Precision + Sens))
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
# BLOCO 3 — Treinar XGBoost
# ------------------------------------------------------------------------------
resultados_xgb <- list()
contador <- 1

for (nome_sub in names(subconjuntos)) {

  vars_sub <- subconjuntos[[nome_sub]]
  dados_sub <- treino[, c(vars_sub, "Class")]
  x_sub <- model.matrix(Class ~ ., data = dados_sub)[, -1, drop = FALSE]
  y_sub <- dados_sub$Class
  folds_cv <- caret::createFolds(y_sub, k = 5, returnTrain = FALSE)

  cat("\n====================================================\n")
  cat("Subconjunto:", nome_sub, "\n")
  cat("Variáveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")

  # Grade enxuta para caber no tempo e manter o padrão do projeto.
  grid_xgb <- expand.grid(
    nrounds = c(100, 150),
    max_depth = c(3, 5),
    eta = 0.1,
    gamma = 0,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    subsample = 0.8
  )

  cat("Treinando modelo: XGBoost\n")

  set.seed(123)
  tabela_xgb_sub <- avaliar_xgb_cv(
    x = x_sub,
    y = y_sub,
    grid_xgb = grid_xgb,
    folds = folds_cv
  )

  res_xgb <- tabela_xgb_sub %>%
    dplyr::arrange(desc(ROC), desc(F1), desc(GMean)) %>%
    dplyr::slice(1) %>%
    dplyr::mutate(
      Modelo = "XGBoost",
      Subconjunto = nome_sub,
      Variaveis = paste(vars_sub, collapse = ", ")
    )

  resultados_xgb[[contador]] <- res_xgb
  contador <- contador + 1
}

# ------------------------------------------------------------------------------
# BLOCO 4 — Consolidar resultados
# ------------------------------------------------------------------------------
tabela_xgb <- dplyr::bind_rows(resultados_xgb) %>%
  dplyr::select(
    Subconjunto, Modelo,
    ROC, Sens, Spec, Precision, F1, GMean,
    dplyr::everything()
  ) %>%
  dplyr::arrange(desc(ROC), desc(F1), desc(GMean))

print(tabela_xgb)

if (file.exists("objetos/tabela_benchmark_glm_rf_sem_balanceamento.rds")) {
  tabela_glm_rf <- readRDS("objetos/tabela_benchmark_glm_rf_sem_balanceamento.rds")

  tabela_benchmark_completa <- dplyr::bind_rows(
    tabela_glm_rf,
    tabela_xgb
  ) %>%
    dplyr::arrange(desc(ROC), desc(F1), desc(GMean))
} else {
  warning("Benchmark de GLM/RF não encontrado. O consolidado ficará apenas com XGBoost.")
  tabela_benchmark_completa <- tabela_xgb
}

print(tabela_benchmark_completa)

# ------------------------------------------------------------------------------
# BLOCO 5 — Gráficos simples
# ------------------------------------------------------------------------------
grafico_roc_modelos <- ggplot2::ggplot(
  tabela_benchmark_completa,
  aes(x = Subconjunto, y = ROC, color = Modelo, group = Modelo)
) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 3) +
  ggplot2::geom_text(
    aes(label = round(ROC, 4)),
    vjust = -0.8,
    size = 3
  ) +
  ggplot2::labs(
    title = "ROC por modelo e subconjunto",
    x = "Subconjunto",
    y = "ROC"
  ) +
  ggplot2::theme_minimal()

print(grafico_roc_modelos)

grafico_f1_modelos <- ggplot2::ggplot(
  tabela_benchmark_completa,
  aes(x = Subconjunto, y = F1, color = Modelo, group = Modelo)
) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 3) +
  ggplot2::geom_text(
    aes(label = round(F1, 4)),
    vjust = -0.8,
    size = 3
  ) +
  ggplot2::labs(
    title = "F1 por modelo e subconjunto",
    x = "Subconjunto",
    y = "F1"
  ) +
  ggplot2::theme_minimal()

print(grafico_f1_modelos)

# ------------------------------------------------------------------------------
# BLOCO 6 — Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(tabela_xgb, "objetos/tabela_xgboost_subconjuntos_sem_balanceamento.rds")
readr::write_csv(tabela_xgb, "resultados/tabela_xgboost_subconjuntos_sem_balanceamento.csv")

saveRDS(
  tabela_benchmark_completa,
  "objetos/tabela_benchmark_glm_rf_xgb_sem_balanceamento.rds"
)
readr::write_csv(
  tabela_benchmark_completa,
  "resultados/tabela_benchmark_glm_rf_xgb_sem_balanceamento.csv"
)

ggplot2::ggsave(
  filename = "figuras/roc_glm_rf_xgb_subconjuntos_sem_balanceamento.png",
  plot = grafico_roc_modelos,
  width = 8,
  height = 5
)

ggplot2::ggsave(
  filename = "figuras/f1_glm_rf_xgb_subconjuntos_sem_balanceamento.png",
  plot = grafico_f1_modelos,
  width = 8,
  height = 5
)

message("05B_xgboost_subconjuntos.R concluído com sucesso.")
