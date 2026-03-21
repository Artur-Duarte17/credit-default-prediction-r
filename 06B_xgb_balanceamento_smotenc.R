# ==============================================================================
# 06B_xgb_balanceamento_smotenc.R
# Responsabilidade: comparar XGBoost sem balanceamento vs com SMOTENC
# nos subconjuntos Top-13 e Top-14, e consolidar com o benchmark de RF.
# ==============================================================================

source("00_setup.R")

# ------------------------------------------------------------------------------
# BLOCO 1 — Carregar dados e ranking
# ------------------------------------------------------------------------------
treino <- readRDS("objetos/treino.rds")
ranking_variaveis <- readRDS("objetos/ranking_variaveis_enet.rds")

ordem_variaveis <- ranking_variaveis$Variavel_Original

subconjuntos <- list(
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

avaliar_xgb_cv <- function(dados, folds, grid_xgb, aplicar_smotenc = FALSE, formula_modelo = NULL) {
  resultados <- vector("list", nrow(grid_xgb))

  for (i in seq_len(nrow(grid_xgb))) {
    params_grid <- grid_xgb[i, ]

    metricas_folds <- lapply(folds, function(idx_valid) {
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

      calcular_metricas_prob(
        obs = valid_fold$Class,
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
# BLOCO 3 — Loop por subconjunto
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

resultados <- list()
contador <- 1

for (nome_sub in names(subconjuntos)) {
  vars_sub <- subconjuntos[[nome_sub]]
  dados_sub <- treino[, c(vars_sub, "Class")]
  formula_sub <- as.formula(paste("Class ~", paste(vars_sub, collapse = " + ")))

  set.seed(123)
  folds_xgb <- caret::createFolds(dados_sub$Class, k = 5, returnTrain = FALSE)

  cat("\n====================================================\n")
  cat("Subconjunto:", nome_sub, "\n")
  cat("Variáveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")

  cat("Treinando XGBoost sem balanceamento...\n")
  set.seed(123)
  tabela_base <- avaliar_xgb_cv(
    dados = dados_sub,
    folds = folds_xgb,
    grid_xgb = grid_xgb,
    aplicar_smotenc = FALSE,
    formula_modelo = formula_sub
  )

  melhor_base <- tabela_base %>%
    dplyr::arrange(desc(ROC), desc(F1), desc(GMean)) %>%
    dplyr::slice(1) %>%
    dplyr::mutate(
      Subconjunto = nome_sub,
      Cenario = "Sem_balanceamento",
      Modelo = "XGBoost",
      Variaveis = paste(vars_sub, collapse = ", ")
    )

  resultados[[contador]] <- melhor_base
  contador <- contador + 1

  cat("Treinando XGBoost com SMOTENC...\n")
  set.seed(123)
  tabela_smotenc <- avaliar_xgb_cv(
    dados = dados_sub,
    folds = folds_xgb,
    grid_xgb = grid_xgb,
    aplicar_smotenc = TRUE,
    formula_modelo = formula_sub
  )

  melhor_smotenc <- tabela_smotenc %>%
    dplyr::arrange(desc(ROC), desc(F1), desc(GMean)) %>%
    dplyr::slice(1) %>%
    dplyr::mutate(
      Subconjunto = nome_sub,
      Cenario = "Com_SMOTENC",
      Modelo = "XGBoost",
      Variaveis = paste(vars_sub, collapse = ", ")
    )

  resultados[[contador]] <- melhor_smotenc
  contador <- contador + 1
}

# ------------------------------------------------------------------------------
# BLOCO 4 — Consolidar resultados
# ------------------------------------------------------------------------------
tabela_xgb_balanceamento <- dplyr::bind_rows(resultados) %>%
  dplyr::select(
    Subconjunto, Modelo, Cenario,
    ROC, Sens, Spec, Precision, F1, GMean,
    dplyr::everything()
  ) %>%
  dplyr::arrange(Subconjunto, desc(ROC), desc(F1))

print(tabela_xgb_balanceamento)

if (file.exists("objetos/tabela_rf_balanceamento_smotenc.rds")) {
  tabela_rf_balanceamento <- readRDS("objetos/tabela_rf_balanceamento_smotenc.rds") %>%
    dplyr::mutate(Modelo = "RF")

  tabela_balanceamento_completa <- dplyr::bind_rows(
    tabela_rf_balanceamento,
    tabela_xgb_balanceamento
  ) %>%
    dplyr::arrange(Modelo, Subconjunto, desc(ROC), desc(F1))
} else {
  warning("Tabela de RF com SMOTENC não encontrada. O consolidado ficará apenas com XGBoost.")
  tabela_balanceamento_completa <- tabela_xgb_balanceamento
}

print(tabela_balanceamento_completa)

# ------------------------------------------------------------------------------
# BLOCO 5 — Gráficos simples
# ------------------------------------------------------------------------------
grafico_roc_balanceamento <- ggplot2::ggplot(
  tabela_balanceamento_completa,
  aes(x = Subconjunto, y = ROC, color = Cenario, group = Cenario)
) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 3) +
  ggplot2::geom_text(
    aes(label = round(ROC, 4)),
    vjust = -0.8,
    size = 3
  ) +
  ggplot2::facet_wrap(~ Modelo) +
  ggplot2::labs(
    title = "ROC: comparação de balanceamento por modelo",
    x = "Subconjunto",
    y = "ROC"
  ) +
  ggplot2::theme_minimal()

print(grafico_roc_balanceamento)

grafico_f1_balanceamento <- ggplot2::ggplot(
  tabela_balanceamento_completa,
  aes(x = Subconjunto, y = F1, color = Cenario, group = Cenario)
) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 3) +
  ggplot2::geom_text(
    aes(label = round(F1, 4)),
    vjust = -0.8,
    size = 3
  ) +
  ggplot2::facet_wrap(~ Modelo) +
  ggplot2::labs(
    title = "F1: comparação de balanceamento por modelo",
    x = "Subconjunto",
    y = "F1"
  ) +
  ggplot2::theme_minimal()

print(grafico_f1_balanceamento)

# ------------------------------------------------------------------------------
# BLOCO 6 — Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(
  tabela_xgb_balanceamento,
  "objetos/tabela_xgb_balanceamento_smotenc.rds"
)
readr::write_csv(
  tabela_xgb_balanceamento,
  "resultados/tabela_xgb_balanceamento_smotenc.csv"
)

saveRDS(
  tabela_balanceamento_completa,
  "objetos/tabela_rf_xgb_balanceamento_smotenc.rds"
)
readr::write_csv(
  tabela_balanceamento_completa,
  "resultados/tabela_rf_xgb_balanceamento_smotenc.csv"
)

ggplot2::ggsave(
  filename = "figuras/roc_rf_xgb_balanceamento_smotenc.png",
  plot = grafico_roc_balanceamento,
  width = 10,
  height = 5
)

ggplot2::ggsave(
  filename = "figuras/f1_rf_xgb_balanceamento_smotenc.png",
  plot = grafico_f1_balanceamento,
  width = 10,
  height = 5
)

message("06B_xgb_balanceamento_smotenc.R concluído com sucesso.")
