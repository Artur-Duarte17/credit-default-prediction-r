# ==============================================================================
# 08_avaliacao_teste.R
# Responsabilidade: avaliar no conjunto de teste os cenários finalistas
# de RF e XGBoost com threshold calibrado via Youden.
# ==============================================================================

source("00_setup.R")

# ------------------------------------------------------------------------------
# BLOCO 1 — Carregar dados e definir variáveis finais
# ------------------------------------------------------------------------------
treino <- readRDS("objetos/treino.rds")
teste  <- readRDS("objetos/teste.rds")
config_modelos <- readRDS("objetos/config_modelos_finais.rds")

parse_variaveis <- function(texto_variaveis) {
  trimws(strsplit(texto_variaveis, ",")[[1]])
}

config_rf_base <- config_modelos %>%
  dplyr::filter(Modelo == "RF", Usa_SMOTENC == FALSE) %>%
  dplyr::slice(1)

config_rf_smotenc <- config_modelos %>%
  dplyr::filter(Modelo == "RF", Usa_SMOTENC == TRUE) %>%
  dplyr::slice(1)

config_xgb_base <- config_modelos %>%
  dplyr::filter(Modelo == "XGBoost", Usa_SMOTENC == FALSE) %>%
  dplyr::slice(1)

config_xgb_smotenc <- config_modelos %>%
  dplyr::filter(Modelo == "XGBoost", Usa_SMOTENC == TRUE) %>%
  dplyr::slice(1)

vars_rf <- parse_variaveis(config_rf_base$Variaveis[1])
vars_xgb <- parse_variaveis(config_xgb_base$Variaveis[1])

treino_rf_sub <- treino[, c(vars_rf, "Class")]
teste_rf_sub  <- teste[, c(vars_rf, "Class")]

treino_xgb_sub <- treino[, c(vars_xgb, "Class")]
teste_xgb_sub  <- teste[, c(vars_xgb, "Class")]

print(vars_rf)
print(vars_xgb)
print(dim(treino_rf_sub))
print(dim(teste_rf_sub))
print(dim(treino_xgb_sub))
print(dim(teste_xgb_sub))
print(config_modelos)

# ------------------------------------------------------------------------------
# BLOCO 2 — Funções auxiliares
# ------------------------------------------------------------------------------
calcular_metricas_teste <- function(obs, prob_deve, threshold, cenario) {

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
  Accuracy <- (TP + TN) / (TP + TN + FP + FN)

  obs_roc <- factor(obs, levels = c("Pago", "Deve"))
  roc_obj <- pROC::roc(
    response = obs_roc,
    predictor = prob_deve,
    levels = c("Pago", "Deve"),
    direction = "<",
    quiet = TRUE
  )
  ROC <- as.numeric(pROC::auc(roc_obj))

  pr_obj <- PRROC::pr.curve(
    scores.class0 = prob_deve[obs == "Deve"],
    scores.class1 = prob_deve[obs == "Pago"],
    curve = FALSE
  )
  PRAUC <- pr_obj$auc.integral

  tibble::tibble(
    Cenario = cenario,
    Threshold = threshold,
    ROC = ROC,
    PRAUC = PRAUC,
    Sens = Sens,
    Spec = Spec,
    Precision = Precision,
    F1 = F1,
    GMean = GMean,
    Accuracy = Accuracy,
    TP = TP,
    FP = FP,
    FN = FN,
    TN = TN
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

treinar_prever_xgb <- function(treino_df, teste_df, config_xgb) {
  x_treino <- model.matrix(Class ~ ., data = treino_df)[, -1, drop = FALSE]
  x_teste  <- model.matrix(Class ~ ., data = teste_df)[, -1, drop = FALSE]
  x_teste  <- alinhar_colunas_matriz(x_treino, x_teste)

  dtrain <- xgboost::xgb.DMatrix(
    data = x_treino,
    label = ifelse(treino_df$Class == "Deve", 1, 0)
  )

  modelo_xgb <- xgboost::xgb.train(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = as.integer(config_xgb$max_depth),
      eta = config_xgb$eta,
      gamma = config_xgb$gamma,
      colsample_bytree = config_xgb$colsample_bytree,
      min_child_weight = config_xgb$min_child_weight,
      subsample = config_xgb$subsample
    ),
    data = dtrain,
    nrounds = as.integer(config_xgb$nrounds),
    verbose = 0
  )

  predict(modelo_xgb, newdata = x_teste)
}
# ------------------------------------------------------------------------------
# BLOCO 3 — Preparar configurações dos modelos
# ------------------------------------------------------------------------------
formula_rf <- as.formula(
  paste("Class ~", paste(vars_rf, collapse = " + "))
)

formula_xgb <- as.formula(
  paste("Class ~", paste(vars_xgb, collapse = " + "))
)

receita_rf_smotenc <- recipes::recipe(formula_rf, data = treino_rf_sub) %>%
  themis::step_smotenc(Class, over_ratio = 1, neighbors = 5, skip = TRUE)

prep_rf_smotenc <- recipes::prep(receita_rf_smotenc, training = treino_rf_sub, retain = TRUE)

treino_rf_smotenc <- recipes::juice(prep_rf_smotenc)
teste_rf_proc     <- recipes::bake(prep_rf_smotenc, new_data = teste_rf_sub)

receita_xgb_smotenc <- recipes::recipe(formula_xgb, data = treino_xgb_sub) %>%
  themis::step_smotenc(Class, over_ratio = 1, neighbors = 5, skip = TRUE)

prep_xgb_smotenc <- recipes::prep(receita_xgb_smotenc, training = treino_xgb_sub, retain = TRUE)

treino_xgb_smotenc <- recipes::juice(prep_xgb_smotenc)
teste_xgb_proc     <- recipes::bake(prep_xgb_smotenc, new_data = teste_xgb_sub)

print(dim(treino_rf_smotenc))
print(dim(teste_rf_proc))
print(dim(treino_xgb_smotenc))
print(dim(teste_xgb_proc))

# ------------------------------------------------------------------------------
# BLOCO 4 — RF final sem balanceamento
# ------------------------------------------------------------------------------
cat("\nTreinando RF final sem balanceamento...\n")

modelo_rf_final_base <- randomForest::randomForest(
  formula_rf,
  data = treino_rf_sub,
  mtry = config_rf_base$mtry,
  ntree = 200,
  importance = TRUE
)

prob_rf_base <- predict(
  modelo_rf_final_base,
  newdata = teste_rf_sub,
  type = "prob"
)[, "Deve"]

metricas_rf_base <- calcular_metricas_teste(
  obs = teste_rf_sub$Class,
  prob_deve = prob_rf_base,
  threshold = config_rf_base$Threshold_Youden,
  cenario = paste0(config_rf_base$Cenario, "_Youden")
)

print(metricas_rf_base)

# ------------------------------------------------------------------------------
# BLOCO 5 — RF final com SMOTENC
# ------------------------------------------------------------------------------
cat("\nTreinando RF final com SMOTENC...\n")

modelo_rf_final_smotenc <- randomForest::randomForest(
  formula_rf,
  data = treino_rf_smotenc,
  mtry = config_rf_smotenc$mtry,
  ntree = 200,
  importance = TRUE
)

prob_rf_smotenc <- predict(
  modelo_rf_final_smotenc,
  newdata = teste_rf_proc,
  type = "prob"
)[, "Deve"]

metricas_rf_smotenc <- calcular_metricas_teste(
  obs = teste_rf_proc$Class,
  prob_deve = prob_rf_smotenc,
  threshold = config_rf_smotenc$Threshold_Youden,
  cenario = paste0(config_rf_smotenc$Cenario, "_Youden")
)

print(metricas_rf_smotenc)

# ------------------------------------------------------------------------------
# BLOCO 6 — XGBoost final sem balanceamento
# ------------------------------------------------------------------------------
cat("\nTreinando XGBoost final sem balanceamento...\n")

prob_xgb_base <- treinar_prever_xgb(
  treino_df = treino_xgb_sub,
  teste_df = teste_xgb_sub,
  config_xgb = config_xgb_base
)

metricas_xgb_base <- calcular_metricas_teste(
  obs = teste_xgb_sub$Class,
  prob_deve = prob_xgb_base,
  threshold = config_xgb_base$Threshold_Youden,
  cenario = paste0(config_xgb_base$Cenario, "_Youden")
)

print(metricas_xgb_base)

# ------------------------------------------------------------------------------
# BLOCO 7 — XGBoost final com SMOTENC
# ------------------------------------------------------------------------------
cat("\nTreinando XGBoost final com SMOTENC...\n")

prob_xgb_smotenc <- treinar_prever_xgb(
  treino_df = treino_xgb_smotenc,
  teste_df = teste_xgb_proc,
  config_xgb = config_xgb_smotenc
)

metricas_xgb_smotenc <- calcular_metricas_teste(
  obs = teste_xgb_proc$Class,
  prob_deve = prob_xgb_smotenc,
  threshold = config_xgb_smotenc$Threshold_Youden,
  cenario = paste0(config_xgb_smotenc$Cenario, "_Youden")
)

print(metricas_xgb_smotenc)

# ------------------------------------------------------------------------------
# BLOCO 8 — Consolidar resultados finais
# ------------------------------------------------------------------------------
tabela_teste <- dplyr::bind_rows(
  metricas_rf_base,
  metricas_rf_smotenc,
  metricas_xgb_base,
  metricas_xgb_smotenc
) %>%
  dplyr::arrange(desc(ROC), desc(F1), desc(GMean))

print(tabela_teste)

# ------------------------------------------------------------------------------
# BLOCO 9 — Gráficos simples
# ------------------------------------------------------------------------------
tabela_longa <- tabela_teste %>%
  dplyr::select(Cenario, ROC, PRAUC, Sens, Spec, Precision, F1, GMean) %>%
  tidyr::pivot_longer(
    cols = c(ROC, PRAUC, Sens, Spec, Precision, F1, GMean),
    names_to = "Metrica",
    values_to = "Valor"
  )

grafico_metricas_teste <- ggplot2::ggplot(
  tabela_longa,
  aes(x = Metrica, y = Valor, fill = Cenario)
) +
  ggplot2::geom_col(position = "dodge") +
  ggplot2::geom_text(
    aes(label = round(Valor, 3)),
    position = ggplot2::position_dodge(width = 0.9),
    vjust = -0.4,
    size = 3
  ) +
  ggplot2::labs(
    title = "Comparação final no conjunto de teste",
    x = "Métrica",
    y = "Valor"
  ) +
  ggplot2::theme_minimal()

print(grafico_metricas_teste)

roc_rf_base <- pROC::roc(
  response = factor(teste_rf_sub$Class, levels = c("Pago", "Deve")),
  predictor = prob_rf_base,
  levels = c("Pago", "Deve"),
  direction = "<",
  quiet = TRUE
)

roc_rf_smotenc <- pROC::roc(
  response = factor(teste_rf_proc$Class, levels = c("Pago", "Deve")),
  predictor = prob_rf_smotenc,
  levels = c("Pago", "Deve"),
  direction = "<",
  quiet = TRUE
)

roc_xgb_base <- pROC::roc(
  response = factor(teste_xgb_sub$Class, levels = c("Pago", "Deve")),
  predictor = prob_xgb_base,
  levels = c("Pago", "Deve"),
  direction = "<",
  quiet = TRUE
)

roc_xgb_smotenc <- pROC::roc(
  response = factor(teste_xgb_proc$Class, levels = c("Pago", "Deve")),
  predictor = prob_xgb_smotenc,
  levels = c("Pago", "Deve"),
  direction = "<",
  quiet = TRUE
)

df_roc <- dplyr::bind_rows(
  tibble::tibble(
    FPR = 1 - roc_rf_base$specificities,
    TPR = roc_rf_base$sensitivities,
    Cenario = paste0(config_rf_base$Cenario, "_Youden")
  ),
  tibble::tibble(
    FPR = 1 - roc_rf_smotenc$specificities,
    TPR = roc_rf_smotenc$sensitivities,
    Cenario = paste0(config_rf_smotenc$Cenario, "_Youden")
  ),
  tibble::tibble(
    FPR = 1 - roc_xgb_base$specificities,
    TPR = roc_xgb_base$sensitivities,
    Cenario = paste0(config_xgb_base$Cenario, "_Youden")
  ),
  tibble::tibble(
    FPR = 1 - roc_xgb_smotenc$specificities,
    TPR = roc_xgb_smotenc$sensitivities,
    Cenario = paste0(config_xgb_smotenc$Cenario, "_Youden")
  )
)

grafico_roc_teste <- ggplot2::ggplot(df_roc, aes(x = FPR, y = TPR, color = Cenario)) +
  ggplot2::geom_line(linewidth = 1) +
  ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  ggplot2::labs(
    title = "Curvas ROC no conjunto de teste",
    x = "1 - Especificidade",
    y = "Sensibilidade"
  ) +
  ggplot2::theme_minimal()

print(grafico_roc_teste)

# ------------------------------------------------------------------------------
# BLOCO 10 — Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(tabela_teste, "objetos/tabela_teste_final.rds")
readr::write_csv(tabela_teste, "resultados/tabela_teste_final.csv")

ggplot2::ggsave(
  filename = "figuras/comparacao_final_teste_metricas.png",
  plot = grafico_metricas_teste,
  width = 12,
  height = 6
)

ggplot2::ggsave(
  filename = "figuras/curvas_roc_teste.png",
  plot = grafico_roc_teste,
  width = 9,
  height = 5
)

message("08_avaliacao_teste.R concluído com sucesso.")
tabela_teste
