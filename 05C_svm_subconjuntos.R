# ==============================================================================
# 05C_svm_subconjuntos.R
# Responsabilidade: treinar SVM radial com busca em grid no subconjunto
# Top-14 e consolidar com o benchmark ja existente.
# ==============================================================================

source("00_setup.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar dados e ranking
# ------------------------------------------------------------------------------
treino <- readRDS("objetos/treino.rds")
ranking_variaveis <- readRDS("objetos/ranking_variaveis_enet.rds")

ordem_variaveis <- ranking_variaveis$Variavel_Original

subconjuntos <- list(
  Top14 = ordem_variaveis[1:14]
)

print(subconjuntos)

# ------------------------------------------------------------------------------
# BLOCO 2 - Funcoes auxiliares
# ------------------------------------------------------------------------------
montar_formula <- function(vars) {
  as.formula(paste("Class ~", paste(vars, collapse = " + ")))
}

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
  F1        <- ifelse(
    is.na(Precision) || is.na(Recall) || (Precision + Recall) == 0,
    NA,
    2 * Precision * Recall / (Precision + Recall)
  )
  GMean <- ifelse(
    is.na(base_metrics["Sens"]) || is.na(base_metrics["Spec"]),
    NA,
    sqrt(base_metrics["Sens"] * base_metrics["Spec"])
  )

  c(
    ROC = unname(base_metrics["ROC"]),
    Sens = unname(base_metrics["Sens"]),
    Spec = unname(base_metrics["Spec"]),
    Precision = unname(Precision),
    F1 = unname(F1),
    GMean = unname(GMean)
  )
}

controle_cv <- caret::trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = metricas_binarias,
  allowParallel = FALSE
)

extrair_melhor_resultado <- function(modelo, nome_modelo, nome_subconjunto, vars_sub) {
  res <- modelo$results

  if (!is.null(modelo$bestTune)) {
    for (col in names(modelo$bestTune)) {
      res <- res[res[[col]] == modelo$bestTune[[col]], , drop = FALSE]
    }
  }

  res %>%
    dplyr::mutate(
      Modelo = nome_modelo,
      Subconjunto = nome_subconjunto,
      Variaveis = paste(vars_sub, collapse = ", "),
      Base_Treino = "AmostraEstratificada5000"
    )
}

estimar_grid_svm <- function(dados_sub) {
  x_sub <- model.matrix(Class ~ ., data = dados_sub)[, -1, drop = FALSE]

  sigma_base <- tryCatch(
    as.numeric(kernlab::sigest(x_sub, frac = 0.5))[2],
    error = function(e) NA_real_
  )

  if (!is.finite(sigma_base) || sigma_base <= 0) {
    sigma_vals <- 0.05
  } else {
    sigma_vals <- signif(sigma_base, 6)
  }

  expand.grid(
    sigma = sigma_vals,
    C = c(0.5, 1)
  )
}

# ------------------------------------------------------------------------------
# BLOCO 3 - Treinar SVM radial
# ------------------------------------------------------------------------------
resultados_svm <- list()
contador <- 1

for (nome_sub in names(subconjuntos)) {
  vars_sub <- subconjuntos[[nome_sub]]
  dados_sub <- treino[, c(vars_sub, "Class")]
  proporcao_amostra <- min(1, 5000 / nrow(dados_sub))
  idx_amostra <- caret::createDataPartition(
    dados_sub$Class,
    p = proporcao_amostra,
    list = FALSE
  )
  dados_sub <- dados_sub[idx_amostra, , drop = FALSE]
  formula_sub <- montar_formula(vars_sub)
  grid_svm <- estimar_grid_svm(dados_sub)

  cat("\n====================================================\n")
  cat("Subconjunto:", nome_sub, "\n")
  cat("Variaveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("Tamanho da amostra estratificada:", nrow(dados_sub), "\n")
  cat("Grid SVM:\n")
  print(grid_svm)
  cat("====================================================\n")

  cat("Treinando modelo: SVM radial\n")

  set.seed(123)
  modelo_svm <- caret::train(
    formula_sub,
    data = dados_sub,
    method = "svmRadial",
    metric = "ROC",
    trControl = controle_cv,
    tuneGrid = grid_svm,
    preProcess = c("center", "scale")
  )

  resultados_svm[[contador]] <- extrair_melhor_resultado(
    modelo = modelo_svm,
    nome_modelo = "SVM_Radial",
    nome_subconjunto = nome_sub,
    vars_sub = vars_sub
  )
  contador <- contador + 1
}

# ------------------------------------------------------------------------------
# BLOCO 4 - Consolidar resultados
# ------------------------------------------------------------------------------
tabela_svm <- dplyr::bind_rows(resultados_svm) %>%
  dplyr::select(
    Subconjunto, Modelo,
    ROC, Sens, Spec, Precision, F1, GMean,
    dplyr::everything()
  ) %>%
  dplyr::arrange(desc(ROC), desc(F1), desc(GMean))

print(tabela_svm)

if (file.exists("objetos/tabela_benchmark_glm_rf_xgb_sem_balanceamento.rds")) {
  tabela_base <- readRDS("objetos/tabela_benchmark_glm_rf_xgb_sem_balanceamento.rds")

  tabela_benchmark_completa <- dplyr::bind_rows(
    tabela_base,
    tabela_svm
  ) %>%
    dplyr::arrange(desc(ROC), desc(F1), desc(GMean))
} else {
  warning("Benchmark base de GLM/RF/XGBoost nao encontrado. O consolidado ficara apenas com SVM.")
  tabela_benchmark_completa <- tabela_svm
}

print(tabela_benchmark_completa)

# ------------------------------------------------------------------------------
# BLOCO 5 - Graficos simples
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
# BLOCO 6 - Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(tabela_svm, "objetos/tabela_svm_subconjuntos_sem_balanceamento.rds")
readr::write_csv(tabela_svm, "resultados/tabela_svm_subconjuntos_sem_balanceamento.csv")

saveRDS(
  tabela_benchmark_completa,
  "objetos/tabela_benchmark_glm_rf_xgb_svm_sem_balanceamento.rds"
)
readr::write_csv(
  tabela_benchmark_completa,
  "resultados/tabela_benchmark_glm_rf_xgb_svm_sem_balanceamento.csv"
)

ggplot2::ggsave(
  filename = "figuras/roc_glm_rf_xgb_svm_subconjuntos_sem_balanceamento.png",
  plot = grafico_roc_modelos,
  width = 8,
  height = 5
)

ggplot2::ggsave(
  filename = "figuras/f1_glm_rf_xgb_svm_subconjuntos_sem_balanceamento.png",
  plot = grafico_f1_modelos,
  width = 8,
  height = 5
)

message("05C_svm_subconjuntos.R concluido com sucesso.")
