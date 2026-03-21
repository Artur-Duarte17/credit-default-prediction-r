# ==============================================================================
# 05A_glm_rf_subconjuntos.R
# Responsabilidade: comparar GLM e RF nos subconjuntos Top-10, Top-13 e Top-14
# sem balanceamento, de forma mais rápida.
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

controle_cv <- caret::trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = metricas_binarias,
  allowParallel = FALSE
)

extrair_melhor_resultado <- function(modelo, nome_modelo, nome_subconjunto) {
  res <- modelo$results
  
  if (!is.null(modelo$bestTune)) {
    for (col in names(modelo$bestTune)) {
      res <- res[res[[col]] == modelo$bestTune[[col]], , drop = FALSE]
    }
  }
  
  res %>%
    dplyr::mutate(
      Modelo = nome_modelo,
      Subconjunto = nome_subconjunto
    )
}

# ------------------------------------------------------------------------------
# BLOCO 3 — Treinar GLM e RF
# ------------------------------------------------------------------------------
resultados_modelos <- list()
contador <- 1

for (nome_sub in names(subconjuntos)) {
  
  vars_sub <- subconjuntos[[nome_sub]]
  dados_sub <- treino[, c(vars_sub, "Class")]
  formula_sub <- montar_formula(vars_sub)
  p <- ncol(dados_sub) - 1
  
  cat("\n====================================================\n")
  cat("Subconjunto:", nome_sub, "\n")
  cat("Variáveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")
  
  # ---------------- GLM ----------------
  cat("Treinando modelo: GLM\n")
  
  set.seed(123)
  modelo_glm <- caret::train(
    formula_sub,
    data = dados_sub,
    method = "glm",
    family = binomial(),
    metric = "ROC",
    trControl = controle_cv
  )
  
  res_glm <- extrair_melhor_resultado(
    modelo = modelo_glm,
    nome_modelo = "GLM",
    nome_subconjunto = nome_sub
  ) %>%
    dplyr::mutate(
      Variaveis = paste(vars_sub, collapse = ", ")
    )
  
  resultados_modelos[[contador]] <- res_glm
  contador <- contador + 1
  
  # ---------------- RF ----------------
  cat("Treinando modelo: RF\n")
  
  grid_rf <- data.frame(
    mtry = sort(unique(c(
      max(1, floor(sqrt(p)) - 1),
      max(1, floor(sqrt(p))),
      min(p, floor(sqrt(p)) + 1)
    )))
  )
  
  set.seed(123)
  modelo_rf <- caret::train(
    formula_sub,
    data = dados_sub,
    method = "rf",
    metric = "ROC",
    trControl = controle_cv,
    tuneGrid = grid_rf,
    importance = TRUE,
    ntree = 200
  )
  
  res_rf <- extrair_melhor_resultado(
    modelo = modelo_rf,
    nome_modelo = "RF",
    nome_subconjunto = nome_sub
  ) %>%
    dplyr::mutate(
      Variaveis = paste(vars_sub, collapse = ", ")
    )
  
  resultados_modelos[[contador]] <- res_rf
  contador <- contador + 1
}

# ------------------------------------------------------------------------------
# BLOCO 4 — Consolidar resultados
# ------------------------------------------------------------------------------
tabela_benchmark <- dplyr::bind_rows(resultados_modelos) %>%
  dplyr::select(
    Subconjunto, Modelo,
    ROC, Sens, Spec, Precision, F1, GMean,
    dplyr::everything()
  ) %>%
  dplyr::arrange(desc(ROC), desc(F1), desc(GMean))

print(tabela_benchmark)

# ------------------------------------------------------------------------------
# BLOCO 5 — Gráficos simples
# ------------------------------------------------------------------------------
grafico_roc_modelos <- ggplot2::ggplot(
  tabela_benchmark,
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
  tabela_benchmark,
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
saveRDS(tabela_benchmark, "objetos/tabela_benchmark_glm_rf_sem_balanceamento.rds")
readr::write_csv(tabela_benchmark, "resultados/tabela_benchmark_glm_rf_sem_balanceamento.csv")

ggplot2::ggsave(
  filename = "figuras/roc_glm_rf_subconjuntos_sem_balanceamento.png",
  plot = grafico_roc_modelos,
  width = 8,
  height = 5
)

ggplot2::ggsave(
  filename = "figuras/f1_glm_rf_subconjuntos_sem_balanceamento.png",
  plot = grafico_f1_modelos,
  width = 8,
  height = 5
)

message("05A_glm_rf_subconjuntos.R concluído com sucesso.")