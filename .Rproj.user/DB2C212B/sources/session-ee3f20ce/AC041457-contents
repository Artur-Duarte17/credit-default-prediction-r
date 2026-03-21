# ==============================================================================
# 06_rf_balanceamento_smotenc.R
# Responsabilidade: comparar RF sem balanceamento vs RF com SMOTENC
# nos subconjuntos Top-13 e Top-14.
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

extrair_melhor_resultado <- function(modelo, cenario, subconjunto, vars_sub) {
  res <- modelo$results
  
  if (!is.null(modelo$bestTune)) {
    for (col in names(modelo$bestTune)) {
      res <- res[res[[col]] == modelo$bestTune[[col]], , drop = FALSE]
    }
  }
  
  res %>%
    dplyr::mutate(
      Cenario = cenario,
      Subconjunto = subconjunto,
      Variaveis = paste(vars_sub, collapse = ", ")
    )
}

# ------------------------------------------------------------------------------
# BLOCO 3 — Loop por subconjunto
# ------------------------------------------------------------------------------
resultados <- list()
contador <- 1

for (nome_sub in names(subconjuntos)) {
  
  vars_sub <- subconjuntos[[nome_sub]]
  dados_sub <- treino[, c(vars_sub, "Class")]
  formula_sub <- montar_formula(vars_sub)
  p <- ncol(dados_sub) - 1
  
  grid_rf <- data.frame(
    mtry = sort(unique(c(
      max(1, floor(sqrt(p)) - 1),
      max(1, floor(sqrt(p))),
      min(p, floor(sqrt(p)) + 1)
    )))
  )
  
  cat("\n====================================================\n")
  cat("Subconjunto:", nome_sub, "\n")
  cat("Variáveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")
  
  # --------------------------------------------------------------------------
  # RF sem balanceamento
  # --------------------------------------------------------------------------
  cat("Treinando RF sem balanceamento...\n")
  
  set.seed(123)
  modelo_rf_base <- caret::train(
    formula_sub,
    data = dados_sub,
    method = "rf",
    metric = "ROC",
    trControl = controle_cv,
    tuneGrid = grid_rf,
    importance = TRUE,
    ntree = 200
  )
  
  resultados[[contador]] <- extrair_melhor_resultado(
    modelo = modelo_rf_base,
    cenario = "Sem_balanceamento",
    subconjunto = nome_sub,
    vars_sub = vars_sub
  )
  contador <- contador + 1
  
  # --------------------------------------------------------------------------
  # RF com SMOTENC
  # --------------------------------------------------------------------------
  cat("Treinando RF com SMOTENC...\n")
  
  receita_smotenc <- recipes::recipe(formula_sub, data = dados_sub) %>%
    themis::step_smotenc(Class, over_ratio = 1, neighbors = 5)
  
  set.seed(123)
  modelo_rf_smotenc <- caret::train(
    receita_smotenc,
    data = dados_sub,
    method = "rf",
    metric = "ROC",
    trControl = controle_cv,
    tuneGrid = grid_rf,
    importance = TRUE,
    ntree = 200
  )
  
  resultados[[contador]] <- extrair_melhor_resultado(
    modelo = modelo_rf_smotenc,
    cenario = "Com_SMOTENC",
    subconjunto = nome_sub,
    vars_sub = vars_sub
  )
  contador <- contador + 1
}

# ------------------------------------------------------------------------------
# BLOCO 4 — Consolidar resultados
# ------------------------------------------------------------------------------
tabela_balanceamento <- dplyr::bind_rows(resultados) %>%
  dplyr::select(
    Subconjunto, Cenario,
    ROC, Sens, Spec, Precision, F1, GMean,
    dplyr::everything()
  ) %>%
  dplyr::arrange(Subconjunto, desc(ROC), desc(F1))

print(tabela_balanceamento)

# ------------------------------------------------------------------------------
# BLOCO 5 — Gráficos simples
# ------------------------------------------------------------------------------
grafico_roc_balanceamento <- ggplot2::ggplot(
  tabela_balanceamento,
  aes(x = Subconjunto, y = ROC, color = Cenario, group = Cenario)
) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 3) +
  ggplot2::geom_text(
    aes(label = round(ROC, 4)),
    vjust = -0.8,
    size = 3
  ) +
  ggplot2::labs(
    title = "RF: ROC sem balanceamento vs com SMOTENC",
    x = "Subconjunto",
    y = "ROC"
  ) +
  ggplot2::theme_minimal()

print(grafico_roc_balanceamento)

grafico_f1_balanceamento <- ggplot2::ggplot(
  tabela_balanceamento,
  aes(x = Subconjunto, y = F1, color = Cenario, group = Cenario)
) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 3) +
  ggplot2::geom_text(
    aes(label = round(F1, 4)),
    vjust = -0.8,
    size = 3
  ) +
  ggplot2::labs(
    title = "RF: F1 sem balanceamento vs com SMOTENC",
    x = "Subconjunto",
    y = "F1"
  ) +
  ggplot2::theme_minimal()

print(grafico_f1_balanceamento)

# ------------------------------------------------------------------------------
# BLOCO 6 — Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(tabela_balanceamento, "objetos/tabela_rf_balanceamento_smotenc.rds")
readr::write_csv(tabela_balanceamento, "resultados/tabela_rf_balanceamento_smotenc.csv")

ggplot2::ggsave(
  filename = "figuras/roc_rf_balanceamento_smotenc.png",
  plot = grafico_roc_balanceamento,
  width = 8,
  height = 5
)

ggplot2::ggsave(
  filename = "figuras/f1_rf_balanceamento_smotenc.png",
  plot = grafico_f1_balanceamento,
  width = 8,
  height = 5
)



message("06_rf_balanceamento_smotenc.R concluído com sucesso.")