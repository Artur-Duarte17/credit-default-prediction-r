# ==============================================================================
# 10_resumo_final.R
# Responsabilidade: carregar os principais resultados finais do projeto.
# ==============================================================================

source("00_setup.R")

tabela_teste <- readRDS("objetos/tabela_teste_final.rds")
tabela_shap  <- readRDS("objetos/tabela_shap_final.rds")

if (file.exists("objetos/metadata_shap_modelo.rds")) {
  metadata_shap <- readRDS("objetos/metadata_shap_modelo.rds")
} else {
  metadata_shap <- NULL
}

if (file.exists("objetos/tabela_benchmark_glm_rf_xgb_sem_balanceamento.rds")) {
  tabela_benchmark_modelos <- readRDS("objetos/tabela_benchmark_glm_rf_xgb_sem_balanceamento.rds")
} else {
  tabela_benchmark_modelos <- NULL
}

if (file.exists("objetos/tabela_rf_xgb_balanceamento_smotenc.rds")) {
  tabela_balanceamento_modelos <- readRDS("objetos/tabela_rf_xgb_balanceamento_smotenc.rds")
} else {
  tabela_balanceamento_modelos <- NULL
}

if (!is.null(tabela_benchmark_modelos)) {
  cat("\n================ BENCHMARK GLM / RF / XGBOOST ================\n")
  print(tabela_benchmark_modelos %>% dplyr::slice(1:10))
}

if (!is.null(tabela_balanceamento_modelos)) {
  cat("\n================ RF / XGBOOST COM E SEM SMOTENC ================\n")
  print(tabela_balanceamento_modelos %>% dplyr::slice(1:10))
}

cat("\n================ RESULTADO FINAL NO TESTE ================\n")
print(tabela_teste)

melhor_cenario <- tabela_teste %>%
  dplyr::slice_max(order_by = ROC, n = 1, with_ties = FALSE)

cat("\n================ MELHOR CENÁRIO NO TESTE ================\n")
print(melhor_cenario)

cat("\n================ TOP 10 SHAP ================\n")
print(tabela_shap %>% dplyr::slice(1:10))

cat("\n================ OBSERVAÇÃO SOBRE SHAP ================\n")
if (!is.null(metadata_shap) && metadata_shap$Cenario == melhor_cenario$Cenario) {
  cat("O SHAP atual corresponde ao modelo vencedor.\n")
} else {
  cat("O SHAP atual não corresponde ao melhor cenário do teste.\n")
  cat("A etapa 09 deve ser refeita para alinhar a interpretabilidade.\n")
}
