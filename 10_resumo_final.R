# ==============================================================================
# 10_resumo_final.R
# Responsabilidade: consolidar benchmark, teste, negocio e interpretabilidade.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_metricas.R")

tabela_teste <- readRDS("objetos/tabela_teste_final.rds")
tabela_shap <- readRDS("objetos/tabela_shap_final.rds")
tabela_shap_vs_enet <- readRDS("objetos/tabela_shap_vs_elastic_net.rds")

if (file.exists("objetos/metadata_shap_modelo.rds")) {
  metadata_shap <- readRDS("objetos/metadata_shap_modelo.rds")
} else {
  metadata_shap <- NULL
}

if (file.exists("objetos/tabela_benchmark_modelos_sem_balanceamento.rds")) {
  tabela_benchmark_modelos <- readRDS("objetos/tabela_benchmark_modelos_sem_balanceamento.rds")
} else if (file.exists("objetos/tabela_benchmark_glm_rf_xgb_svm_sem_balanceamento.rds")) {
  tabela_benchmark_modelos <- readRDS("objetos/tabela_benchmark_glm_rf_xgb_svm_sem_balanceamento.rds")
} else if (file.exists("objetos/tabela_benchmark_glm_rf_xgb_sem_balanceamento.rds")) {
  tabela_benchmark_modelos <- readRDS("objetos/tabela_benchmark_glm_rf_xgb_sem_balanceamento.rds")
} else {
  tabela_benchmark_modelos <- NULL
}

if (!is.null(tabela_benchmark_modelos)) {
  cat("\n================ BENCHMARK MODELOS SEM BALANCEAMENTO ================\n")
  print(tabela_benchmark_modelos %>% dplyr::slice(1:10))
}

cat("\n================ RESULTADO FINAL NO TESTE ================\n")
print(tabela_teste)

melhor_cenario <- tabela_teste %>%
  dplyr::slice(1)

cat("\n================ MELHOR CENARIO NO TESTE ================\n")
print(melhor_cenario)

cat("\n================ TOP 10 SHAP ================\n")
print(tabela_shap %>% dplyr::slice(1:10))

cat("\n================ SHAP VS ELASTIC NET ================\n")
print(tabela_shap_vs_enet %>% dplyr::slice(1:10))

cat("\n================ METRICAS DE NEGOCIO ================\n")
cat("Supostos ilustrativos:\n")
cat("* custo de falso negativo =", CUSTO_FALSO_NEGATIVO, "\n")
cat("* custo de falso positivo =", CUSTO_FALSO_POSITIVO, "\n")
cat("* beneficio de verdadeiro positivo =", BENEFICIO_VERDADEIRO_POSITIVO, "\n")
print(
  tabela_teste %>%
    dplyr::select(
      Cenario, Sens, Spec, F1, TP, FP, FN, TN,
      Custo_FN, Custo_FP, Custo_Erro_Total, Beneficio_TP, Resultado_Liquido
    )
)

cat("\n================ OBSERVACAO SOBRE SHAP ================\n")
if (!is.null(metadata_shap) && metadata_shap$Cenario == melhor_cenario$Cenario) {
  cat("O SHAP atual corresponde ao modelo vencedor.\n")
} else {
  cat("O SHAP atual nao corresponde ao melhor cenario do teste.\n")
  cat("A etapa 09 deve ser refeita para alinhar a interpretabilidade.\n")
}
