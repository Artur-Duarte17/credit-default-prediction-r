# ==============================================================================
# 00_setup.R
# Responsabilidade: instalar/carregar pacotes e definir configuracoes iniciais.
# ==============================================================================

options(repos = c(CRAN = "https://cran.rstudio.com"))

if (!interactive()) {
  # Em execucao batch, evita a criacao automatica do arquivo Rplots.pdf
  # quando algum grafico e impresso antes do ggsave().
  options(device = function(...) grDevices::pdf(file = NULL))
}

if (!require(pacman, quietly = TRUE)) install.packages("pacman")

pacman::p_load(
  tidyverse,
  readxl,
  caret,
  randomForest,
  e1071,
  xgboost,
  nnet,
  kernlab,
  themis,
  pROC,
  PRROC,
  MLmetrics,
  doParallel,
  knitr,
  kableExtra,
  glmnet,
  recipes,
  fastshap,
  shapviz
)

PASTAS_PROJETO <- c(
  "dados",
  "resultados",
  "figuras",
  "objetos",
  "objetos/splits",
  "R",
  "docs"
)

for (pasta in PASTAS_PROJETO) {
  if (!dir.exists(pasta)) dir.create(pasta, recursive = TRUE)
}

SEED_PROJETO <- 123
SPLITS_TREINO_DISPONIVEIS <- c(0.70, 0.80)
SPLIT_TREINO_PADRAO <- 0.70
CV_FOLDS_EXPLORATORIO <- 5
CV_REPEATS_EXPLORATORIO <- 1
CV_FOLDS_CONFIRMACAO <- 5
CV_REPEATS_CONFIRMACAO <- 2
CV_FOLDS_PADRAO <- CV_FOLDS_CONFIRMACAO
CV_REPEATS_PADRAO <- CV_REPEATS_CONFIRMACAO

TOPN_VIZINHANCA_CANDIDATA <- 2
TOPN_CANDIDATOS_FALLBACK <- c(10, 13, 14)
RODAR_TOPN_COMPLETO_MODELOS_CARET <- FALSE
N_FINALISTAS_CONFIRMACAO_SEM_BALANCEAMENTO <- 2
N_FINALISTAS_BALANCEAMENTO_POR_MODELO <- 1

SVM_C_EXPLORATORIO <- c(0.5, 1, 2)
NNET_SIZE_EXPLORATORIO <- c(3, 5)
NNET_DECAY_EXPLORATORIO <- c(0.001, 0.01)
AVNNET_SIZE_EXPLORATORIO <- NNET_SIZE_EXPLORATORIO
AVNNET_DECAY_EXPLORATORIO <- NNET_DECAY_EXPLORATORIO
SMOTENC_OVER_RATIO <- 1
SMOTENC_NEIGHBORS <- 5
CUSTO_FALSO_NEGATIVO <- 5000
CUSTO_FALSO_POSITIVO <- 1000
BENEFICIO_VERDADEIRO_POSITIVO <- 5000

set.seed(SEED_PROJETO)

message("00_setup.R carregado com sucesso.")
