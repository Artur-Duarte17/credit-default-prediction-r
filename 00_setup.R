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
CV_FOLDS_PADRAO <- 5
CV_REPEATS_PADRAO <- 2
SMOTENC_OVER_RATIO <- 1
SMOTENC_NEIGHBORS <- 5
CUSTO_FALSO_NEGATIVO <- 5000
CUSTO_FALSO_POSITIVO <- 1000
BENEFICIO_VERDADEIRO_POSITIVO <- 5000

set.seed(SEED_PROJETO)

message("00_setup.R carregado com sucesso.")
