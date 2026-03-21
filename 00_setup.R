# ==============================================================================
# 00_setup.R
# Responsabilidade: instalar/carregar pacotes e definir configurações iniciais.
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

pastas <- c("dados", "resultados", "figuras", "objetos")

for (pasta in pastas) {
  if (!dir.exists(pasta)) dir.create(pasta)
}

set.seed(123)

message("Setup concluído com sucesso.")
