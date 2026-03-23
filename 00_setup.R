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

FASES_SAIDA <- c("exploratorio", "confirmacao", "final", "interpretabilidade")
CLASSIFICACOES_FIGURA <- c("principal", "suplementar", "tecnico")

SALVAR_FIGURAS_SUPLEMENTARES <- FALSE
SALVAR_FIGURAS_TECNICAS <- FALSE

PASTAS_PROJETO <- unique(c(
  "dados",
  "resultados",
  "figuras",
  "objetos",
  "objetos/splits",
  "R",
  "docs",
  file.path("objetos", FASES_SAIDA),
  file.path("objetos", "exploratorio", "topn"),
  file.path("objetos", "exploratorio", "benchmark"),
  file.path("objetos", "confirmacao", "benchmark"),
  file.path("objetos", "confirmacao", "balanceamento"),
  file.path("objetos", "final", c("threshold", "teste")),
  file.path("objetos", "interpretabilidade", "shap"),
  file.path("resultados", FASES_SAIDA),
  file.path("resultados", "exploratorio", "topn"),
  file.path("resultados", "exploratorio", "benchmark"),
  file.path("resultados", "confirmacao", "benchmark"),
  file.path("resultados", "confirmacao", "balanceamento"),
  file.path("resultados", "final", c("threshold", "teste")),
  file.path("resultados", "interpretabilidade", "shap"),
  file.path("figuras", c(FASES_SAIDA, "suplementares")),
  file.path("figuras", "exploratorio", "topn"),
  file.path("figuras", "confirmacao", c("benchmark", "balanceamento")),
  file.path("figuras", "final", c("threshold", "teste")),
  file.path("figuras", "interpretabilidade", "shap"),
  file.path("figuras", "suplementares", FASES_SAIDA),
  file.path("figuras", "suplementares", "exploratorio", "topn"),
  file.path("figuras", "suplementares", "interpretabilidade", "shap"),
  file.path("figuras", "suplementares", "tecnico"),
  file.path("figuras", "suplementares", "tecnico", "confirmacao", c("benchmark", "balanceamento")),
  file.path("figuras", "suplementares", "tecnico", FASES_SAIDA),
  file.path("docs", "interpretabilidade")
))

garantir_pasta <- function(pasta) {
  if (!dir.exists(pasta)) {
    dir.create(pasta, recursive = TRUE)
  }

  pasta
}

normalizar_componentes_caminho <- function(...) {
  partes <- unlist(list(...), use.names = FALSE)
  partes <- partes[!is.na(partes)]
  partes <- partes[nzchar(partes)]
  partes
}

montar_caminho_saida <- function(diretorio_raiz, fase = NULL, arquivo = NULL, subpastas = NULL) {
  partes <- normalizar_componentes_caminho(diretorio_raiz, fase, subpastas, arquivo)
  caminho <- do.call(file.path, as.list(partes))
  garantir_pasta(dirname(caminho))
  caminho
}

caminho_objeto_saida <- function(fase, arquivo, subpastas = NULL) {
  montar_caminho_saida("objetos", fase = fase, arquivo = arquivo, subpastas = subpastas)
}

caminho_resultado_saida <- function(fase, arquivo, subpastas = NULL) {
  montar_caminho_saida("resultados", fase = fase, arquivo = arquivo, subpastas = subpastas)
}

caminho_figura_saida <- function(
  fase,
  arquivo,
  subpastas = NULL,
  classificacao = c("principal", "suplementar", "tecnico")
) {
  classificacao <- match.arg(classificacao)

  partes_base <- switch(
    classificacao,
    principal = c("figuras", fase),
    suplementar = c("figuras", "suplementares", fase),
    tecnico = c("figuras", "suplementares", "tecnico", fase)
  )

  partes <- normalizar_componentes_caminho(partes_base, subpastas, arquivo)
  caminho <- do.call(file.path, as.list(partes))
  garantir_pasta(dirname(caminho))
  caminho
}

caminho_documento_saida <- function(arquivo, subpastas = NULL) {
  montar_caminho_saida("docs", arquivo = arquivo, subpastas = subpastas)
}

resolver_caminho_existente <- function(caminho_preferencial, legados = character()) {
  candidatos <- unique(c(caminho_preferencial, legados))
  existentes <- candidatos[file.exists(candidatos)]

  if (length(existentes) == 0) {
    stop(
      sprintf(
        "Nenhum arquivo encontrado. Caminho esperado: %s",
        caminho_preferencial
      ),
      call. = FALSE
    )
  }

  existentes[1]
}

ler_rds_saida <- function(fase, arquivo, subpastas = NULL, legados = character()) {
  caminho <- resolver_caminho_existente(
    caminho_preferencial = caminho_objeto_saida(fase, arquivo, subpastas = subpastas),
    legados = legados
  )

  readRDS(caminho)
}

salvar_rds_saida <- function(objeto, fase, arquivo, subpastas = NULL) {
  caminho <- caminho_objeto_saida(fase = fase, arquivo = arquivo, subpastas = subpastas)
  saveRDS(objeto, caminho)
  invisible(caminho)
}

salvar_csv_saida <- function(tabela, fase, arquivo, subpastas = NULL) {
  caminho <- caminho_resultado_saida(fase = fase, arquivo = arquivo, subpastas = subpastas)
  readr::write_csv(tabela, caminho)
  invisible(caminho)
}

deve_salvar_figura <- function(classificacao) {
  switch(
    classificacao,
    principal = TRUE,
    suplementar = isTRUE(SALVAR_FIGURAS_SUPLEMENTARES),
    tecnico = isTRUE(SALVAR_FIGURAS_TECNICAS),
    FALSE
  )
}

salvar_figura_saida <- function(
  plot,
  fase,
  arquivo,
  subpastas = NULL,
  classificacao = c("principal", "suplementar", "tecnico"),
  width = 8,
  height = 5,
  dpi = 300,
  ...
) {
  classificacao <- match.arg(classificacao)

  if (!deve_salvar_figura(classificacao)) {
    return(invisible(NULL))
  }

  caminho <- caminho_figura_saida(
    fase = fase,
    arquivo = arquivo,
    subpastas = subpastas,
    classificacao = classificacao
  )

  ggplot2::ggsave(
    filename = caminho,
    plot = plot,
    width = width,
    height = height,
    dpi = dpi,
    ...
  )

  invisible(caminho)
}

salvar_texto_saida <- function(linhas, arquivo, subpastas = NULL) {
  caminho <- caminho_documento_saida(arquivo = arquivo, subpastas = subpastas)
  writeLines(linhas, caminho)
  invisible(caminho)
}

for (pasta in PASTAS_PROJETO) {
  garantir_pasta(pasta)
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
