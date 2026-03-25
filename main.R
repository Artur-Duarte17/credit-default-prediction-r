# ==============================================================================
# 99_run_pipeline.R
# Responsabilidade: executar o pipeline completo com progresso e log.
# ==============================================================================

opcoes_execucao <- list(
  rodar_ate = "fim",   # base | exploratorio | confirmacao | final | fim
  parar_no_erro = TRUE,
  mostrar_tempo = TRUE,
  salvar_log = TRUE
)

scripts_pipeline <- list(
  base = c(
    "00_setup.R",
    "01_dados.R",
    "02_preprocessamento.R",
    "03_selecao_variaveis.R"
  ),
  exploratorio = c(
    "04_topn_base.R",
    "04B_topn_xgboost.R",
    "04C_topn_modelos_caret.R"
  ),
  confirmacao = c(
    "05A_glm_rf_subconjuntos.R",
    "05B_xgboost_subconjuntos.R",
    "05C_svm_subconjuntos.R",
    "05D_redes_neurais_subconjuntos.R",
    "06_rf_balanceamento_smotenc.R",
    "06B_xgb_balanceamento_smotenc.R",
    "06C_modelos_caret_balanceamento_smotenc.R"
  ),
  final = c(
    "07_threshold_calibracao.R",
    "08_avaliacao_teste.R"
  ),
  interpretabilidade = c(
    "09_shap.R",
    "10_resumo_final.R"
  )
)

expandir_scripts <- function(rodar_ate) {
  ordem_fases <- c("base", "exploratorio", "confirmacao", "final", "fim")

  if (!rodar_ate %in% ordem_fases) {
    stop("Valor inválido em opcoes_execucao$rodar_ate")
  }

  if (rodar_ate == "fim") {
    return(unlist(scripts_pipeline, use.names = FALSE))
  }

  idx <- match(rodar_ate, ordem_fases)
  fases_escolhidas <- ordem_fases[seq_len(idx)]
  fases_escolhidas <- fases_escolhidas[fases_escolhidas != "fim"]

  unlist(scripts_pipeline[fases_escolhidas], use.names = FALSE)
}

timestamp_msg <- function(...) {
  cat(sprintf("[%s] ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")), ..., "\n")
}

executar_script <- function(arquivo, i, total) {
  timestamp_msg(sprintf("(%d/%d) Iniciando %s", i, total, arquivo))
  inicio <- Sys.time()

  ok <- tryCatch(
    {
      source(arquivo, local = new.env(parent = globalenv()), echo = FALSE)
      TRUE
    },
    error = function(e) {
      timestamp_msg(sprintf("ERRO em %s: %s", arquivo, conditionMessage(e)))
      if (isTRUE(opcoes_execucao$parar_no_erro)) stop(e)
      FALSE
    }
  )

  fim <- Sys.time()
  tempo <- round(as.numeric(difftime(fim, inicio, units = "secs")), 2)

  if (ok) {
    timestamp_msg(sprintf("Concluído %s | %.2f segundos", arquivo, tempo))
  }

  ok
}

main <- function() {
  scripts <- expandir_scripts(opcoes_execucao$rodar_ate)
  total <- length(scripts)

  if (isTRUE(opcoes_execucao$salvar_log)) {
    dir.create("logs", showWarnings = FALSE, recursive = TRUE)
    log_file <- file.path(
      "logs",
      paste0("pipeline_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".log")
    )
    
    log_con <- file(log_file, open = "wt")
    
    sink(log_con, split = TRUE)
    sink(log_con, type = "message")
    
    on.exit({
      sink(type = "message")
      sink()
      close(log_con)
    }, add = TRUE)
  }

  timestamp_msg("Pipeline iniciado")
  timestamp_msg(sprintf("Total de scripts: %d", total))

  pb <- txtProgressBar(min = 0, max = total, style = 3)

  resultados <- logical(total)

  for (i in seq_along(scripts)) {
    resultados[i] <- executar_script(scripts[i], i, total)
    setTxtProgressBar(pb, i)
  }

  close(pb)

  timestamp_msg(sprintf("Pipeline finalizado | Sucesso: %d/%d", sum(resultados), total))
}

main()