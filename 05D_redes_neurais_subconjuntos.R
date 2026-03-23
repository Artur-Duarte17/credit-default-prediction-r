# ==============================================================================
# 05D_redes_neurais_subconjuntos.R
# Responsabilidade:
# 1) fase exploratoria barata para NNET e avNNet em subconjuntos candidatos;
# 2) fase de confirmacao apenas nos finalistas sem balanceamento.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar dados e definir subconjuntos candidatos
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(readRDS("objetos/treino.rds"))
ranking_variaveis <- readRDS("objetos/ranking_variaveis_enet.rds")

ordem_variaveis <- ranking_variaveis$Variavel_Original
topn_candidatos <- obter_topn_candidatos(ordem_variaveis = ordem_variaveis)
subconjuntos <- obter_subconjuntos_candidatos(
  ordem_variaveis = ordem_variaveis,
  topn_candidatos = topn_candidatos$TopN
)
modelos_redes <- c("NNET", "avNNet")

folds_exploratorios <- criar_folds_estratificados(
  y = treino$Class,
  fase = "exploratorio"
)
folds_confirmacao <- criar_folds_estratificados(
  y = treino$Class,
  fase = "confirmacao"
)

print(topn_candidatos)

# ------------------------------------------------------------------------------
# BLOCO 2 - Benchmark exploratorio sem balanceamento
# ------------------------------------------------------------------------------
resultados_exploratorios <- list()
contador <- 1

for (nome_sub in names(subconjuntos)) {
  vars_sub <- subconjuntos[[nome_sub]]
  topn_sub <- as.integer(gsub("^Top", "", nome_sub))
  dados_sub <- treino[, c(vars_sub, "Class"), drop = FALSE]
  formula_sub <- montar_formula(vars_sub)

  cat("\n====================================================\n")
  cat("Fase exploratoria - Subconjunto:", nome_sub, "\n")
  cat("Variaveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")

  for (modelo_rede in modelos_redes) {
    modelo_ajustado <- treinar_modelo_caret(
      modelo = modelo_rede,
      formula_modelo = formula_sub,
      dados_sub = dados_sub,
      fase_validacao = "exploratorio",
      folds_cv = folds_exploratorios
    )

    resultados_exploratorios[[contador]] <- extrair_melhor_resultado_caret(
      modelo_ajustado,
      metadata = list(
        Subconjunto = nome_sub,
        TopN = topn_sub,
        Modelo = modelo_rede,
        Variaveis = paste(vars_sub, collapse = ", "),
        Base_Treino = "TreinoCompleto",
        Usa_SMOTENC = FALSE
      )
    ) %>%
      adicionar_contexto_validacao(
        fase = "exploratorio",
        folds_cv = folds_exploratorios
      )
    contador <- contador + 1
  }
}

tabela_exploratoria <- dplyr::bind_rows(resultados_exploratorios) %>%
  dplyr::left_join(topn_candidatos, by = "TopN") %>%
  dplyr::select(
    Subconjunto, TopN, Modelo,
    ROC, Sens, Spec, Precision, F1, GMean,
    dplyr::everything()
  ) %>%
  dplyr::arrange(Modelo, TopN)

finalistas_confirmacao <- selecionar_finalistas_modelagem(
  tabela = tabela_exploratoria,
  n_por_modelo = N_FINALISTAS_CONFIRMACAO_SEM_BALANCEAMENTO
)

print(finalistas_confirmacao)

# ------------------------------------------------------------------------------
# BLOCO 3 - Confirmacao sem balanceamento apenas nos finalistas
# ------------------------------------------------------------------------------
resultados_confirmacao <- list()

for (i in seq_len(nrow(finalistas_confirmacao))) {
  config_atual <- finalistas_confirmacao[i, , drop = FALSE]
  vars_sub <- parse_variaveis(config_atual$Variaveis[1])
  dados_sub <- treino[, c(vars_sub, "Class"), drop = FALSE]
  formula_sub <- montar_formula(vars_sub)

  cat("\n====================================================\n")
  cat("Fase de confirmacao -", config_atual$Modelo[1], "-", config_atual$Subconjunto[1], "\n")
  cat("Variaveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")

  modelo_ajustado <- treinar_modelo_caret(
    modelo = config_atual$Modelo[1],
    formula_modelo = formula_sub,
    dados_sub = dados_sub,
    fase_validacao = "confirmacao",
    folds_cv = folds_confirmacao
  )

  resultados_confirmacao[[i]] <- extrair_melhor_resultado_caret(
    modelo_ajustado,
    metadata = list(
      Subconjunto = config_atual$Subconjunto[1],
      TopN = config_atual$TopN[1],
      Modelo = config_atual$Modelo[1],
      Variaveis = config_atual$Variaveis[1],
      Base_Treino = "TreinoCompleto",
      Usa_SMOTENC = FALSE
    )
  ) %>%
    adicionar_contexto_validacao(
      fase = "confirmacao",
      folds_cv = folds_confirmacao
    ) %>%
    dplyr::left_join(topn_candidatos, by = "TopN")
}

# ------------------------------------------------------------------------------
# BLOCO 4 - Consolidar resultados confirmados
# ------------------------------------------------------------------------------
tabela_redes <- dplyr::bind_rows(resultados_confirmacao) %>%
  dplyr::select(
    Subconjunto, TopN, Modelo,
    ROC, Sens, Spec, Precision, F1, GMean,
    dplyr::everything()
  ) %>%
  ordenar_resultados_modelagem()

if (file.exists(caminho_objeto_saida("confirmacao", "tabela_benchmark_glm_rf_xgb_svm_sem_balanceamento.rds", subpastas = "benchmark")) ||
    file.exists("objetos/tabela_benchmark_glm_rf_xgb_svm_sem_balanceamento.rds")) {
  tabela_base <- ler_rds_saida(
    "confirmacao",
    "tabela_benchmark_glm_rf_xgb_svm_sem_balanceamento.rds",
    subpastas = "benchmark",
    legados = "objetos/tabela_benchmark_glm_rf_xgb_svm_sem_balanceamento.rds"
  )
} else if (file.exists(caminho_objeto_saida("confirmacao", "tabela_benchmark_glm_rf_xgb_sem_balanceamento.rds", subpastas = "benchmark")) ||
           file.exists("objetos/tabela_benchmark_glm_rf_xgb_sem_balanceamento.rds")) {
  tabela_base <- ler_rds_saida(
    "confirmacao",
    "tabela_benchmark_glm_rf_xgb_sem_balanceamento.rds",
    subpastas = "benchmark",
    legados = "objetos/tabela_benchmark_glm_rf_xgb_sem_balanceamento.rds"
  )
} else {
  tabela_base <- NULL
}

if (file.exists(caminho_objeto_saida("exploratorio", "tabela_benchmark_glm_rf_xgb_svm_sem_balanceamento_exploratorio.rds", subpastas = "benchmark")) ||
    file.exists("objetos/tabela_benchmark_glm_rf_xgb_svm_sem_balanceamento_exploratorio.rds")) {
  tabela_base_expl <- ler_rds_saida(
    "exploratorio",
    "tabela_benchmark_glm_rf_xgb_svm_sem_balanceamento_exploratorio.rds",
    subpastas = "benchmark",
    legados = "objetos/tabela_benchmark_glm_rf_xgb_svm_sem_balanceamento_exploratorio.rds"
  )
} else if (file.exists(caminho_objeto_saida("exploratorio", "tabela_benchmark_glm_rf_xgb_sem_balanceamento_exploratorio.rds", subpastas = "benchmark")) ||
           file.exists("objetos/tabela_benchmark_glm_rf_xgb_sem_balanceamento_exploratorio.rds")) {
  tabela_base_expl <- ler_rds_saida(
    "exploratorio",
    "tabela_benchmark_glm_rf_xgb_sem_balanceamento_exploratorio.rds",
    subpastas = "benchmark",
    legados = "objetos/tabela_benchmark_glm_rf_xgb_sem_balanceamento_exploratorio.rds"
  )
} else {
  tabela_base_expl <- NULL
}

if (!is.null(tabela_base)) {
  tabela_benchmark_completa <- dplyr::bind_rows(tabela_base, tabela_redes) %>%
    ordenar_resultados_modelagem()
} else {
  tabela_benchmark_completa <- tabela_redes
}

if (!is.null(tabela_base_expl)) {
  tabela_benchmark_exploratoria <- dplyr::bind_rows(tabela_base_expl, tabela_exploratoria) %>%
    ordenar_resultados_modelagem()
} else {
  tabela_benchmark_exploratoria <- tabela_exploratoria
}

print(tabela_benchmark_completa)

# ------------------------------------------------------------------------------
# BLOCO 5 - Grafico principal de benchmark confirmado
# ------------------------------------------------------------------------------
tabela_plot_benchmark <- tabela_benchmark_completa %>%
  dplyr::mutate(
    Cenario = reorder(paste0(Modelo, " / ", Subconjunto), ROC),
    Base = min(ROC, na.rm = TRUE) - 0.01
  )

grafico_benchmark_principal <- ggplot2::ggplot(
  tabela_plot_benchmark,
  ggplot2::aes(x = ROC, y = Cenario, color = Modelo)
) +
  ggplot2::geom_segment(
    ggplot2::aes(
      x = Base,
      xend = ROC,
      y = Cenario,
      yend = Cenario
    ),
    linewidth = 0.9,
    alpha = 0.5,
    show.legend = FALSE
  ) +
  ggplot2::geom_point(size = 3) +
  ggplot2::labs(
    title = "Benchmark confirmado entre modelos",
    subtitle = "Comparacao final sem balanceamento apos filtro exploratorio",
    x = "ROC",
    y = NULL
  ) +
  ggplot2::theme_minimal()

print(grafico_benchmark_principal)

# ------------------------------------------------------------------------------
# BLOCO 6 - Salvar resultados
# ------------------------------------------------------------------------------
salvar_rds_saida(
  tabela_exploratoria,
  "exploratorio",
  "tabela_redes_neurais_subconjuntos_sem_balanceamento_exploratorio.rds",
  subpastas = "benchmark"
)
salvar_csv_saida(
  tabela_exploratoria,
  "exploratorio",
  "tabela_redes_neurais_subconjuntos_sem_balanceamento_exploratorio.csv",
  subpastas = "benchmark"
)

salvar_rds_saida(
  tabela_redes,
  "confirmacao",
  "tabela_redes_neurais_subconjuntos_sem_balanceamento.rds",
  subpastas = "benchmark"
)
salvar_csv_saida(
  tabela_redes,
  "confirmacao",
  "tabela_redes_neurais_subconjuntos_sem_balanceamento.csv",
  subpastas = "benchmark"
)

salvar_rds_saida(
  tabela_benchmark_exploratoria,
  "exploratorio",
  "tabela_benchmark_modelos_sem_balanceamento_exploratorio.rds",
  subpastas = "benchmark"
)
salvar_csv_saida(
  tabela_benchmark_exploratoria,
  "exploratorio",
  "tabela_benchmark_modelos_sem_balanceamento_exploratorio.csv",
  subpastas = "benchmark"
)

salvar_rds_saida(
  tabela_benchmark_completa,
  "confirmacao",
  "tabela_benchmark_modelos_sem_balanceamento.rds",
  subpastas = "benchmark"
)
salvar_csv_saida(
  tabela_benchmark_completa,
  "confirmacao",
  "tabela_benchmark_modelos_sem_balanceamento.csv",
  subpastas = "benchmark"
)

salvar_figura_saida(
  plot = grafico_benchmark_principal,
  fase = "confirmacao",
  arquivo = "roc_benchmark_confirmado_principal.png",
  subpastas = "benchmark",
  classificacao = "principal",
  width = 9,
  height = 6
)

message("05D_redes_neurais_subconjuntos.R concluido com sucesso.")
