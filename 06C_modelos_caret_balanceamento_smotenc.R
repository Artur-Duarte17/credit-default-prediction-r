# ==============================================================================
# 06C_modelos_caret_balanceamento_smotenc.R
# Responsabilidade: fase de confirmacao para SVM radial, NNET e avNNet,
# comparando sem balanceamento versus com SMOTENC apenas nos subconjuntos
# finalistas de cada modelo. O passo step_smotenc() fica sempre com skip = TRUE.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar dados e finalistas confirmados
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(ler_rds_base("treino.rds"))
modelos_balancear <- c("SVM_Radial", "NNET", "avNNet")

tabela_sem_balanceamento <- dplyr::bind_rows(
  ler_rds_saida(
    "confirmacao",
    "tabela_svm_subconjuntos_sem_balanceamento.rds",
    subpastas = "benchmark",
    legados = caminho_objeto_legado("tabela_svm_subconjuntos_sem_balanceamento.rds")
  ),
  ler_rds_saida(
    "confirmacao",
    "tabela_redes_neurais_subconjuntos_sem_balanceamento.rds",
    subpastas = "benchmark",
    legados = caminho_objeto_legado("tabela_redes_neurais_subconjuntos_sem_balanceamento.rds")
  )
) %>%
  dplyr::filter(Modelo %in% modelos_balancear)

finalistas_modelos <- selecionar_finalistas_modelagem(
  tabela = tabela_sem_balanceamento,
  n_por_modelo = N_FINALISTAS_BALANCEAMENTO_POR_MODELO
)

folds_confirmacao <- criar_folds_estratificados(
  y = treino$Class,
  fase = "confirmacao"
)

print(finalistas_modelos)

resultados <- list()
contador <- 1

treinar_cenario_balanceamento_caret <- function(
  config_atual,
  formula_sub,
  dados_sub,
  folds_confirmacao,
  usar_smotenc,
  tune_grid_atual
) {
  cenario <- if (isTRUE(usar_smotenc)) "Com_SMOTENC" else "Sem_balanceamento"

  resultado <- tryCatch(
    {
      modelo_treinado <- treinar_modelo_caret(
        modelo = config_atual$Modelo[1],
        formula_modelo = formula_sub,
        dados_sub = dados_sub,
        tune_grid = tune_grid_atual,
        usar_smotenc = usar_smotenc,
        fase_validacao = "confirmacao",
        folds_cv = folds_confirmacao
      )

      extrair_melhor_resultado_caret(
        modelo_treinado,
        metadata = list(
          Subconjunto = config_atual$Subconjunto[1],
          TopN = config_atual$TopN[1],
          Modelo = config_atual$Modelo[1],
          Cenario = cenario,
          Variaveis = config_atual$Variaveis[1],
          Usa_SMOTENC = usar_smotenc
        )
      ) %>%
        adicionar_contexto_validacao(
          fase = "confirmacao",
          folds_cv = folds_confirmacao
        )
    },
    error = function(e) {
      message(
        sprintf(
          "Ignorando %s | %s | %s: %s",
          config_atual$Modelo[1],
          config_atual$Subconjunto[1],
          cenario,
          conditionMessage(e)
        )
      )
      NULL
    }
  )

  if (is.null(resultado)) {
    return(NULL)
  }

  metricas_principais <- intersect(c("ROC", "F1", "GMean"), names(resultado))
  possui_metrica_valida <- length(metricas_principais) > 0 &&
    any(vapply(resultado[metricas_principais], function(coluna) any(is.finite(coluna)), logical(1)))

  if (!possui_metrica_valida) {
    message(
      sprintf(
        "Ignorando %s | %s | %s: sem metricas validas apos o treino.",
        config_atual$Modelo[1],
        config_atual$Subconjunto[1],
        cenario
      )
    )
    return(NULL)
  }

  resultado
}

# ------------------------------------------------------------------------------
# BLOCO 2 - Confirmacao do balanceamento nos finalistas
# ------------------------------------------------------------------------------
for (i in seq_len(nrow(finalistas_modelos))) {
  config_atual <- finalistas_modelos[i, , drop = FALSE]
  vars_sub <- parse_variaveis(config_atual$Variaveis[1])
  dados_sub <- treino[, c(vars_sub, "Class"), drop = FALSE]
  formula_sub <- montar_formula(vars_sub)
  tune_grid_atual <- obter_grid_modelo_config(config_atual)

  cat("\n====================================================\n")
  cat("Confirmacao de balanceamento -", config_atual$Modelo[1], "-", config_atual$Subconjunto[1], "\n")
  cat("Variaveis:", paste(vars_sub, collapse = ", "), "\n")
  cat("====================================================\n")

  resultado_base <- treinar_cenario_balanceamento_caret(
    config_atual = config_atual,
    formula_sub = formula_sub,
    dados_sub = dados_sub,
    folds_confirmacao = folds_confirmacao,
    usar_smotenc = FALSE,
    tune_grid_atual = tune_grid_atual
  )

  if (!is.null(resultado_base)) {
    resultados[[contador]] <- resultado_base
    contador <- contador + 1
  }

  resultado_smotenc <- treinar_cenario_balanceamento_caret(
    config_atual = config_atual,
    formula_sub = formula_sub,
    dados_sub = dados_sub,
    folds_confirmacao = folds_confirmacao,
    usar_smotenc = TRUE,
    tune_grid_atual = tune_grid_atual
  )

  if (!is.null(resultado_smotenc)) {
    resultados[[contador]] <- resultado_smotenc
    contador <- contador + 1
  }
}

if (length(resultados) == 0) {
  tabela_balanceamento <- tibble::tibble(
    Subconjunto = character(),
    TopN = integer(),
    Modelo = character(),
    Cenario = character(),
    ROC = double(),
    Sens = double(),
    Spec = double(),
    Precision = double(),
    F1 = double(),
    GMean = double(),
    Variaveis = character(),
    Usa_SMOTENC = logical(),
    Fase_Protocolo = character(),
    CV_Folds = double(),
    CV_Repeats = double()
  )
} else {
  tabela_balanceamento <- dplyr::bind_rows(resultados) %>%
    dplyr::select(
      Subconjunto, TopN, Modelo, Cenario,
      ROC, Sens, Spec, Precision, F1, GMean,
      dplyr::everything()
    ) %>%
    dplyr::arrange(Modelo, Subconjunto, desc(ROC), desc(F1), desc(GMean))
}

print(tabela_balanceamento)

salvar_rds_saida(
  tabela_balanceamento,
  "confirmacao",
  "tabela_modelos_caret_balanceamento_smotenc.rds",
  subpastas = "balanceamento"
)
salvar_csv_saida(
  tabela_balanceamento,
  "confirmacao",
  "tabela_modelos_caret_balanceamento_smotenc.csv",
  subpastas = "balanceamento"
)

tabelas_balanceamento_previas <- list()

tabela_rf_balanceamento <- ler_rds_saida(
  "confirmacao",
  "tabela_rf_balanceamento_smotenc.rds",
  subpastas = "balanceamento",
  legados = caminho_objeto_legado("tabela_rf_balanceamento_smotenc.rds"),
  obrigatorio = FALSE
)
if (!is.null(tabela_rf_balanceamento)) {
  tabelas_balanceamento_previas[[length(tabelas_balanceamento_previas) + 1]] <- tabela_rf_balanceamento
}

tabela_xgb_balanceamento <- ler_rds_saida(
  "confirmacao",
  "tabela_xgb_balanceamento_smotenc.rds",
  subpastas = "balanceamento",
  legados = caminho_objeto_legado("tabela_xgb_balanceamento_smotenc.rds"),
  obrigatorio = FALSE
)
if (!is.null(tabela_xgb_balanceamento)) {
  tabelas_balanceamento_previas[[length(tabelas_balanceamento_previas) + 1]] <- tabela_xgb_balanceamento
}

tabela_balanceamento_total <- dplyr::bind_rows(
  c(tabelas_balanceamento_previas, list(tabela_balanceamento))
)

salvar_rds_saida(
  tabela_balanceamento_total,
  "confirmacao",
  "tabela_balanceamento_modelos_confirmados.rds",
  subpastas = "balanceamento"
)
salvar_csv_saida(
  tabela_balanceamento_total,
  "confirmacao",
  "tabela_balanceamento_modelos_confirmados.csv",
  subpastas = "balanceamento"
)

tabela_balanceamento_wide <- tabela_balanceamento_total %>%
  dplyr::select(Modelo, Subconjunto, Cenario, ROC) %>%
  dplyr::filter(Cenario %in% c("Sem_balanceamento", "Com_SMOTENC")) %>%
  tidyr::pivot_wider(
    names_from = Cenario,
    values_from = ROC
  ) %>%
  dplyr::mutate(
    Cenario_Modelo = reorder(paste0(Modelo, " / ", Subconjunto), Sem_balanceamento)
  )

tabela_balanceamento_longa <- tabela_balanceamento_wide %>%
  dplyr::select(Cenario_Modelo, Sem_balanceamento, Com_SMOTENC) %>%
  tidyr::pivot_longer(
    cols = c(Sem_balanceamento, Com_SMOTENC),
    names_to = "Cenario",
    values_to = "ROC"
  )

grafico_balanceamento_principal <- ggplot2::ggplot(
  tabela_balanceamento_longa,
  ggplot2::aes(x = ROC, y = Cenario_Modelo, color = Cenario)
) +
  ggplot2::geom_segment(
    data = tabela_balanceamento_wide,
    ggplot2::aes(
      x = Sem_balanceamento,
      xend = Com_SMOTENC,
      y = Cenario_Modelo,
      yend = Cenario_Modelo
    ),
    inherit.aes = FALSE,
    color = "#bdbdbd",
    linewidth = 1,
    show.legend = FALSE
  ) +
  ggplot2::geom_point(size = 3) +
  ggplot2::scale_color_manual(
    values = c(
      Sem_balanceamento = "#636363",
      Com_SMOTENC = "#2c7fb8"
    ),
    labels = c(
      Sem_balanceamento = "Sem SMOTENC",
      Com_SMOTENC = "Com SMOTENC"
    )
  ) +
  ggplot2::labs(
    title = "Impacto do SMOTENC nos finalistas confirmados",
    subtitle = "ROC por modelo no melhor subconjunto confirmado",
    x = "ROC",
    y = NULL,
    color = NULL
  ) +
  ggplot2::theme_minimal()

print(grafico_balanceamento_principal)

salvar_figura_saida(
  plot = grafico_balanceamento_principal,
  fase = "confirmacao",
  arquivo = "roc_balanceamento_confirmado_principal.png",
  subpastas = "balanceamento",
  classificacao = "principal",
  width = 10,
  height = 6
)

message("06C_modelos_caret_balanceamento_smotenc.R concluido com sucesso.")
