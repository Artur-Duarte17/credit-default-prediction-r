# ==============================================================================
# 08_avaliacao_teste.R
# Responsabilidade: avaliar no teste apenas os cenarios finalistas calibrados.
# Nota metodologica:
# Sensibilidade mede a capacidade de encontrar inadimplentes reais.
# Especificidade mede a capacidade de preservar bons pagadores.
# Em credito, falso negativo costuma ser o erro mais caro; por isso a leitura
# final prioriza sensibilidade, sem ignorar ROC, F1, especificidade e custo.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_modelos.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar dados e configuracoes finais
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(readRDS("objetos/treino.rds"))
teste <- garantir_ordem_classe(readRDS("objetos/teste.rds"))
config_modelos <- readRDS("objetos/config_modelos_finais.rds")

print(config_modelos)

# ------------------------------------------------------------------------------
# BLOCO 2 - Treinar modelos finais e avaliar no teste
# ------------------------------------------------------------------------------
resultados_teste <- list()
curvas_roc <- list()

for (i in seq_len(nrow(config_modelos))) {
  config_atual <- config_modelos[i, , drop = FALSE]

  cat("\n====================================================\n")
  cat("Treinando cenario finalista:", config_atual$Cenario[1], "\n")
  cat("Modelo:", config_atual$Modelo[1], "\n")
  cat("====================================================\n")

  ajuste_final <- treinar_prever_modelo_final(
    config_modelo = config_atual,
    treino_df = treino,
    teste_df = teste
  )

  metricas_teste <- calcular_metricas_prob(
    obs = ajuste_final$teste$Class,
    prob_deve = ajuste_final$prob,
    threshold = config_atual$Threshold_Youden[1],
    cenario = paste0(config_atual$Cenario[1], "_Youden"),
    regra = "Youden"
  ) %>%
    dplyr::mutate(
      Modelo = config_atual$Modelo[1],
      Subconjunto = config_atual$Subconjunto[1],
      Usa_SMOTENC = config_atual$Usa_SMOTENC[1],
      TopN = config_atual$TopN[1],
      Origem_Finalista = config_atual$Origem_Finalista[1],
      Fase_Protocolo = config_atual$Fase_Protocolo[1]
    )

  resultados_teste[[i]] <- metricas_teste

  roc_obj <- pROC::roc(
    response = factor(ajuste_final$teste$Class, levels = c("Pago", "Deve")),
    predictor = ajuste_final$prob,
    levels = c("Pago", "Deve"),
    direction = "<",
    quiet = TRUE
  )

  curvas_roc[[i]] <- tibble::tibble(
    FPR = 1 - roc_obj$specificities,
    TPR = roc_obj$sensitivities,
    Cenario = paste0(config_atual$Cenario[1], "_Youden")
  )
}

tabela_teste <- dplyr::bind_rows(resultados_teste) %>%
  adicionar_metricas_negocio() %>%
  dplyr::arrange(desc(Sens), desc(ROC), desc(F1), Custo_Erro_Total)

print(tabela_teste)

# ------------------------------------------------------------------------------
# BLOCO 3 - Graficos
# ------------------------------------------------------------------------------
tabela_longa <- tabela_teste %>%
  dplyr::select(
    Cenario,
    ROC, PRAUC, Sens, Spec, Precision, F1, GMean, Accuracy
  ) %>%
  tidyr::pivot_longer(
    cols = c(ROC, PRAUC, Sens, Spec, Precision, F1, GMean, Accuracy),
    names_to = "Metrica",
    values_to = "Valor"
  )

grafico_metricas_teste <- ggplot2::ggplot(
  tabela_longa,
  ggplot2::aes(x = Metrica, y = Valor, fill = Cenario)
) +
  ggplot2::geom_col(position = "dodge") +
  ggplot2::geom_text(
    ggplot2::aes(label = round(Valor, 3)),
    position = ggplot2::position_dodge(width = 0.9),
    vjust = -0.4,
    size = 3
  ) +
  ggplot2::labs(
    title = "Comparacao final no conjunto de teste",
    subtitle = "Sensibilidade recebe prioridade analitica em credito",
    x = "Metrica",
    y = "Valor"
  ) +
  ggplot2::theme_minimal()

df_roc <- dplyr::bind_rows(curvas_roc)

grafico_roc_teste <- ggplot2::ggplot(
  df_roc,
  ggplot2::aes(x = FPR, y = TPR, color = Cenario)
) +
  ggplot2::geom_line(linewidth = 1) +
  ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  ggplot2::labs(
    title = "Curvas ROC no conjunto de teste",
    x = "1 - Especificidade",
    y = "Sensibilidade"
  ) +
  ggplot2::theme_minimal()

print(grafico_metricas_teste)
print(grafico_roc_teste)

# ------------------------------------------------------------------------------
# BLOCO 4 - Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(tabela_teste, "objetos/tabela_teste_final.rds")
readr::write_csv(tabela_teste, "resultados/tabela_teste_final.csv")

ggplot2::ggsave(
  filename = "figuras/comparacao_final_teste_metricas.png",
  plot = grafico_metricas_teste,
  width = 14,
  height = 7
)

ggplot2::ggsave(
  filename = "figuras/curvas_roc_teste.png",
  plot = grafico_roc_teste,
  width = 10,
  height = 6
)

message("08_avaliacao_teste.R concluido com sucesso.")
