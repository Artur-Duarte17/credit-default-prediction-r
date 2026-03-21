# ==============================================================================
# 04_topn_base.R
# Responsabilidade: testar subconjuntos Top-1 até Top-N usando
# regressão logística como modelo-base.
# ==============================================================================

source("00_setup.R")

# ------------------------------------------------------------------------------
# BLOCO 1 — Carregar treino e ranking
# ------------------------------------------------------------------------------
treino <- readRDS("objetos/treino.rds")
ranking_variaveis <- readRDS("objetos/ranking_variaveis_enet.rds")

print(ranking_variaveis)

# Vetor com a ordem das variáveis
ordem_variaveis <- ranking_variaveis$Variavel_Original

# ------------------------------------------------------------------------------
# BLOCO 2 — Controle de validação cruzada
# ------------------------------------------------------------------------------
controle_cv <- caret::trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel = FALSE
)

# ------------------------------------------------------------------------------
# BLOCO 3 — Função para montar fórmula com Top-K variáveis
# ------------------------------------------------------------------------------
montar_formula_topk <- function(k, ordem_variaveis) {
  vars_k <- ordem_variaveis[1:k]
  formula_texto <- paste("Class ~", paste(vars_k, collapse = " + "))
  as.formula(formula_texto)
}

# Teste rápido
print(montar_formula_topk(3, ordem_variaveis))

# ------------------------------------------------------------------------------
# BLOCO 4 — Loop Top-1 até Top-N
# ------------------------------------------------------------------------------
resultados_topn <- list()

for (k in seq_along(ordem_variaveis)) {
  
  cat("\n=============================\n")
  cat("Rodando Top-", k, "\n", sep = "")
  cat("=============================\n")
  
  formula_k <- montar_formula_topk(k, ordem_variaveis)
  
  set.seed(123)
  
  modelo_glm_k <- caret::train(
    formula_k,
    data = treino,
    method = "glm",
    family = binomial(),
    metric = "ROC",
    trControl = controle_cv
  )
  
  # Melhor linha do resultado do modelo
  res_k <- modelo_glm_k$results %>%
    dplyr::mutate(
      TopN = k,
      Variaveis = paste(ordem_variaveis[1:k], collapse = ", ")
    ) %>%
    dplyr::select(TopN, ROC, Sens, Spec, everything(), Variaveis)
  
  resultados_topn[[k]] <- res_k
}

# ------------------------------------------------------------------------------
# BLOCO 5 — Consolidar tabela final
# ------------------------------------------------------------------------------
tabela_topn <- dplyr::bind_rows(resultados_topn) %>%
  dplyr::arrange(desc(ROC), desc(Sens), desc(Spec))

print(tabela_topn)

# Top 10 melhores por ROC
top10_topn <- tabela_topn %>%
  dplyr::slice(1:10)

print(top10_topn)

# ------------------------------------------------------------------------------
# BLOCO 5A — Gráfico (Lollipop) dos 10 melhores Top-N por ROC
# ------------------------------------------------------------------------------
top10_plot <- top10_topn %>%
  dplyr::mutate(
    Subconjunto = paste0("Top-", TopN),
    Subconjunto = reorder(Subconjunto, ROC)
  )

grafico_melhores_topn <- ggplot2::ggplot(top10_plot, aes(x = ROC, y = Subconjunto)) +
  # Cria a linha do "pirulito"
  ggplot2::geom_segment(
    aes(x = min(ROC) - 0.001, xend = ROC, y = Subconjunto, yend = Subconjunto), 
    color = "#a8ddb5", 
    linewidth = 1
  ) +
  # Cria a ponta do "pirulito"
  ggplot2::geom_point(color = "#006d2c", size = 4) +
  ggplot2::geom_text(
    aes(label = round(ROC, 5)), 
    hjust = -0.3, 
    size = 3.5
  ) +
  ggplot2::labs(
    title = "Top 10 Melhores Subconjuntos por ROC",
    x = "Área Sob a Curva (ROC)",
    y = NULL
  ) +
  ggplot2::xlim(min(top10_plot$ROC) - 0.001, max(top10_plot$ROC) + 0.001) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", size = 14),
    panel.grid.major.y = ggplot2::element_blank()
  )

print(grafico_melhores_topn)
# ------------------------------------------------------------------------------
# BLOCO 6 — Curva por quantidade de variáveis
# ------------------------------------------------------------------------------
curva_topn <- dplyr::bind_rows(resultados_topn) %>%
  dplyr::arrange(TopN)

print(curva_topn)

grafico_roc_topn <- ggplot2::ggplot(curva_topn, aes(x = TopN, y = ROC)) +
  ggplot2::geom_line(color = "#2ca25f", linewidth = 1) +
  ggplot2::geom_point(color = "#2ca25f", size = 2) +
  # Destacar o ponto máximo absoluto
  ggplot2::geom_point(
    data = curva_topn %>% dplyr::filter(ROC == max(ROC)),
    aes(x = TopN, y = ROC),
    color = "#00441b", 
    size = 4
  ) +
  ggplot2::labs(
    title = "Desempenho (ROC) por Quantidade de Variáveis",
    subtitle = "Avaliação do impacto ao adicionar atributos sucessivos",
    x = "Quantidade de Variáveis (Top-N)",
    y = "Área Sob a Curva (ROC)"
  ) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", size = 14)
  )

print(grafico_roc_topn)

# ------------------------------------------------------------------------------
# BLOCO 7 — Identificar melhor Top-N
# ------------------------------------------------------------------------------
melhor_topn <- curva_topn %>%
  dplyr::slice_max(order_by = ROC, n = 1)

print(melhor_topn)

# ------------------------------------------------------------------------------
# BLOCO 8 — Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(curva_topn, "objetos/curva_topn_glm.rds")
write_csv(curva_topn, "resultados/curva_topn_glm.csv")
write_csv(top10_topn, "resultados/top10_topn_glm.csv")

ggplot2::ggsave(
  filename = "figuras/curva_roc_topn_glm.png",
  plot = grafico_roc_topn,
  width = 8,
  height = 5
)

ggplot2::ggsave(
  filename = "figuras/top10_melhores_subconjuntos_glm.png",
  plot = grafico_melhores_topn,
  width = 8,
  height = 5
)

message("04_topn_base.R concluído com sucesso.")