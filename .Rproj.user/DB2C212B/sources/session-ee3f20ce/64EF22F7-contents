# ==============================================================================
# 03_selecao_variaveis.R
# Responsabilidade: gerar ranking de variáveis com Elastic Net
# usando apenas o conjunto de treino.
# ==============================================================================

source("00_setup.R")

# ------------------------------------------------------------------------------
# BLOCO 1 — Carregar treino
# ------------------------------------------------------------------------------
treino <- readRDS("objetos/treino.rds")

# ------------------------------------------------------------------------------
# BLOCO 2 — Criar matriz de preditores
# ------------------------------------------------------------------------------
# O glmnet trabalha com matriz numérica.
# Como temos fatores (SEX, EDUCATION, MARRIAGE), o model.matrix()
# transforma essas categorias em dummies automaticamente.

x_treino <- model.matrix(Class ~ ., data = treino)[, -1]
y_treino <- treino$Class

# Conferência rápida
print(dim(x_treino))
print(levels(y_treino))

# ------------------------------------------------------------------------------
# BLOCO 3 — Controle de validação cruzada
# ------------------------------------------------------------------------------
# twoClassSummary usa ROC, Sens e Spec
# classProbs = TRUE permite trabalhar com probabilidades

controle_cv <- caret::trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel = FALSE
)

# ------------------------------------------------------------------------------
# BLOCO 4 — Grade de tuning do Elastic Net
# ------------------------------------------------------------------------------
# alpha:
# 0   = Ridge
# 1   = LASSO
# entre 0 e 1 = Elastic Net
#
# lambda:
# força da penalização

grid_enet <- expand.grid(
  alpha = c(0, 0.25, 0.5, 0.75, 1),
  lambda = 10 ^ seq(-3, 0, length.out = 25)
)

# ------------------------------------------------------------------------------
# BLOCO 5 — Treinar Elastic Net
# ------------------------------------------------------------------------------
set.seed(123)

modelo_enet <- caret::train(
  x = x_treino,
  y = y_treino,
  method = "glmnet",
  family = "binomial",
  metric = "ROC",
  trControl = controle_cv,
  tuneGrid = grid_enet,
  preProcess = c("center", "scale"),
  standardize = FALSE
)

# Resultado do tuning
print(modelo_enet)
print(modelo_enet$bestTune)

# ------------------------------------------------------------------------------
# BLOCO 6 — Extrair coeficientes do melhor modelo
# ------------------------------------------------------------------------------
# Vamos pegar os coeficientes do melhor alpha/lambda encontrado.

coef_best <- coef(modelo_enet$finalModel, s = modelo_enet$bestTune$lambda)

coef_df <- data.frame(
  Variavel_Modelo = rownames(as.matrix(coef_best)),
  Coeficiente = as.numeric(coef_best)
) %>%
  dplyr::filter(Variavel_Modelo != "(Intercept)") %>%
  dplyr::mutate(AbsCoef = abs(Coeficiente))

print(coef_df)

# ------------------------------------------------------------------------------
# BLOCO 7 — Mapear dummies para variável original
# ------------------------------------------------------------------------------
# Exemplo:
# SEXFeminino -> SEX
# EDUCATIONUniversidade -> EDUCATION
# PAY_0 -> PAY_0
#
# Isso é importante porque o ranking Top-N deve ser por variável original,
# não por dummy separada.

nomes_originais <- setdiff(names(treino), "Class")

mapear_variavel_original <- function(nome_modelo, nomes_originais) {
  candidatos <- nomes_originais[stringr::str_starts(
    string = nome_modelo,
    pattern = stringr::fixed(nomes_originais)
  )]
  
  if (length(candidatos) == 0) {
    return(nome_modelo)
  }
  
  candidatos[which.max(nchar(candidatos))]
}

coef_df <- coef_df %>%
  dplyr::mutate(
    Variavel_Original = purrr::map_chr(
      Variavel_Modelo,
      mapear_variavel_original,
      nomes_originais = nomes_originais
    )
  )

print(coef_df)

# ------------------------------------------------------------------------------
# BLOCO 8 — Criar ranking final por variável original
# ------------------------------------------------------------------------------
# Como variáveis categóricas geram várias dummies, vamos resumir por variável
# original usando o MAIOR coeficiente absoluto.
#
# Por que usar max() e não soma()?
# Porque somar favoreceria variáveis com mais categorias/dummies.

ranking_variaveis <- coef_df %>%
  dplyr::group_by(Variavel_Original) %>%
  dplyr::summarise(
    Importancia = max(AbsCoef),
    .groups = "drop"
  ) %>%
  dplyr::arrange(desc(Importancia)) %>%
  dplyr::mutate(Posicao = dplyr::row_number())

print(ranking_variaveis)

# Top 10
top10 <- ranking_variaveis %>%
  dplyr::slice(1:10)

print(top10)

# ------------------------------------------------------------------------------
# BLOCO 8A — Gráfico do Top 10 ranking Elastic Net
# ------------------------------------------------------------------------------
grafico_top10_enet <- ggplot2::ggplot(
  top10,
  aes(
    x = reorder(Variavel_Original, Importancia),
    y = Importancia
  )
) +
  ggplot2::geom_col(fill = "#2ca25f", width = 0.7) + # Preenchimento verde elegante
  ggplot2::coord_flip() +
  ggplot2::geom_text(
    aes(label = round(Importancia, 4)), 
    hjust = -0.1, 
    size = 3.5, 
    color = "black"
  ) +
  ggplot2::labs(
    title = "Top 10 Variáveis - Ranking Elastic Net",
    subtitle = "Importância baseada no maior coeficiente absoluto",
    x = NULL, # Remove o título do eixo Y (já fica óbvio pelos nomes)
    y = "Importância"
  ) +
  ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.15))) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", size = 14),
    panel.grid.major.y = ggplot2::element_blank() # Limpa linhas horizontais do fundo
  )

print(grafico_top10_enet)

# ------------------------------------------------------------------------------
# BLOCO 9 — Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(modelo_enet, "objetos/modelo_enet_ranking.rds")
saveRDS(ranking_variaveis, "objetos/ranking_variaveis_enet.rds")
write_csv(ranking_variaveis, "resultados/ranking_variaveis_enet.csv")
write_csv(coef_df, "resultados/coeficientes_enet_dummies.csv")

ggplot2::ggsave(
  filename = "figuras/top10_ranking_enet.png",
  plot = grafico_top10_enet,
  width = 8,
  height = 5
)

message("03_selecao_variaveis.R concluído com sucesso.")