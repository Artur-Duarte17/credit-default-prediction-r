# ==============================================================================
# 03_selecao_variaveis.R
# Responsabilidade: gerar ranking de variaveis com Elastic Net no treino.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_preprocessamento.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar treino
# ------------------------------------------------------------------------------
treino <- garantir_ordem_classe(ler_rds_base("treino.rds"))

# ------------------------------------------------------------------------------
# BLOCO 2 - Criar matriz de preditores
# ------------------------------------------------------------------------------
x_treino <- model.matrix(Class ~ ., data = treino)[, -1, drop = FALSE]
y_treino <- treino$Class

print(dim(x_treino))
print(levels(y_treino))

# ------------------------------------------------------------------------------
# BLOCO 3 - Controle de validacao cruzada
# ------------------------------------------------------------------------------
controle_cv <- caret::trainControl(
  method = "repeatedcv",
  number = CV_FOLDS_PADRAO,
  repeats = CV_REPEATS_PADRAO,
  classProbs = TRUE,
  summaryFunction = caret::twoClassSummary,
  allowParallel = usar_backend_paralelo()
)

# ------------------------------------------------------------------------------
# BLOCO 4 - Grade de tuning do Elastic Net
# ------------------------------------------------------------------------------
grid_enet <- expand.grid(
  alpha = c(0, 0.25, 0.5, 0.75, 1),
  lambda = 10 ^ seq(-3, 0, length.out = 25)
)

# ------------------------------------------------------------------------------
# BLOCO 5 - Treinar Elastic Net
# ------------------------------------------------------------------------------
set.seed(SEED_PROJETO)

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

print(modelo_enet)
print(modelo_enet$bestTune)

# ------------------------------------------------------------------------------
# BLOCO 6 - Extrair coeficientes do melhor modelo
# ------------------------------------------------------------------------------
coef_best <- coef(modelo_enet$finalModel, s = modelo_enet$bestTune$lambda)

coef_df <- data.frame(
  Variavel_Modelo = rownames(as.matrix(coef_best)),
  Coeficiente = as.numeric(coef_best)
) %>%
  dplyr::filter(Variavel_Modelo != "(Intercept)") %>%
  dplyr::mutate(AbsCoef = abs(Coeficiente))

nomes_originais <- setdiff(names(treino), "Class")

coef_df <- coef_df %>%
  dplyr::mutate(
    Variavel_Original = purrr::map_chr(
      Variavel_Modelo,
      mapear_variavel_original,
      nomes_originais = nomes_originais
    )
  )

# ------------------------------------------------------------------------------
# BLOCO 7 - Criar ranking final por variavel original
# ------------------------------------------------------------------------------
ranking_variaveis <- coef_df %>%
  dplyr::group_by(Variavel_Original) %>%
  dplyr::summarise(
    Importancia = max(AbsCoef),
    .groups = "drop"
  ) %>%
  dplyr::arrange(desc(Importancia)) %>%
  dplyr::mutate(Posicao = dplyr::row_number())

top10 <- ranking_variaveis %>%
  dplyr::slice(1:10)

print(ranking_variaveis)
print(top10)

# ------------------------------------------------------------------------------
# BLOCO 8 - Grafico do Top 10 Elastic Net
# ------------------------------------------------------------------------------
grafico_top10_enet <- ggplot2::ggplot(
  top10,
  ggplot2::aes(
    x = reorder(Variavel_Original, Importancia),
    y = Importancia
  )
) +
  ggplot2::geom_col(fill = "#2ca25f", width = 0.7) +
  ggplot2::coord_flip() +
  ggplot2::geom_text(
    ggplot2::aes(label = round(Importancia, 4)),
    hjust = -0.1,
    size = 3.5
  ) +
  ggplot2::labs(
    title = "Top 10 Variaveis - Ranking Elastic Net",
    subtitle = "Importancia baseada no maior coeficiente absoluto",
    x = NULL,
    y = "Importancia"
  ) +
  ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.15))) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", size = 14),
    panel.grid.major.y = ggplot2::element_blank()
  )

print(grafico_top10_enet)

# ------------------------------------------------------------------------------
# BLOCO 9 - Salvar resultados
# ------------------------------------------------------------------------------
salvar_rds_base(modelo_enet, "modelo_enet_ranking.rds")
salvar_rds_base(ranking_variaveis, "ranking_variaveis_enet.rds")
salvar_csv_base(ranking_variaveis, "ranking_variaveis_enet.csv")
salvar_csv_base(coef_df, "coeficientes_enet_dummies.csv")

salvar_figura_base(
  plot = grafico_top10_enet,
  arquivo = "top10_ranking_enet_principal.png",
  classificacao = "principal",
  width = 8,
  height = 5
)

message("03_selecao_variaveis.R concluido com sucesso.")
