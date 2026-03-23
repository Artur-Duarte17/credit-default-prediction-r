# ==============================================================================
# 09_shap.R
# Responsabilidade: gerar interpretabilidade SHAP do modelo vencedor atual
# no conjunto de teste final.
# ==============================================================================

source("00_setup.R")

# ------------------------------------------------------------------------------
# BLOCO 1 — Carregar dados e identificar o modelo vencedor
# ------------------------------------------------------------------------------
treino <- readRDS("objetos/treino.rds")
teste  <- readRDS("objetos/teste.rds")
tabela_teste <- readRDS("objetos/tabela_teste_final.rds")
config_modelos <- readRDS("objetos/config_modelos_finais.rds")

parse_variaveis <- function(texto_variaveis) {
  trimws(strsplit(texto_variaveis, ",")[[1]])
}

melhor_cenario_teste <- tabela_teste %>%
  dplyr::slice_max(order_by = ROC, n = 1, with_ties = FALSE)

cenario_base <- sub("_Youden$", "", melhor_cenario_teste$Cenario)
config_modelo <- config_modelos %>%
  dplyr::filter(Cenario == cenario_base) %>%
  dplyr::slice(1)

vars_modelo <- parse_variaveis(config_modelo$Variaveis[1])

treino_sub <- treino[, c(vars_modelo, "Class")]
teste_sub  <- teste[, c(vars_modelo, "Class")]

print(melhor_cenario_teste)
print(config_modelo)
print(vars_modelo)

# ------------------------------------------------------------------------------
# BLOCO 2 — Funções auxiliares
# ------------------------------------------------------------------------------
alinhar_colunas_matriz <- function(x_ref, x_new) {
  x_ref <- as.matrix(x_ref)
  x_new <- as.matrix(x_new)

  cols_faltantes <- setdiff(colnames(x_ref), colnames(x_new))
  if (length(cols_faltantes) > 0) {
    matriz_zero <- matrix(
      0,
      nrow = nrow(x_new),
      ncol = length(cols_faltantes),
      dimnames = list(NULL, cols_faltantes)
    )
    x_new <- cbind(x_new, matriz_zero)
  }

  cols_extras <- setdiff(colnames(x_new), colnames(x_ref))
  if (length(cols_extras) > 0) {
    x_new <- x_new[, !colnames(x_new) %in% cols_extras, drop = FALSE]
  }

  x_new[, colnames(x_ref), drop = FALSE]
}

preparar_dados_modelo <- function(treino_df, teste_df, usar_smotenc, formula_modelo) {
  if (usar_smotenc) {
    receita_smotenc <- recipes::recipe(formula_modelo, data = treino_df) %>%
      themis::step_smotenc(Class, over_ratio = 1, neighbors = 5, skip = TRUE)

    prep_smotenc <- recipes::prep(receita_smotenc, training = treino_df, retain = TRUE)

    list(
      treino = recipes::juice(prep_smotenc),
      teste = recipes::bake(prep_smotenc, new_data = teste_df)
    )
  } else {
    list(
      treino = treino_df,
      teste = teste_df
    )
  }
}

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

# ------------------------------------------------------------------------------
# BLOCO 3 — Preparar base do modelo vencedor
# ------------------------------------------------------------------------------
formula_sub <- as.formula(
  paste("Class ~", paste(vars_modelo, collapse = " + "))
)

dados_modelo <- preparar_dados_modelo(
  treino_df = treino_sub,
  teste_df = teste_sub,
  usar_smotenc = isTRUE(config_modelo$Usa_SMOTENC),
  formula_modelo = formula_sub
)

treino_modelo <- dados_modelo$treino
teste_modelo  <- dados_modelo$teste

print(dim(treino_modelo))
print(dim(teste_modelo))

# ------------------------------------------------------------------------------
# BLOCO 4 — Treinar o modelo vencedor
# ------------------------------------------------------------------------------
threshold_final <- melhor_cenario_teste$Threshold

if (config_modelo$Modelo == "RF") {
  set.seed(123)
  modelo_final <- randomForest::randomForest(
    formula_sub,
    data = treino_modelo,
    mtry = config_modelo$mtry,
    ntree = 200,
    importance = TRUE
  )

  x_treino_bg <- as.data.frame(treino_modelo[, vars_modelo])
  x_teste_full <- as.data.frame(teste_modelo[, vars_modelo])

  pred_wrapper_modelo <- function(object, newdata) {
    newdata <- as.data.frame(newdata)
    predict(object, newdata = newdata, type = "prob")[, "Deve"]
  }

  coluna_dependencia <- "PAY_0"
} else {
  x_treino_bg <- model.matrix(Class ~ ., data = treino_modelo)[, -1, drop = FALSE]
  x_teste_full <- model.matrix(Class ~ ., data = teste_modelo)[, -1, drop = FALSE]
  x_teste_full <- alinhar_colunas_matriz(x_treino_bg, x_teste_full)

  dtrain <- xgboost::xgb.DMatrix(
    data = x_treino_bg,
    label = ifelse(treino_modelo$Class == "Deve", 1, 0)
  )

  set.seed(123)
  modelo_final <- xgboost::xgb.train(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = as.integer(config_modelo$max_depth),
      eta = config_modelo$eta,
      gamma = config_modelo$gamma,
      colsample_bytree = config_modelo$colsample_bytree,
      min_child_weight = config_modelo$min_child_weight,
      subsample = config_modelo$subsample
    ),
    data = dtrain,
    nrounds = as.integer(config_modelo$nrounds),
    verbose = 0
  )

  pred_wrapper_modelo <- function(object, newdata) {
    newdata <- as.matrix(newdata)
    newdata <- alinhar_colunas_matriz(x_treino_bg, newdata)
    predict(object, newdata = newdata)
  }

  coluna_dependencia <- if ("PAY_0" %in% colnames(x_treino_bg)) "PAY_0" else colnames(x_treino_bg)[1]
}

# ------------------------------------------------------------------------------
# BLOCO 5 — Definir amostra para SHAP
# ------------------------------------------------------------------------------
set.seed(123)
n_amostra_shap <- min(200, nrow(x_teste_full))
ids_amostra <- sample(seq_len(nrow(x_teste_full)), size = n_amostra_shap)

x_teste_shap <- x_teste_full[ids_amostra, , drop = FALSE]
y_teste_shap <- teste_modelo$Class[ids_amostra]

print(class(x_treino_bg))
print(class(x_teste_shap))
print(dim(x_treino_bg))
print(dim(x_teste_shap))

prob_teste_amostra <- pred_wrapper_modelo(modelo_final, x_teste_shap)
print(head(prob_teste_amostra))

# ------------------------------------------------------------------------------
# BLOCO 6 — Calcular SHAP aproximado
# ------------------------------------------------------------------------------
set.seed(123)
nsim_shap <- 30

shap_raw <- fastshap::explain(
  object = modelo_final,
  X = x_treino_bg,
  pred_wrapper = pred_wrapper_modelo,
  newdata = x_teste_shap,
  nsim = nsim_shap,
  adjust = TRUE,
  shap_only = FALSE
)

sv <- shapviz::shapviz(shap_raw)

print(sv)

# ------------------------------------------------------------------------------
# BLOCO 7 — Importância global e tabela Top-10
# ------------------------------------------------------------------------------
matriz_shap <- shap_raw$shapley_values
nomes_shap <- colnames(matriz_shap)

tabela_shap <- data.frame(
  Variavel = nomes_shap,
  Importancia_SHAP = colMeans(abs(matriz_shap))
) %>%
  dplyr::arrange(desc(Importancia_SHAP)) %>%
  dplyr::mutate(Posicao = dplyr::row_number())

print(tabela_shap)

if (config_modelo$Modelo == "XGBoost") {
  tabela_shap_agregada <- tabela_shap %>%
    dplyr::mutate(
      Variavel_Original = purrr::map_chr(
        Variavel,
        mapear_variavel_original,
        nomes_originais = vars_modelo
      )
    ) %>%
    dplyr::group_by(Variavel_Original) %>%
    dplyr::summarise(
      Importancia_SHAP = sum(Importancia_SHAP),
      .groups = "drop"
    ) %>%
    dplyr::arrange(desc(Importancia_SHAP)) %>%
    dplyr::mutate(Posicao = dplyr::row_number()) %>%
    dplyr::rename(Variavel = Variavel_Original)
} else {
  tabela_shap_agregada <- tabela_shap
}

top10_shap <- tabela_shap_agregada %>%
  dplyr::slice(1:10)

print(top10_shap)

metadata_shap <- tibble::tibble(
  Cenario = melhor_cenario_teste$Cenario,
  Modelo = config_modelo$Modelo,
  Usa_SMOTENC = config_modelo$Usa_SMOTENC,
  Threshold = threshold_final
)

print(metadata_shap)

# ------------------------------------------------------------------------------
# BLOCO 8 — Gráficos SHAP
# ------------------------------------------------------------------------------
grafico_shap_importancia <- shapviz::sv_importance(
  sv,
  kind = "bar",
  max_display = 10
)

print(grafico_shap_importancia)

grafico_shap_bee <- shapviz::sv_importance(
  sv,
  kind = "bee",
  max_display = 10
)

print(grafico_shap_bee)

grafico_dependencia <- shapviz::sv_dependence(
  sv,
  v = coluna_dependencia
)

print(grafico_dependencia)

classe_prevista <- ifelse(prob_teste_amostra >= threshold_final, "Deve", "Pago")
id_caso_deve <- which(classe_prevista == "Deve")[1]

if (!is.na(id_caso_deve)) {
  grafico_waterfall <- shapviz::sv_waterfall(
    sv,
    row_id = id_caso_deve,
    max_display = 10
  )

  print(grafico_waterfall)

  caso_explicado <- tibble::tibble(
    Linha_amostra = id_caso_deve,
    Classe_real = y_teste_shap[id_caso_deve],
    Prob_Deve = prob_teste_amostra[id_caso_deve],
    Classe_prevista = classe_prevista[id_caso_deve]
  )

  print(caso_explicado)
} else {
  grafico_waterfall <- NULL
  caso_explicado <- NULL
}

# ------------------------------------------------------------------------------
# BLOCO 9 — Salvar resultados
# ------------------------------------------------------------------------------
saveRDS(tabela_shap_agregada, "objetos/tabela_shap_final.rds")
readr::write_csv(tabela_shap_agregada, "resultados/tabela_shap_final.csv")
readr::write_csv(top10_shap, "resultados/top10_shap_final.csv")

saveRDS(metadata_shap, "objetos/metadata_shap_modelo.rds")
readr::write_csv(metadata_shap, "resultados/metadata_shap_modelo.csv")

ggplot2::ggsave(
  filename = "figuras/shap_importancia_top10.png",
  plot = grafico_shap_importancia,
  width = 8,
  height = 5
)

ggplot2::ggsave(
  filename = "figuras/shap_beeswarm_top10.png",
  plot = grafico_shap_bee,
  width = 9,
  height = 6
)

ggplot2::ggsave(
  filename = "figuras/shap_dependencia_pay0.png",
  plot = grafico_dependencia,
  width = 8,
  height = 5
)

if (!is.null(grafico_waterfall)) {
  ggplot2::ggsave(
    filename = "figuras/shap_waterfall_exemplo.png",
    plot = grafico_waterfall,
    width = 9,
    height = 5
  )
}

message("09_shap.R concluído com sucesso.")
