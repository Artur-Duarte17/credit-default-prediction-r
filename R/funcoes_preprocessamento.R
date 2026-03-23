# ==============================================================================
# R/funcoes_preprocessamento.R
# Responsabilidade: helpers de split, formula, SMOTENC e matrizes.
# ==============================================================================

garantir_ordem_classe <- function(dados, resposta = "Class", positivo = "Deve", negativo = "Pago") {
  dados[[resposta]] <- factor(dados[[resposta]], levels = c(positivo, negativo))
  dados
}

parse_variaveis <- function(texto_variaveis) {
  if (length(texto_variaveis) == 0 || is.na(texto_variaveis) || !nzchar(texto_variaveis)) {
    return(character(0))
  }

  trimws(strsplit(texto_variaveis, ",")[[1]])
}

montar_formula <- function(vars, resposta = "Class") {
  stats::as.formula(paste(resposta, "~", paste(vars, collapse = " + ")))
}

montar_formula_topk <- function(k, ordem_variaveis, resposta = "Class") {
  montar_formula(ordem_variaveis[seq_len(k)], resposta = resposta)
}

obter_subconjuntos_fixos <- function(ordem_variaveis, tamanhos = c(10, 13, 14)) {
  tamanhos_validos <- tamanhos[tamanhos <= length(ordem_variaveis)]
  stats::setNames(
    lapply(tamanhos_validos, function(k) ordem_variaveis[seq_len(k)]),
    paste0("Top", tamanhos_validos)
  )
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

criar_splits_estratificados <- function(
  dados,
  proporcoes = SPLITS_TREINO_DISPONIVEIS,
  resposta = "Class",
  seed = SEED_PROJETO
) {
  dados <- garantir_ordem_classe(dados, resposta = resposta)

  purrr::set_names(as.character(proporcoes)) |>
    purrr::map(function(p_str) {
      p <- as.numeric(p_str)

      set.seed(seed)
      idx_treino <- caret::createDataPartition(dados[[resposta]], p = p, list = FALSE)

      list(
        proporcao_treino = p,
        treino = dados[idx_treino, , drop = FALSE],
        teste = dados[-idx_treino, , drop = FALSE]
      )
    })
}

salvar_splits_estratificados <- function(
  splits,
  dados,
  proporcao_padrao = SPLIT_TREINO_PADRAO,
  dir_base = "splits"
) {
  salvar_rds_base(dados, "dados_preprocessados.rds")

  for (nome_split in names(splits)) {
    split_atual <- splits[[nome_split]]
    sufixo <- gsub("\\.", "", sprintf("p%02d", round(split_atual$proporcao_treino * 100)))

    saveRDS(
      split_atual$treino,
      montar_caminho_saida("objetos", subpastas = dir_base, arquivo = paste0("treino_", sufixo, ".rds"))
    )
    saveRDS(
      split_atual$teste,
      montar_caminho_saida("objetos", subpastas = dir_base, arquivo = paste0("teste_", sufixo, ".rds"))
    )
  }

  nome_padrao <- as.character(proporcao_padrao)
  split_padrao <- splits[[nome_padrao]]

  salvar_rds_base(split_padrao$treino, "treino.rds")
  salvar_rds_base(split_padrao$teste, "teste.rds")
}

preparar_receita_smotenc <- function(
  formula_modelo,
  dados,
  over_ratio = SMOTENC_OVER_RATIO,
  neighbors = SMOTENC_NEIGHBORS
) {
  recipes::recipe(formula_modelo, data = dados) %>%
    themis::step_smotenc(
      Class,
      over_ratio = over_ratio,
      neighbors = neighbors,
      skip = TRUE
    )
}

preparar_treino_teste_modelo <- function(
  treino_df,
  teste_df,
  formula_modelo,
  usar_smotenc = FALSE
) {
  treino_df <- garantir_ordem_classe(treino_df)
  teste_df <- garantir_ordem_classe(teste_df)

  if (!usar_smotenc) {
    return(list(treino = treino_df, teste = teste_df, prep = NULL))
  }

  receita_smotenc <- preparar_receita_smotenc(
    formula_modelo = formula_modelo,
    dados = treino_df
  )

  prep_smotenc <- recipes::prep(receita_smotenc, training = treino_df, retain = TRUE)

  list(
    treino = recipes::juice(prep_smotenc),
    teste = recipes::bake(prep_smotenc, new_data = teste_df),
    prep = prep_smotenc
  )
}
