# ==============================================================================
# 02_preprocessamento.R
# Responsabilidade: tratar categorias, ajustar tipos e salvar splits 70/30 e
# 80/20 para manter o projeto reproduzivel.
# ==============================================================================

source("00_setup.R")
source("R/funcoes_preprocessamento.R")

# ------------------------------------------------------------------------------
# BLOCO 1 - Carregar base limpa
# ------------------------------------------------------------------------------
dados <- ler_rds_base("dados_limpos.rds")

# ------------------------------------------------------------------------------
# BLOCO 2 - Tratar categorias problematicas
# ------------------------------------------------------------------------------
dados <- dados %>%
  dplyr::mutate(
    EDUCATION = dplyr::case_when(
      EDUCATION %in% c(0, 5, 6) ~ 4,
      TRUE ~ EDUCATION
    ),
    MARRIAGE = dplyr::case_when(
      MARRIAGE == 0 ~ 3,
      TRUE ~ MARRIAGE
    )
  )

# ------------------------------------------------------------------------------
# BLOCO 3 - Ajustar tipos
# ------------------------------------------------------------------------------
dados <- dados %>%
  dplyr::mutate(
    SEX = factor(SEX, levels = c(1, 2), labels = c("Masculino", "Feminino")),
    EDUCATION = factor(
      EDUCATION,
      levels = c(1, 2, 3, 4),
      labels = c("PosGraduacao", "Universidade", "EnsinoMedio", "Outros")
    ),
    MARRIAGE = factor(
      MARRIAGE,
      levels = c(1, 2, 3),
      labels = c("Casado", "Solteiro", "Outros")
    )
  ) %>%
  garantir_ordem_classe()

# ------------------------------------------------------------------------------
# BLOCO 4 - Conferencia rapida
# ------------------------------------------------------------------------------
print(table(dados$SEX))
print(table(dados$EDUCATION))
print(table(dados$MARRIAGE))
print(str(dados))

# ------------------------------------------------------------------------------
# BLOCO 5 - Gerar splits estratificados
# ------------------------------------------------------------------------------
# O projeto passa a salvar explicitamente os cenarios 70/30 e 80/20.
# O split canonico usado pelos demais scripts fica configurado em
# SPLIT_TREINO_PADRAO, definido em 00_setup.R.
splits <- criar_splits_estratificados(
  dados = dados,
  proporcoes = SPLITS_TREINO_DISPONIVEIS,
  resposta = "Class",
  seed = SEED_PROJETO
)

salvar_splits_estratificados(
  splits = splits,
  dados = dados,
  proporcao_padrao = SPLIT_TREINO_PADRAO
)

# ------------------------------------------------------------------------------
# BLOCO 6 - Conferir proporcoes das classes
# ------------------------------------------------------------------------------
cat("Proporcao da classe no conjunto completo:\n")
print(prop.table(table(dados$Class)))

for (nome_split in names(splits)) {
  split_atual <- splits[[nome_split]]

  cat("\nProporcao treino no split", nome_split, ":\n")
  print(prop.table(table(split_atual$treino$Class)))

  cat("\nProporcao teste no split", nome_split, ":\n")
  print(prop.table(table(split_atual$teste$Class)))
}

message("02_preprocessamento.R concluido com sucesso.")
