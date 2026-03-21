# ==============================================================================
# 02_preprocessamento.R
# Responsabilidade: tratar categorias, ajustar tipos das variáveis
# e criar treino/teste sem vazamento.
# ==============================================================================

source("00_setup.R")

# ------------------------------------------------------------------------------
# BLOCO 1 — Carregar base limpa
# ------------------------------------------------------------------------------
dados <- readRDS("objetos/dados_limpos.rds")

# ------------------------------------------------------------------------------
# BLOCO 2 — Tratar categorias problemáticas
# ------------------------------------------------------------------------------
# EDUCATION:
# 1 = graduate school
# 2 = university
# 3 = high school
# 4 = others
# 0, 5 e 6 aparecem na base mas por ser pouco será agrupado como "4"

dados <- dados %>%
  mutate(
    EDUCATION = case_when(
      EDUCATION %in% c(0, 5, 6) ~ 4,
      TRUE ~ EDUCATION
    ),
    
    MARRIAGE = case_when(
      MARRIAGE == 0 ~ 3,
      TRUE ~ MARRIAGE
    )
  )

# ------------------------------------------------------------------------------
# BLOCO 3 — Transformar variáveis categóricas em factor
# ------------------------------------------------------------------------------
dados <- dados %>%
  mutate(
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
  )

# ------------------------------------------------------------------------------
# BLOCO 4 — Conferência após tratamento
# ------------------------------------------------------------------------------
print(table(dados$SEX))
print(table(dados$EDUCATION))
print(table(dados$MARRIAGE))
print(str(dados))

# ------------------------------------------------------------------------------
# BLOCO 5 — Split treino/teste estratificado
# ------------------------------------------------------------------------------
# p = 0.80 -> 80% treino, 20% teste
# list = FALSE -> devolve índices em vez de lista

set.seed(123)

idx_treino <- caret::createDataPartition(dados$Class, p = 0.80, list = FALSE)

treino <- dados[idx_treino, ]
teste  <- dados[-idx_treino, ]

# ------------------------------------------------------------------------------
# BLOCO 6 — Conferir proporção das classes
# ------------------------------------------------------------------------------
cat("Proporção da classe no conjunto completo:\n")
print(prop.table(table(dados$Class)))

cat("\nProporção da classe no treino:\n")
print(prop.table(table(treino$Class)))

cat("\nProporção da classe no teste:\n")
print(prop.table(table(teste$Class)))

# ------------------------------------------------------------------------------
# BLOCO 7 — Salvar objetos
# ------------------------------------------------------------------------------
saveRDS(dados,  "objetos/dados_preprocessados.rds")
saveRDS(treino, "objetos/treino.rds")
saveRDS(teste,  "objetos/teste.rds")

message("02_preprocessamento.R concluído com sucesso.")