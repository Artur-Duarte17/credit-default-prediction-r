# ==============================================================================
# 01_dados.R
# Responsabilidade: ler a base, inspecionar, fazer limpeza inicial
# e salvar uma versão organizada para as próximas etapas.
# ==============================================================================

source("00_setup.R")

# ------------------------------------------------------------------------------
# BLOCO 1 — Ler a base
# ------------------------------------------------------------------------------
# Importante:
# nesse arquivo .xls, a primeira linha costuma ser só o título da planilha.
# Por isso usamos skip = 1 para começar a leitura na linha correta.

dados_raw <- readxl::read_excel(
  path = "dados/default_of_credit_card_clients.xls",
  skip = 1
)

# ------------------------------------------------------------------------------
# BLOCO 2 — Inspeção inicial
# ------------------------------------------------------------------------------
print(dim(dados_raw))      # quantidade de linhas e colunas
print(names(dados_raw))    # nomes das colunas
glimpse(dados_raw)         # estrutura geral da base
head(dados_raw)            # primeiras linhas

# ------------------------------------------------------------------------------
# BLOCO 3 — Renomear colunas
# ------------------------------------------------------------------------------
names(dados_raw) <- c(
  "ID",
  "LIMIT_BAL",
  "SEX",
  "EDUCATION",
  "MARRIAGE",
  "AGE",
  "PAY_0",
  "PAY_2",
  "PAY_3",
  "PAY_4",
  "PAY_5",
  "PAY_6",
  "BILL_AMT1",
  "BILL_AMT2",
  "BILL_AMT3",
  "BILL_AMT4",
  "BILL_AMT5",
  "BILL_AMT6",
  "PAY_AMT1",
  "PAY_AMT2",
  "PAY_AMT3",
  "PAY_AMT4",
  "PAY_AMT5",
  "PAY_AMT6",
  "Class"
)

# ------------------------------------------------------------------------------
# BLOCO 4 — Remover ID
# ------------------------------------------------------------------------------
# O ID só identifica a linha.
# Ele não descreve o cliente de forma útil para previsão.
dados <- dados_raw %>%
  select(-ID)

# ------------------------------------------------------------------------------
# BLOCO 5 — Ajustar variável alvo
# ------------------------------------------------------------------------------
# 0 = não deu default
# 1 = deu default 
dados <- dados %>%
  mutate(
    Class = factor(Class,
                   levels = c(1, 0),
                   labels = c("Deve", "Pago")) #como objeitvo é identificar os devedores coloquei primeiro.
  )

# ------------------------------------------------------------------------------
# BLOCO 6 — Checagens básicas
# ------------------------------------------------------------------------------
print(dim(dados)) 
print(table(dados$Class))
print(prop.table(table(dados$Class)))

# Quantidade de valores ausentes por coluna
print(colSums(is.na(dados)))

# Resumo geral
summary(dados)

# ------------------------------------------------------------------------------
# BLOCO 7 — Salvar base limpa
# ------------------------------------------------------------------------------
write_csv(dados, "dados/dados_limpos.csv")
saveRDS(dados, "objetos/dados_limpos.rds")
        
message("01_dados.R concluído com sucesso.")        