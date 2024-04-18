# Instalar e carregar pacotes necessários
if (!require("neuralnet")) install.packages("neuralnet")
if (!require("readxl")) install.packages("readxl")
library(neuralnet)
library(readxl)

# Especificar o caminho do arquivo Excel
# Substitua o texto abaixo pelo caminho completo do seu arquivo Excel
caminho_arquivo <- "C:/TCC_MBA/previsao_ssa.xlsx"

# Carregar dados do arquivo Excel
caminho_arquivo <- "C:/TCC_MBA/previsao_ssa.xlsx"
dados <- read_excel(caminho_arquivo, sheet = "#3_6")

colnames(dados)

# Normalização dos dados
colunas_norm <- c("C", "H", "N", "O","VM", "Ash", "FC", "Cel", "Hem", "Lig", "Ext", "T", "RT", "HR")
dados_norm <- dados
dados_norm[colunas_norm] <- scale(dados[colunas_norm])

# Divisão em conjuntos de treinamento e teste
set.seed(123)  # Para reprodutibilidade
indices <- sample(1:nrow(dados_norm), size = 0.8 * nrow(dados_norm))  # 80% para treinamento
treino <- dados_norm[indices, ]
teste <- dados_norm[-indices, ]

# Treinamento da rede neural com ajustes
rede <- neuralnet(SSA ~ C + H + N + O + VM + Ash + FC + Cel + Hem + Lig + Ext + T + RT + HR,
                  data = treino,
                  hidden = c(12, 6),
                  learningrate = 0.01,
                  stepmax = 1e+06,
                  algorithm = "rprop+",
                  linear.output = TRUE,
                  threshold = 0.01)

# Fazendo previsões no conjunto de treino e teste
previsoes_treino <- compute(rede, treino[, colnames(treino) != "SSA"])$net.result
previsoes_teste <- compute(rede, teste[, colnames(teste) != "SSA"])$net.result

# Calculando RMSE para o conjunto de treino
RMSE_treino <- sqrt(mean((previsoes_treino - treino$SSA)^2))
print(paste("Root Mean Square Error (RMSE) - Treino:", RMSE_treino))

# Calculando R² para o conjunto de treino
R2_treino <- cor(previsoes_treino, treino$SSA)^2
print(paste("R² (Treino):", R2_treino))

# Calculando R² e RMSE para o conjunto de teste
R2_teste <- cor(previsoes_teste, teste$SSA)^2
print(paste("R² (Teste):", R2_teste))

MSE_teste <- mean((previsoes_teste - teste$SSA)^2)
RMSE_teste <- sqrt(MSE_teste)
print(paste("Erro Quadrático Médio (MSE) - Teste:", MSE_teste))
print(paste("Root Mean Square Error (RMSE) - Teste:", RMSE_teste))

# Calculando o Coeficiente de Correlação de Pearson para o conjunto de treino
coeficiente_pearson_treino <- cor(previsoes_treino, treino$SSA)
print(paste("Coeficiente de Correlação de Pearson (Treino):", coeficiente_pearson_treino))

# Calculando o Coeficiente de Correlação de Pearson para o conjunto de teste
coeficiente_pearson_teste <- cor(previsoes_teste, teste$SSA)
print(paste("Coeficiente de Correlação de Pearson (Teste):", coeficiente_pearson_teste))

