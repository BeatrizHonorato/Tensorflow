import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# metragem do apartamento
X = np.array([[30], [45], [60], [75], [90], [105], [120]])

# preço do aluguel em reais
Y = np.array([[900], [1300], [1700], [2000], [2400], [2700], [3100]])

# verificando o grafico
#plt.scatter(X,Y)

# Criando o regressor
regressor = LinearRegression()

# irá realizar o treinamento
regressor.fit(X,Y)

#valor do aluguel
b0 = regressor.intercept_

# declive da linha
b1 = regressor.coef_

# realizando as previsoes de metragem do apartamento
previsao = regressor.predict(X)

# verificando o quanto o scrit errou
mae = mean_absolute_error(Y, previsao)
mse = mean_squared_error(Y, previsao)

#criando grafico
plt.plot(X, Y, "o")
plt.plot(X, previsao, "*",color = "red")
plt.title("Regressão linear simples")
plt.xlabel("Metragem")
plt.ylabel("Aluguel")