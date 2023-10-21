import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error, r2_score

# Descargar los datos
np.random.seed(4)
data = yf.download('BSAC', start='2010-01-01', end='2023-09-29')

# Preprocesar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Crear secuencias de entrada y salida
lookback = 90
horizon = 5 
X, y = [], []

for i in range(lookback, len(scaled_data) - horizon):
    X.append(scaled_data[i-lookback:i, 0])
    y.append(scaled_data[i:i+horizon, 0])

X, y = np.array(X), np.array(y)

# Dividir los datos en conjuntos de entrenamiento y prueba
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Construir el modelo LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(lookback, 1)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128)) 
model.add(Dropout(0.2))
model.add(Dense(horizon)) 
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=64)

# Hacer predicciones con el modelo
predictions = model.predict(X_test)

# Invertir la transformación de escala en las predicciones y los valores reales
predictions = scaler.inverse_transform(predictions)
y_test_inverse = scaler.inverse_transform(y_test)

# Calcular el ECM
mse = mean_squared_error(y_test_inverse, predictions)

# Calcular el RMSE
rmse = np.sqrt(mse)

# Calcular el coeficiente de determinación (R²)
r2 = r2_score(y_test_inverse, predictions)

# Imprimir los resultados
print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse}")
print(f"Coeficiente de Determinación (R²): {r2}")

