import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error
import joblib

# Obtener datos históricos de 
data = yf.download('BSAC', start='2010-01-01', end='2023-09-29')

# Preprocesar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Crear secuencias de datos de entrada y salida
lookback = 80
X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i-lookback:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)

# Dividir los datos en conjuntos de entrenamiento y prueba
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Crear y entrenar la red LSTM
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(lookback, 1)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compilar el modelo con un optimizador y función de pérdida
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo con un mayor número de épocas y un tamaño de lote adecuado
model.fit(X_train, y_train, epochs=50, batch_size=64)

# Hacer predicciones en el conjunto de prueba
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
joblib.dump(scaler, 'PruebaLSTMScaler.pkl')  # Guardar el escalador
model.save('PruebaLSTM.keras')  # Guardar el modelo LSTM