import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

import joblib
np.random.seed(4)

data = yf.download('BSAC', start='2010-01-01', end='2023-09-29')

# aqui procesamos los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# tomamos los 80 datos anteriores y creamos secuancias de entreda y salida 
lookback = 80
X, y = [], []
for i in range(lookback, len(scaled_data) - 5):  # Restar para hacer predicciones a 5 días 
    X.append(scaled_data[i-lookback:i, 0])
    y.append(scaled_data[i+1:i+6, 0])  # Predicción a 5 días 
X, y = np.array(X), np.array(y)

# Dividir los datos en conjuntos de entrenamiento y prueba 80 % para entrenamiento y 20 para validacionc
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(lookback, 1)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(5))  # salida de 5 


model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train, y_train, epochs=50, batch_size=64)


joblib.dump(scaler, 'PruebaLSTMScaler5Dias.pkl')
model.save('PruebaLSTM5Dias.keras')