import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go

# Obtener los datos históricos de APPL utilizando yfinance
data = yf.download('BSAC', start="2005-05-01", end="2023-09-28")['Close']

# Preparar los datos
df = data.reset_index()
df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
df['ds'] = pd.to_datetime(df['ds'])
df.sort_values('ds', inplace=True)
df.set_index('ds', inplace=True)

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Crear los conjuntos de entrenamiento y prueba
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Función para crear los datos de entrada y salida para el modelo LSTM
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

# Definir el número de pasos de tiempo para las secuencias
n_steps = 5

# Crear las secuencias de entrenamiento y prueba
X_train, y_train = create_sequences(train_data, n_steps)
X_test, y_test = create_sequences(test_data, n_steps)

# Crear el modelo LSTM
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=2)
import joblib
joblib.dump(scaler, 'scalerprueba3.pkl')  # Guardar el escalador
model.save('lstm_modelprueba3.keras')  # Guardar el modelo LSTM