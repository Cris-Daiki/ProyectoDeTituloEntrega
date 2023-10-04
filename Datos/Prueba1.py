import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
 
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
import joblib
import datetime as dt
import pandas as pd
# Obtener datos de Yahoo Finance
ticker = "BSAC"  
today = dt.date.today()
end_date = today.strftime('%Y-%m-%d')
data = yf.download(ticker, start="2005-05-01", end=end_date)

# Seleccionar los precios de cierre ajustados
prices = data['Close'].values

# Eliminar filas con valores faltantes
prices = prices[~np.isnan(prices)]

# Normalizar los datos
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

# Función para crear secuencias de datos
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Longitud de las secuencias
seq_length = 20

# Creación de secuencias y objetivos
sequences, targets = create_sequences(prices_scaled, seq_length)

# Crear el modelo LSTM
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.add(Dropout(0.2))
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
model.fit(sequences, targets, epochs=40, batch_size=64, verbose=2)# Realizar predicciones con enfoque de "rolling forecast"
forecast_days = 3
forecast = []

for _ in range(forecast_days):
    last_sequence = prices_scaled[-seq_length:]  # Utilizar las últimas seq_length observaciones
    predicted_value = model.predict(last_sequence.reshape(1, seq_length, 1))[0, 0]
    forecast.append(predicted_value)
    prices_scaled = np.append(prices_scaled, predicted_value)  # Agregar la predicción a los datos

# Deshacer la normalización de las predicciones
forecast = np.array(forecast).reshape(-1, 1)
forecast_prices = scaler.inverse_transform(np.vstack((prices[-1:], forecast)))

# Crear un DataFrame de fechas para las predicciones
forecast_index = pd.date_range(start=data.index[-1], periods=forecast_days, closed='right')[1:]
forecast_prices = forecast_prices[1:]
forecast_index = forecast_index[1:]

# Imprimir las predicciones
print("Predicciones de precios para los próximos {} días:".format(forecast_days))
for i in range(len(forecast_index)):
    print("{}: {:.2f}".format(forecast_index[i], forecast_prices[i][0]))
joblib.dump(scaler, 'sdasdasd.pkl')  # Guardar el escalador
model.save('Prueba1.keras')  # Guardar el modelo LSTM