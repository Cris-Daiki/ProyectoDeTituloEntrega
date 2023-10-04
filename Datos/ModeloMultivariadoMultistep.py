import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib
import datetime as dt

# Obtener datos de Yahoo Finance
ticker = "BSAC"  
today = dt.date.today()
end_date = today.strftime('%Y-%m-%d')
data = yf.download(ticker, start="2009-10-02", end=end_date)

# Seleccionar los precios de cierre ajustados, el volumen, el precio más alto y el precio más bajo
prices = data['Adj Close'].values
volume = data['Volume'].values
high_prices = data['High'].values
low_prices = data['Low'].values

# Eliminar filas con valores faltantes
prices = prices[~np.isnan(prices)]
volume = volume[~np.isnan(volume)]
high_prices = high_prices[~np.isnan(high_prices)]
low_prices = low_prices[~np.isnan(low_prices)]

# Normalizar los datos de precios, volumen, High y Low
scaler_prices = MinMaxScaler()
prices_scaled = scaler_prices.fit_transform(prices.reshape(-1, 1))

scaler_volume = MinMaxScaler()
volume_scaled = scaler_volume.fit_transform(volume.reshape(-1, 1))

scaler_high = MinMaxScaler()
high_scaled = scaler_high.fit_transform(high_prices.reshape(-1, 1))

scaler_low = MinMaxScaler()
low_scaled = scaler_low.fit_transform(low_prices.reshape(-1, 1))

# Función para crear secuencias de datos con las nuevas características
def create_sequences(data, volume, high, low, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = np.hstack((data[i:i + seq_length], 
                         volume[i:i + seq_length], 
                         high[i:i + seq_length], 
                         low[i:i + seq_length]))
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Longitud de las secuencias
seq_length = 60

# Creación de secuencias y objetivos
sequences, targets = create_sequences(prices_scaled, volume_scaled, high_scaled, low_scaled, seq_length)

# Crear el modelo LSTM
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(seq_length, 4)))  # 4 características
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
model.fit(sequences, targets, epochs=50, batch_size=64, verbose=2)

# Guardar el escalador y el modelo
joblib.dump(scaler_prices, 'scaler_prices.pkl')
joblib.dump(scaler_volume, 'scaler_volume.pkl')
joblib.dump(scaler_high, 'scaler_high.pkl')
joblib.dump(scaler_low, 'scaler_low.pkl')
model.save('lstm_modelMVMSTP.h5')
