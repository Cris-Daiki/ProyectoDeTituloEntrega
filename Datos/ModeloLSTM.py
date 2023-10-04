import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
 
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
import joblib
import datetime as dt
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
seq_length = 40

# Creación de secuencias y objetivos
sequences, targets = create_sequences(prices_scaled, seq_length)

# Crear el modelo LSTM
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.add(Dropout(0.2))
model.compile(optimizer='adam', loss='mse',metrics=['mean_absolute_error'])

# Entrenar el modelo
model.fit(sequences, targets, epochs=40, batch_size=64, verbose=2)


from sklearn.metrics import mean_squared_error, mean_absolute_error



joblib.dump(scaler, 'scaler.pkl')  # Guardar el escalador
model.save('lstm_model.keras')  # Guardar el modelo LSTM