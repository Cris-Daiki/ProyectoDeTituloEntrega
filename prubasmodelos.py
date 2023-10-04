import yfinance as yf
import numpy as np
import joblib
from keras.models import load_model
import pandas as pd
# Obtener datos de Yahoo Finance
ticker = "BSAC"  

start_date = "2023-08-15"  # Fecha de inicio
end_date = "2023-09-10"    # Fecha final + días para predecir (10)
data = yf.download(ticker, start=start_date, end=end_date)

# Seleccionar los precios de cierre ajustados
prices = data['Adj Close'].values.reshape(-1, 1)

# Cargar el escalador
scaler = joblib.load('scaler.pkl')

# Normalizar los datos
prices_scaled = scaler.transform(prices)

# Cargar el modelo LSTM
model = load_model('lstm_model.h5')

# Función para crear secuencias de datos
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

# Longitud de las secuencias y creación de secuencias
seq_length = 10  # Puedes ajustar esto según tus necesidades
sequences = create_sequences(prices_scaled, seq_length)

# Realizar predicciones para 'forecast_days' días después de la fecha final proporcionada por el usuario
forecast_days = 10
last_sequence = sequences[-1].reshape(1, -1, 1)
forecast = []

for _ in range(forecast_days):
    predicted_value = model.predict(last_sequence)[0, 0]
    forecast.append(predicted_value)
    last_sequence = np.roll(last_sequence, shift=-1, axis=1)
    last_sequence[0, -1, 0] = predicted_value

# Deshacer la normalización de las predicciones
forecast = np.array(forecast).reshape(-1, 1)
forecast_prices = scaler.inverse_transform(np.vstack((prices[-1:], forecast)))

# Crear un DataFrame de fechas para las predicciones
forecast_index = pd.date_range(start=data.index[-1], periods=forecast_days + 1, closed='right')[1:]

# Imprimir las predicciones
print("Predicciones de precios para los próximos {} días:".format(forecast_days))
for i in range(len(forecast_index)):
    print("{}: {:.2f}".format(forecast_index[i], forecast_prices[i][0]))


