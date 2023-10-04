import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go
import joblib
# Obtener los datos históricos de APPL utilizando yfinance
data = yf.download('AAPL', start="2005-05-01", end="2023-09-28")['Close']

# Preparar los datos
df = data.reset_index()
df=df[~np.isnan(data)] #eliminar filas que le falten valores 

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
n_steps = 10

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

# Realizar las predicciones
predictions = model.predict(X_test)

# Desnormalizar las predicciones y los valores reales
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# Obtener las predicciones para los próximos 10 días a partir del 29 de septiembre de 2023
last_sequence = scaled_data[-n_steps:]
next_sequence = []
for _ in range(10):
    prediction = model.predict(last_sequence.reshape(1, n_steps, 1))
    next_sequence.append(prediction)
    last_sequence = np.append(last_sequence[1:], prediction, axis=0)

# Desnormalizar las predicciones para los próximos 10 días
next_sequence = np.array(next_sequence).reshape(10, 1)
next_sequence = scaler.inverse_transform(next_sequence)
print(next_sequence)
# Crear el gráfico utilizando Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['y'], name='Datos reales', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df.index[train_size:], y=predictions.reshape(-1), name='Predicciones', line=dict(color='red' if predictions[-1]<df['y'].iloc[train_size-1] else 'green')))

# Configurar el diseño del gráfico
fig.update_layout(title='Predicciones de APPL para los próximos 10 días',
                  xaxis_title='Fecha',
                  yaxis_title='Precio de cierre')

# Mostrar el gráfico
fig.show()
