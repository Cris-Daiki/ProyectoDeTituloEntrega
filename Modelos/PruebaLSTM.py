import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import joblib
import datetime as dt
np.random.seed(4)
# Obtener datos históricos de 
today = dt.date.today()
end_date = today.strftime('%Y-%m-%d')
data = yf.download('BSAC', start='2005-01-01', end = end_date)

# aqui procesamos los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# tomamos los datos anteriores y creamos secuancias de entreda y salida 
lookback = 90
X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i-lookback:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)

# Dividir los datos en conjuntos de entrenamiento y prueba 80 % para entrenamiento y 20 para validacionc
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Crear y entrenar la red LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(lookback, 1)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compilar el modelo con un optimizador y función de pérdida
model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train, y_train, epochs=100, batch_size=64)

# Hacer predicciones en el conjunto de prueba
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calcular el Error Cuadrático Medio (ECM)
mse = mean_squared_error(y_test_inverse, predictions)

# Calcular la Raíz del Error Cuadrático Medio (RMSE)
rmse = np.sqrt(mse)

print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse}")
r2 = r2_score(y_test_inverse, predictions)

print(f"Coeficiente de Determinación (R²): {r2}")
joblib.dump(scaler, 'PruebaLSTMScaler.pkl')  # Guardar el escalador
model.save('PruebaLSTM.keras')  # Guardar el modelo LSTM