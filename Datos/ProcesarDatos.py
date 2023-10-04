import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import yfinance as yf
import datetime as dt

ticker_symbol = 'BSAC' 
start_date = '2000-01-01'
today = dt.date.today()
end_date = '2023-09-22'

# dataset = yf.download(ticker_symbol, start=start_date, end=end_date)

dataset = yf.download(ticker_symbol, start_date, end_date)

print(dataset)
print('cantidad de Nans:')
for column in dataset: #se ven datos que esten incompletos que pueden afectar la prediccion del modelo
    nans = dataset[column].isna().sum()#en este caso no hay datos que esten incompletos 
    print(f'tcolumna {column}:{nans}')
df_time_diffs = dataset.index.to_series().diff().dt.total_seconds() #al igual que los datos, se necesita que el tiempo entre datos sean los mismos, como estoy utilizando series de tiempo de un dia necesito que todos esten con un dia de distancia 
print(df_time_diffs.value_counts())
#asi que tengo que corregir los registros que sean distintos a un dia 
#dataset.drop_duplicates(keep='first',inplace=True, ignore_index=False)#que nos deje el primer dato duplicado y que elimine los repetidos

df2 =dataset.asfreq(freq='D',method='bfill')#reinterpolar el dataset para 1 dia 
#cuando hay datos que no hay en el rango de un dia, se llena con el metodo bfill que toma el dato anterior y lo rellena 
df_time_diffs=df2.index.to_series().diff().dt.total_seconds()

print(df_time_diffs.value_counts())
print(df2)
df2.to_csv('Datos/dataset_procesado.csv')
