from flask import Flask, render_template, request, jsonify, redirect
import datetime as dt
import yfinance as yf
import plotly.graph_objs as go
import requests
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import joblib
from datetime import timedelta
app = Flask(__name__)
global fig

@app.route('/')
def principal():
    return render_template('index.html')
fechas_validas = []
@app.route('/Descriptiva', methods=['GET', 'POST'])
def obtenerinformaciongeneral():
    global fechas_validas
    global FechaPrediccionInicial
    global FechaPreddicionFinal
    global fig 
    ticker = "BSAC"
    today = dt.date.today()
    end_date1 = today.strftime('%Y-%m-%d')
    accion = 'BSAC'
    
    if request.method == 'POST':
        
        start_date_str = request.form['start_date']
        end_date_str = request.form['end_date']
        
        # if not start_date_str or not end_date_str:
        #     return "Por favor, ingresa valores válidos para las fechas"
        # if start_date_str > today.strftime('%Y-%m-%d'):
        #     return "por favor, ingresa valores válidos para las fechas" # nota para mi //hacer esto una burbuja o algo asi
        # if start_date_str == end_date_str:
        #     return "por favor, ingresa valores válidos para las fechas"
        # if start_date_str > end_date_str:
        #     return "por favor, ingresa valores válidos para las fechas"
        
        data1 = obtenerData(ticker, start_date_str,end_date_str)
        precio_inicial = data1['Close'][0]
        precio_actual = data1['Close'][-1]
        porcentaje = round((precio_actual - precio_inicial) / precio_inicial * 100, 2)
        
        
        # start_date = dt.datetime.strptime(start_date_str, '%Y-%m-%d')
        # end_date = dt.datetime.strptime(end_date_str, '%Y-%m-%d')
        
        chart_type = request.form['chart_type'] 
        data = obtenerData(accion, start_date_str,end_date_str)
        
        fig = ConstruccionGrafico(data,chart_type)
        fechas_validas.append((start_date_str, end_date_str))
        if 'BorrarIndicadores' in request.form:
            if fechas_validas:
                start_date_str, end_date_str = fechas_validas[-1]
            
            selected_indicators = []
        else:
            selected_indicators = ContruirIndicadores(data,start_date_str,end_date_str)
        #selected_indicators = ContruirIndicadores(fig,data,start_date,end_date)
        
        plot_json = fig.to_json()
        
        return render_template('Descriptiva.html', plot_json=plot_json,start_date_str=start_date_str, end_date_str=end_date_str,chart_type=chart_type,selected_indicators=selected_indicators,porcentaje=porcentaje)
    else:
        start_date_str = '2023-02-01'
        end_date_str = end_date1
        if fechas_validas:
            start_date_str, end_date_str = fechas_validas[-1]
        # FechaPrediccionInicial =start_date_str
        # FechaPreddicionFinal = end_date_str
        data2 = obtenerData(ticker, start_date_str,end_date_str)
        
        precio_inicial = data2['Close'][0]
        precio_actual = data2['Close'][-1]
        porcentaje = round((precio_actual - precio_inicial) / precio_inicial * 100, 2)

        default_chart_type = 'candlestick'

        fig =  ConstruccionGrafico(data2,default_chart_type)
        fig.update_layout(font=dict(color='white'))
        plot_json = fig.to_json()
        return render_template('Descriptiva.html', plot_json=plot_json, default_chart_type=default_chart_type,start_date_str=start_date_str, end_date_str=end_date_str,porcentaje=porcentaje)




def obtenerData(accion, start_date, end_date):
    # if  isinstance(end_date, str):
    #     return "end_date_str no es una cadena"
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    end_date_nuevo = end_date + timedelta(days=1)
    data =yf.download(accion, start_date, end_date_nuevo)
    return data



def ConstruccionGrafico(data,chart_type):
    
    global fig
    
    if chart_type == 'candlestick':
        
        fig = go.Figure(data=go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        ))
        title = 'Gráfico velas japonesas'
    elif chart_type == 'line':
        fig = go.Figure(data=go.Scatter(
            x=data.index,
            y=data['Close']
        ))
        title = 'Gráfico Lineas'
    elif chart_type == 'barra':
        fig = go.Figure(data=go.Bar(
            x=data.index,
            y=data['Close']
        ))
        title = 'Gráfico de Barra'

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        #width=650,
        #height=500
    )
    fig.update_layout(font=dict(color='white'))
    
    return fig

def ContruirIndicadores(data,start_date_indi,end_date_indi):
    global fig 
    print(start_date_indi,end_date_indi)
    indicators = request.form.getlist('indicators')
    indicadoresCalculados = request.form.getlist('indicators')
    if 'sma' in indicators:
        sma = data['Close'].loc[start_date_indi:end_date_indi].rolling(window=20).mean()
        fig.add_trace(go.Scatter(x=data.index, y=sma, name='SMA'))
    if 'ma' in indicators:
        ma = data['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(x=data.index, y=ma, name='MA'))

    if 'ema' in indicators:
        ema = data['Close'].ewm(span=20, adjust=False).mean()
        fig.add_trace(go.Scatter(x=data.index, y=ema, name='EMA'))

    if 'macd' in indicators:
        macd = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
        signal = macd.ewm(span=9, adjust=False).mean()
        fig.add_trace(go.Scatter(x=data.index[26:], y=macd[26:], name='MACD'))
        fig.add_trace(go.Scatter(x=data.index[26:], y=signal[26:], name='Signal'))
    if 'bb' in indicators:
        window = 20  #periodos
        std_multiplier = 2  # Multiplicador para el cálculo de las desviaciones estándar
        sma = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        upper_band = sma + (std * std_multiplier)
        lower_band = sma - (std * std_multiplier)
        fig.add_trace(go.Scatter(x=data.index, y=upper_band, name='BBUBB'))
        #fig.add_trace(go.Scatter(x=data.index, y=sma, name='SMA'))
        fig.add_trace(go.Scatter(x=data.index, y=lower_band, name='BBLBB'))

    if 'stddev' in indicators:
        std_dev = data['Close'].rolling(window=20).std()            # Calcular la Desviación Estándar
        upper_band = data['Close'].rolling(window=20).mean() + 2 * std_dev
        lower_band = data['Close'].rolling(window=20).mean() - 2 * std_dev

        fig.add_trace(go.Scatter(x=data.index, y=upper_band, name='STDDEVUB', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=data.index, y=lower_band, name='STDDEVLB', line=dict(color='orange')))
    # if forecast is not None:
    #     fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicción', line=dict(color='purple')))
    #fig.update_layout(
        #width = 650,
        #height=500
    #)
    return indicadoresCalculados


@app.route('/convertir', methods=['GET', 'POST'])
def convertir():
    if request.method == 'POST':
        today = dt.date.today()
        end_date = today.strftime('%Y-%m-%d')
        conversion_pair = ''
        cantidad = float(request.form['cantidad'])
        moneda_origen = request.form['moneda_origen']
        moneda_destino = request.form['moneda_destino']
        #obtenerdatos
        if(moneda_origen == moneda_destino):
            return f'Conversión de {cantidad:.2f} {moneda_origen} a {moneda_destino}: {cantidad:.2f}'
        if moneda_origen == 'USD' and moneda_destino == 'CLP':
            data = yf.download('CLP=X', start='2023-06-01', end=end_date)
            conversion_pair = 'CLP=X'
        elif moneda_origen == 'CLP' and moneda_destino == 'USD':
            data = yf.download('USDCLP=X', start='2023-06-01', end=end_date)
            conversion_pair = 'USDCLP=X'
        elif moneda_origen == 'USD' and moneda_destino == 'EUR':
            
            data = yf.download('EURUSD=X', start='2023-06-01', end=end_date)
            conversion_pair = 'USDEUR=X'
            print(data)
        elif moneda_origen == 'EUR' and moneda_destino == 'USD':
            data = yf.download('USDEUR=X', start='2023-06-01', end=end_date)
            conversion_pair = 'USDEUR=X'
        elif moneda_origen == 'CLP' and moneda_destino == 'EUR':
            data = yf.download('EURCLP=X', start='2023-06-01', end=end_date)
            conversion_pair = 'EURCLP=X'
        elif moneda_origen == 'EUR' and moneda_destino == 'CLP':
            data = yf.download('CLPEUR=X', start='2023-06-01', end=end_date)
            conversion_pair = 'CLPEUR=X'
        else:
            return 'No se encontraron datos disponibles para la combinación de monedas seleccionada.'
        
        if conversion_pair != '':
            if conversion_pair == 'CLPEUR=X' or conversion_pair=='CLP=X':
                if not data.empty and len(data) > 0:
                    price = data['Close'].iloc[-1]
                    conversion = price*cantidad
                    return f'Conversión de {cantidad:.2f} {moneda_origen} a {moneda_destino}: {conversion:.2f}'
                else:
                    return 'No se encontraron datos disponibles para la combinación de monedas seleccionada.'
                
            else:
                if not data.empty and len(data) > 0:
                    price = data['Close'].iloc[-1]
                    conversion = cantidad/price
                    return f'Conversión de {cantidad:.2f} {moneda_origen} a {moneda_destino}: {conversion:.2f}'
                else:
                    return 'No se encontraron datos disponibles para la combinación de monedas seleccionada.'
        else:
            return 'No se logro hacer la conversion.'

def obtener_informacion_indicadores(indicators):
    info = ""
    if 'sma' in indicators:
        info += "Información del SMA\n"
    
    if 'ma' in indicators:
        info += "Información del MA\n"
    
    if 'ema' in indicators:
        info += "Información del EMA\n"
    if 'macd' in indicators:
        info += "Información del macd\n"
    if 'bb' in indicators:
        info += "Información del bb\n"
    if 'stddev' in indicators:
        info += "Información del stddev\n"
    
    
    return info


@app.route('/buscar-noticias', methods=['GET', 'POST'])
def buscar_noticias():
    if request.method == 'POST':
        # Obtener los parámetros de búsqueda enviados desde el formulario
        keyword = request.form['keyword']
        language = request.form['language']

    else:
       keyword ='Banco Santander'
       language ='es'
     # Hacer una solicitud a la API de NewsAPI para obtener las noticias
    api_key = '13bc4f6b8fb546c88fa3908ab27d498a'
    url = f'https://newsapi.org/v2/everything?q={keyword}&language={language}&apiKey={api_key}'
    response = requests.get(url)
    
    if response.status_code == 200:
        # Procesar la respuesta de la API y extraer los datos relevantes
        noticias = response.json()['articles']
        return render_template('Noticias.html', noticias=noticias,keyword=keyword,language=language)
    else:
        return 'Error al obtener las noticias'

def Procesesar_datos_usuario(symbol,inicio,fin):

    ticker_symbol = symbol
    start_date = inicio

    end_date = fin

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
    return df2


import matplotlib.pyplot as plt

def graficar_predicciones(real, prediccion,fechas):
    x = np.arange(len(fechas))
    
    plt.plot(x, real, color='red', label='Valor real de la acción')
    plt.plot(x, prediccion, color='blue', label='Predicción de la acción')
    plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
    
    # Personalizar las etiquetas del eje x con las fechas
    plt.xticks(x, fechas, rotation=45)
    
    plt.xlabel('Fecha')
    plt.ylabel('Valor de la acción')
    plt.legend()
    plt.show()

import plotly.express as px
import pandas as pd

from datetime import datetime, timedelta
def get_next_weekday(date):
    next_day = date + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5 y 6 representan sábado y domingo
        next_day += timedelta(days=1)
    return next_day
def find_next_weekday(date):
    while date.weekday() >= 5:  
        date += timedelta(days=1)
    return date
@app.route('/Predecir_accion', methods=['POST'])
def predecir_accion():
    cantidad_dias = int(request.form['cantidad_dias'])  # Obtener la opción seleccionada (1 o 5 días)
    start_date = request.form['start_date']  
    end_date_str = request.form['end_date']
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    end_date_nuevo_1 = end_date + timedelta(days=1)

    print("Primerendate",end_date_nuevo_1)
    global fig
    
    end_date_nuevo = find_next_weekday(end_date)
    print("Segundoendate",end_date_nuevo)
    start_date_str = datetime.strptime(start_date, '%Y-%m-%d')
    date_difference = end_date - start_date_str
    if date_difference < timedelta(days=90):
        start_date_str = start_date_str - timedelta(days=90)

    new_data = yf.download('BSAC', start=start_date_str, end=end_date_nuevo_1, interval='1d')
    New_data_2 = yf.download('BSAC', start=start_date, end=end_date_nuevo, interval='1d')




    if cantidad_dias == 1:
        model = load_model('PruebaLSTM.keras')
        scaler = joblib.load('PruebaLSTMScaler.pkl')
        new_scaled_data = scaler.transform(new_data['Close'].values.reshape(-1, 1))
        lookback = 90  
        X_new = []
        X_new.append(new_scaled_data[-lookback:, 0])
        X_new = np.array(X_new)
        predicted_value = model.predict(X_new)
        predicted_value = scaler.inverse_transform(predicted_value)
        last_date = new_data.index[-1]
        # if end_date_nuevo.weekday() == 5:
        #     next_date = end_date_nuevo + pd.DateOffset(days=1)
        # elif end_date_nuevo.weekday() == 6:
        #     next_date = end_date_nuevo + pd.DateOffset(days=0)
        # elif end_date_nuevo.weekday() == 4:
        #     next_date = end_date_nuevo + pd.DateOffset(days=2)
        # else:
        #     next_date = end_date_nuevo + pd.DateOffset(days=0)
        # next_date = last_date + pd.DateOffset(days=1)
        next_date = find_next_weekday(end_date + timedelta(days=1))
        predicted_df = pd.DataFrame({'Date': [next_date], 'Predicted': predicted_value[:, 0]})
        predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])
        fig = GraficarPredicciones1dia(new_data, predicted_df)
        plot_json = fig.to_json()
        return jsonify({'plot_json': plot_json})
    if cantidad_dias == 5:
        model = load_model('PruebaLSTM5Dias.keras')
        scaler = joblib.load('PruebaLSTMScaler5Dias.pkl')
        scaled_data = scaler.transform(new_data['Close'].values.reshape(-1, 1))
        lookback = 80
        X_input = scaled_data[-lookback:].reshape(1, lookback, 1)
        predictions = model.predict(X_input)
        predictions = scaler.inverse_transform(predictions)
        start_date_str = find_next_weekday(end_date + timedelta(days=1))
    
        # Generar las fechas de pronóstico para los próximos 5 días hábiles
        forecast_dates = [start_date_str]
        for _ in range(4):
            next_day = get_next_weekday(forecast_dates[-1])
            forecast_dates.append(next_day)
        
        print(forecast_dates)
        fig = GraficarPredicciones5dias(new_data, forecast_dates, predictions)

        plot_json = fig.to_json()

        return jsonify({'plot_json': plot_json})

def GraficarPredicciones5dias(new_data, forecast_dates,predictions):
    global fig
    combined_dates = list(new_data.index) + list(forecast_dates)
    combined_values = list(new_data['Close']) + list(predictions[0])
    fig.add_trace(go.Scatter(x=combined_dates, y=combined_values, mode='lines', name='Precio y predicción', line=dict(color='black')))
    return fig
def GraficarPredicciones1dia(new_data, predicted_df):
    global fig
    combined_dates = list(new_data.index) + list(predicted_df['Date'])
    combined_values = list(new_data['Close']) + list(predicted_df['Predicted'])
    fig.add_trace(go.Scatter(x=combined_dates, y=combined_values, mode='lines', name='Precio y predicción', line=dict(color='black')))
    return fig

if __name__ == '__main__':
    app.run(debug=True)
