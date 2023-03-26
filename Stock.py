
import yfinance as yf
from neuralprophet import NeuralProphet
import streamlit as st
import pandas as pd
from datetime import date

st.title('Stock Price Prediction')
st.write('Enter the stock symbol and the number of days to predict')

symbol = st.text_input('Stock Symbol', value='AAPL')
n_days = st.slider('Days to Predict', min_value=1, max_value=365, value=30, step=1)

df=yf.download(symbol, "2018-01-01", date.today().strftime("%Y-%m-%d"))
#df.reset_index(inplace=True)

df = df.dropna()

df.info()


# dataf= df.rename(columns={"Date": "ds", "Close": "y"})
# print(dataf.head())

# st.subheader('Raw data')
# st.write(dataf.head())

# dataf = dataf[['ds','y']]
# print(dataf.head())


# dataf= df.rename(columns={"Date": "ds", "Close": "y"})
# print(dataf)

# df.rename(columns = {'Date':'ds', 'Close':'y'}, inplace = True)
# df.head()

model = NeuralProphet(
    n_changepoints=10,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
)
daf = pd.DataFrame()
daf['ds'] = df.index
daf['y'] = df['Close']
model.fit(daf, freq='D')

future = model.make_future_dataframe(daf,periods=n_days)
# forecast = model.predict(future)

# st.write(forecast)

# # print(forecast[['ds', 'yhat']])
# st.write('Predicted stock prices:')
# st.line_chart(forecast[['ds']])

# #https://blog.finxter.com/python-streamlit-i-made-this-stock-price-prediction-app/