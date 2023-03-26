import streamlit as st
from neuralprophet import NeuralProphet
import yfinance as yf
import pandas as pd
from datetime import date

# Set the page title
st.set_page_config(page_title='Stock Price Prediction', page_icon=':moneybag:')

# Add a title to the app
st.title('Stock Price Prediction')

# Get the stock symbol from the user
stock_symbol = st.text_input('Enter the stock symbol (e.g., AAPL):',value='AAPL')

# Set the default start and end dates
start_date = pd.to_datetime('2017-01-01')
end_date = date.today().strftime("%Y-%m-%d")

# Create a slider to adjust the number of days to predict
num_days = st.slider('Select the number of days to predict:', min_value=1, max_value=365, value=30)

# Download the stock price data from Yahoo Finance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

print(stock_data)
stock_data.info()
print(stock_data.isnull().sum())


# Create a NeuralProphet model
model = NeuralProphet(
    n_changepoints=10,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
)

# Fit the model to the data
df = pd.DataFrame()
df['ds'] = stock_data.index
df['y'] = stock_data['Close'].values
print(df)
model.fit(df, freq='D')

# Make predictions for the future
future = model.make_future_dataframe(df, periods=num_days)
forecast = model.predict(future)

print(forecast.head())
print(forecast.columns)
print(forecast.index)

# Show the forecast data
st.subheader('Forecast Data')
st.write(forecast.tail())


# # Show the forecast data
# st.subheader('Forecast Data')
# st.write(forecast[['ds', 'yhat1']].tail())

# # Plot the predicted prices
# st.subheader('Predicted Stock Prices')
# st.line_chart(forecast[['ds', 'yhat1']].set_index('ds'))



