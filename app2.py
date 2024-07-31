from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title='Trado Stock Prediction', page_icon = "favicon.png",)

# def main():
Start = '2013-01-01'
End = datetime.now()

st.title('Stock trend prediction')

user_input = st.text_input('Enter Stock Ticker' , 'SBIN.NS')
df = yf.download(user_input, start=Start, end=End)

# Describing Data
st.subheader('Data from 2013 - 2023')
st.write(df.describe())

# visualisation
st.subheader('CLosing Price vs time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Open)
st.pyplot(fig)

st.subheader('CLosing Price vs time chart with 100MA')
ma100 = df.Open.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Open)
st.pyplot(fig)

st.subheader('CLosing Price vs time chart with 100MA & 200MA')
ma100 = df.Open.rolling(100).mean()
ma200 = df.Open.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Open)
st.pyplot(fig)

data_training = pd.DataFrame(df['Open'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Open'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# Loading Model
model = load_model('stock_model.h5')

# testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

    # Scaling Up the predicted result
sf = scaler.scale_
scale_factor = 1/sf
y_predicted = y_predicted * sf
y_test = y_test*sf

    # calculating mean squared error and R square(accuracy) value
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)
print("MSE: ", mse)
print("R-squared: ", r2)

    # final graph
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
