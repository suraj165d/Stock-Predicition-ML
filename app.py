import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



model = load_model(r'C:\Python\Stock\Stock Predictions Model.keras')

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symnbol', 'AAPL')
start = '2013-01-01'
end = '2023-12-31'

data = yf.download(stock, start, end)

st.subheader('Stock Data From 2013 to 2023')
st.write(data.describe())

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))

# Check if data_test is not empty before scaling
if not data_test.empty:
    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)


    st.subheader('Price vs MA50')
    ma_50_days = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, 'r', label = 'Predicted Price')
    plt.plot(data.Close, 'g', label = 'Original Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.pyplot(fig1)
    
    
    #OLD CODE
    # st.subheader('Price vs MA50')
    # ma_50_days = data.Close.rolling(50).mean()
    # fig1 = plt.figure(figsize=(8, 6))
    # plt.plot(ma_50_days, 'r')
    # plt.plot(data.Close, 'g')
    # plt.show()
    # st.pyplot(fig1)

    # ... (similarly update other plots)

    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i, 0])

    x, y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale

    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8, 6))
    plt.plot(predict, 'r', label='Original Price')
    plt.plot(y, 'g', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.pyplot(fig4)
    

    
else:
    st.warning("Enter Valid Stock Ticker")
    st.warning(
    "If you are facing issues while searching for your stock ticker, you can try these methods:\n"
    "1. ✅ Double-check your spelling.\n"
    "2. ✅ Enter the appropriate stock ticker.\n"
    "3. ✅ It's possible that your requested stock ticker is no longer in the database."
)

