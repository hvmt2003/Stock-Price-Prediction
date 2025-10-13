import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime

st.set_page_config(page_title="Stock Market Predictor", layout="wide")
st.title("📈 Stock Market Predictor")

# Load model (make sure model file is in your repo)
try:
    model = load_model("Stock_Prediction_new.keras")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f" Failed to load model: {e}")
    model = None


# Stock symbol input
stock = st.text_input("Enter Stock Symbol", "GOOG")

# Date input
start = st.date_input("Start Date", datetime.date(2015, 1, 1))
end = st.date_input("End Date", datetime.date.today())

# Button to fetch and process data
if st.button("Fetch & Predict"):
    if stock and model:
        with st.spinner("Fetching stock data..."):
            df = yf.download(stock, start=start, end=end)
            if df.empty:
                st.error("No data found for this stock symbol.")
            else:
                st.subheader("Stock Data (Last 5 rows)")
                st.write(df.tail())

                # Data preparation
                data_train = pd.DataFrame(df.Close[0:int(len(df)*0.8)])
                data_test = pd.DataFrame(df.Close[int(len(df)*0.8):])

                scaler = MinMaxScaler(feature_range=(0,1))
                past_100_days = data_train.tail(100)
                data_test = pd.concat([past_100_days, data_test], ignore_index=True)
                data_test_scaled = scaler.fit_transform(data_test)

                # Moving averages plots
                st.subheader("Price vs MA50")
                ma50 = df.Close.rolling(50).mean()
                fig1, ax1 = plt.subplots(figsize=(10,6))
                ax1.plot(ma50, 'r', label='MA50')
                ax1.plot(df.Close, 'g', label='Close Price')
                ax1.legend()
                st.pyplot(fig1)

                st.subheader("Price vs MA50 vs MA100")
                ma100 = df.Close.rolling(100).mean()
                fig2, ax2 = plt.subplots(figsize=(10,6))
                ax2.plot(ma50, 'r', label='MA50')
                ax2.plot(ma100, 'b', label='MA100')
                ax2.plot(df.Close, 'g', label='Close Price')
                ax2.legend()
                st.pyplot(fig2)

                st.subheader("Price vs MA100 vs MA200")
                ma200 = df.Close.rolling(200).mean()
                fig3, ax3 = plt.subplots(figsize=(10,6))
                ax3.plot(ma100, 'r', label='MA100')
                ax3.plot(ma200, 'b', label='MA200')
                ax3.plot(df.Close, 'g', label='Close Price')
                ax3.legend()
                st.pyplot(fig3)

                # Prepare data for prediction
                x_test, y_test = [], []
                for i in range(100, data_test_scaled.shape[0]):
                    x_test.append(data_test_scaled[i-100:i])
                    y_test.append(data_test_scaled[i,0])
                x_test, y_test = np.array(x_test), np.array(y_test)

                # Prediction
                predict = model.predict(x_test)
                scale = 1 / scaler.scale_[0]
                predict = predict * scale
                y_test = y_test * scale

                st.subheader("Original vs Predicted Prices")
                fig4, ax4 = plt.subplots(figsize=(10,6))
                ax4.plot(y_test, 'g', label="Original Price")
                ax4.plot(predict, 'r', label="Predicted Price")
                ax4.set_xlabel("Time")
                ax4.set_ylabel("Price")
                ax4.legend()
                st.pyplot(fig4)
    else:
        st.warning("Please enter a valid stock symbol and ensure the model is loaded.")
