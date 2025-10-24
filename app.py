import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime
import os

def get_currency(symbol):
    symbol = symbol.upper()
    if symbol.endswith(".NS") or symbol.endswith(".BSE") or symbol.endswith(".BO"):
        return "₹"  # Indian Rupee
    else:
        return "$"  # US Dollar


st.set_page_config(page_title="Stock Market Predictor", layout="wide")
st.title("Stock Price Predictor")

# --- MODEL LOADING ---
model_file_h5 = "Stock_Prediction_new.h5"
model_file_keras = "Stock_Prediction_new.keras"

model = None
if os.path.exists(model_file_h5):
    model = load_model(model_file_h5, compile=False)
    st.success(f"✅ Loaded model: {model_file_h5}")
elif os.path.exists(model_file_keras):
    model = load_model(model_file_keras, compile=False)
    st.success(f"✅ Loaded model: {model_file_keras}")
else:
    st.error("❌ Model file not found! Make sure .h5 or .keras file is in the repo folder.")

# --- STOCK INPUT ---
stock = st.text_input("Enter Stock Symbol", "GOOG")
start = st.date_input("Start Date", datetime.date(2015, 1, 1))
end = st.date_input("End Date", datetime.date.today())

# --- CACHE YFINANCE DOWNLOAD ---
@st.cache_data(show_spinner=True)
def get_stock_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

# --- BUTTON ACTION ---
if st.button("Fetch & Predict"):
    if not stock:
        st.warning("Please enter a stock symbol.")
    elif model is None:
        st.warning("Model is not loaded.")
    elif (end - start).days > 3650:  # limit ~10 years
        st.warning("Please select a date range shorter than 10 years.")
    else:
        with st.spinner("Fetching stock data..."):
            df = get_stock_data(stock, start, end)
            if df.empty:
                st.error("No data found for this stock symbol.")
            else:
                st.subheader("Stock Data (Last 5 rows)")
                st.write(df.tail())

                # --- DATA PREPARATION ---
                data_train = pd.DataFrame(df.Close[0:int(len(df)*0.8)])
                data_test = pd.DataFrame(df.Close[int(len(df)*0.8):])
                # --- CORRECTED SCALING ---
                scaler = MinMaxScaler(feature_range=(0, 1))

                # Fit scaler ONLY on training data
                scaled_train = scaler.fit_transform(data_train)

                # Prepare test data (use same scaler)
                past_100_days = data_train.tail(100)
                data_test = pd.concat([past_100_days, data_test], ignore_index=True)
                data_test_scaled = scaler.transform(data_test)

                # --- MOVING AVERAGES PLOTS ---
                ma50 = df.Close.rolling(50).mean()
                ma100 = df.Close.rolling(100).mean()
                ma200 = df.Close.rolling(200).mean()

                st.subheader("Price vs MA50")
                fig1, ax1 = plt.subplots(figsize=(10,6))
                ax1.plot(ma50, 'r', label='MA50')
                ax1.plot(df.Close, 'g', label='Close Price')
                ax1.legend()
                st.pyplot(fig1)

                st.subheader("Price vs MA50 vs MA100")
                fig2, ax2 = plt.subplots(figsize=(10,6))
                ax2.plot(ma50, 'r', label='MA50')
                ax2.plot(ma100, 'b', label='MA100')
                ax2.plot(df.Close, 'g', label='Close Price')
                ax2.legend()
                st.pyplot(fig2)

                st.subheader("Price vs MA100 vs MA200")
                fig3, ax3 = plt.subplots(figsize=(10,6))
                ax3.plot(ma100, 'r', label='MA100')
                ax3.plot(ma200, 'b', label='MA200')
                ax3.plot(df.Close, 'g', label='Close Price')
                ax3.legend()
                st.pyplot(fig3)

                # --- PREDICTION ---
                x_test, y_test = [], []
                for i in range(100, data_test_scaled.shape[0]):
                    x_test.append(data_test_scaled[i-100:i])
                    y_test.append(data_test_scaled[i,0])
                x_test, y_test = np.array(x_test), np.array(y_test)

                predict = model.predict(x_test)
                predict = scaler.inverse_transform(predict)
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

                st.subheader("Original vs Predicted Prices")
                fig4, ax4 = plt.subplots(figsize=(10,6))
                # --- USE REAL DATES ON X-AXIS ---
                test_dates = df.index[int(len(df)*0.8):]  # Dates for test period

                fig4, ax4 = plt.subplots(figsize=(10,6))
                ax4.plot(test_dates, y_test, 'g', label="Original Price")
                ax4.plot(test_dates, predict, 'r', label="Predicted Price")
                ax4.set_xlabel("Date")
                currency_symbol = get_currency(stock)
                ax4.set_ylabel(f"Price ({currency_symbol})")

                ax4.legend()
                fig4.autofmt_xdate()
                st.pyplot(fig4)

