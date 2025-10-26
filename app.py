import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime
import os

# --- FUNCTION TO GET CURRENCY SYMBOL BASED ON STOCK EXCHANGE ---
def get_currency(symbol):
    symbol = symbol.upper()
    if symbol.endswith(".NS") or symbol.endswith(".BSE") or symbol.endswith(".BO"):
        return "₹"  # Indian Rupee
    else:
        return "$"  # US Dollar


st.set_page_config(page_title="Stock Market Predictor", layout="wide")
st.title("📈 Stock Price Predictor")

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

# --- FETCH DATA & PREDICT HISTORICAL ---
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
                st.session_state['df'] = df  # Save data for future use
                st.subheader("Stock Data (Last 5 rows)")
                st.write(df.tail())

                # --- DATA PREPARATION ---
                data_train = pd.DataFrame(df.Close[0:int(len(df)*0.8)])
                data_test = pd.DataFrame(df.Close[int(len(df)*0.8):])

                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_train = scaler.fit_transform(data_train)

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

                # --- PREDICTION ON TEST DATA ---
                x_test, y_test = [], []
                for i in range(100, data_test_scaled.shape[0]):
                    x_test.append(data_test_scaled[i-100:i])
                    y_test.append(data_test_scaled[i,0])
                x_test, y_test = np.array(x_test), np.array(y_test)

                predict = model.predict(x_test)
                predict = scaler.inverse_transform(predict)
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

                st.subheader("Original vs Predicted Prices")
                test_dates = df.index[int(len(df)*0.8):]

                fig4, ax4 = plt.subplots(figsize=(10,6))
                ax4.plot(test_dates, y_test, 'g', label="Original Price")
                ax4.plot(test_dates, predict, 'r', label="Predicted Price")
                ax4.set_xlabel("Date")
                currency_symbol = get_currency(stock)
                ax4.set_ylabel(f"Price ({currency_symbol})")
                ax4.legend()
                fig4.autofmt_xdate()
                st.pyplot(fig4)


# ===============================
# 🚀 FUTURE FORECAST BUTTON BELOW
# ===============================

if st.button("Predict Future 30 Days"):
    if 'df' not in st.session_state:
        st.error("Please fetch and predict first before forecasting future prices.")
    elif model is None:
        st.warning("Model not loaded. Please check your model file.")
    else:
        df = st.session_state['df']
        st.subheader("Predicting Next 30 Days...")

        # --- SCALE THE DATA ---
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

        # --- TAKE LAST 100 DAYS AS INPUT FOR FORECAST ---
        last_100_days = scaled_data[-100:]
        last_100_days = list(last_100_days)

        future_output = []
        for i in range(30):  # predict next 30 days
            x_input = np.array(last_100_days[-100:]).reshape(1, 100, 1)
            y_pred = model.predict(x_input, verbose=0)
            last_100_days.append(y_pred[0])
            future_output.append(y_pred[0])

        # --- INVERSE TRANSFORM TO ORIGINAL SCALE ---
        future_output = scaler.inverse_transform(np.array(future_output).reshape(-1, 1))

        # --- CREATE FUTURE DATES ---
        last_date = df.index[-1]
        future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(30)]
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_output.flatten()})
        future_df.set_index('Date', inplace=True)

        # --- PLOT HISTORICAL + FUTURE ---
        st.subheader(" Historical vs Future Predicted Prices")
        fig5, ax5 = plt.subplots(figsize=(10,6))
        ax5.plot(df['Close'], label="Historical Price", color='blue')
        ax5.plot(future_df['Predicted_Price'], label="Predicted Future Price", linestyle='dashed', color='orange')
        ax5.set_xlabel("Date")
        ax5.set_ylabel(f"Price ({get_currency(stock)})")
        ax5.legend()
        st.pyplot(fig5)

        # --- SHOW TABLE ---
        st.subheader("30-Day Forecasted Prices")
        st.dataframe(future_df.style.format({"Predicted_Price": "{:.2f}"}))

        st.success("Future Forecast Completed Successfully!")
