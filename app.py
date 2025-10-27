import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --------------------------
# Utility Functions
# --------------------------

def get_currency(symbol):
    symbol = symbol.upper()
    if symbol.endswith(".NS") or symbol.endswith(".BSE") or symbol.endswith(".BO"):
        return "₹"
    else:
        return "$"

@st.cache_data(show_spinner=True)
def get_stock_data(stock, start, end):
    df = yf.download(stock, start=start, end=end, progress=False)

    # Handle multi-level columns (some yfinance versions return tuples)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    if df.empty:
        raise ValueError("No data fetched. Please check the stock symbol or date range.")

    # Ensure required columns exist
    expected_cols = ['Open', 'High', 'Low', 'Close']
    for col in expected_cols:
        if col not in df.columns:
            raise KeyError(f"Missing expected column: {col}")

    # Clean missing values
    df = df.dropna(subset=expected_cols)

    return df

def compute_indicators(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().apply(lambda x: (x + 1)).rolling(14).mean()))
    df['Upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
    df['Lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()
    return df

# --------------------------
# Streamlit Setup
# --------------------------
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("Stock Market Predictor Dashboard")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Forecast", "Analytics"])

# --------------------------
# Model Loading
# --------------------------
model = None
if os.path.exists("Stock_Prediction_new.h5"):
    model = load_model("Stock_Prediction_new.h5", compile=False)
    st.sidebar.success("Model Loaded (.h5)")
elif os.path.exists("Stock_Prediction_new.keras"):
    model = load_model("Stock_Prediction_new.keras", compile=False)
    st.sidebar.success("Model Loaded (.keras)")
else:
    st.sidebar.error("Model file not found!")

# --------------------------
# Common Inputs
# --------------------------
stock = st.sidebar.text_input("Enter Stock Symbol", "GOOG")
start = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end = st.sidebar.date_input("End Date", datetime.date.today())

# --------------------------
# DASHBOARD SECTION
# --------------------------
if page == "Dashboard":
    st.header("Stock Overview & Prediction")

    if st.button("Fetch & Predict"):
        if not model:
            st.warning("Model not loaded.")
        else:
            df = get_stock_data(stock, start, end)
            if df.empty:
                st.error("No data found.")
            else:
                st.session_state['df'] = df
                df = compute_indicators(df)
                st.write(df.tail())

                # Plotly Candlestick
                st.subheader("Interactive Candlestick Chart")
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'], name='Price'
                )])
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA12'], mode='lines', name='EMA12'))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA26'], mode='lines', name='EMA26'))
                fig.update_layout(title=f"{stock} Price Chart", xaxis_title="Date", yaxis_title=f"Price ({get_currency(stock)})")
                st.plotly_chart(fig, use_container_width=True)

                # Moving Averages
                ma50 = df['Close'].rolling(50).mean()
                ma100 = df['Close'].rolling(100).mean()
                fig_ma, ax_ma = plt.subplots(figsize=(10,5))
                ax_ma.plot(df['Close'], label="Close", color='blue')
                ax_ma.plot(ma50, label="MA50", color='red')
                ax_ma.plot(ma100, label="MA100", color='green')
                ax_ma.legend()
                st.pyplot(fig_ma)

                # Prepare Data
                data_train = pd.DataFrame(df.Close[0:int(len(df)*0.8)])
                data_test = pd.DataFrame(df.Close[int(len(df)*0.8):])
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_train = scaler.fit_transform(data_train)
                past_100_days = data_train.tail(100)
                data_test = pd.concat([past_100_days, data_test], ignore_index=True)
                data_test_scaled = scaler.transform(data_test)

                x_test, y_test = [], []
                for i in range(100, data_test_scaled.shape[0]):
                    x_test.append(data_test_scaled[i-100:i])
                    y_test.append(data_test_scaled[i,0])
                x_test, y_test = np.array(x_test), np.array(y_test)

                predict = model.predict(x_test)
                predict = scaler.inverse_transform(predict)
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

                # Metrics
                mse = mean_squared_error(y_test, predict)
                mae = mean_absolute_error(y_test, predict)
                st.metric(label="Mean Absolute Error", value=f"{mae:.2f}")
                st.metric(label="Mean Squared Error", value=f"{mse:.2f}")

                # Plot Original vs Predicted
                st.subheader("Actual vs Predicted Prices")
                fig_pred, ax_pred = plt.subplots(figsize=(10,5))
                ax_pred.plot(df.index[int(len(df)*0.8):], y_test, 'g', label="Actual")
                ax_pred.plot(df.index[int(len(df)*0.8):], predict, 'r', label="Predicted")
                ax_pred.legend()
                st.pyplot(fig_pred)

# --------------------------
# FORECAST SECTION
# --------------------------
elif page == "Forecast":
    st.header("🔮 Future Price Forecasting")
    future_days = st.slider("Select forecast days", 1, 60, 7)

    if st.button("Predict Future Prices"):
        if 'df' not in st.session_state:
            st.error("Please fetch stock data first.")
        else:
            df = st.session_state['df']
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

            last_100_days = list(scaled_data[-100:])
            future_output = []

            for i in range(future_days):
                x_input = np.array(last_100_days[-100:]).reshape(1, 100, 1)
                y_pred = model.predict(x_input, verbose=0)
                last_100_days.append(y_pred[0])
                future_output.append(y_pred[0])

            future_output = scaler.inverse_transform(np.array(future_output).reshape(-1, 1))
            last_date = df.index[-1]
            future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(future_days)]
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_output.flatten()})
            future_df.set_index('Date', inplace=True)

            st.subheader("Forecasted Price Chart")
            fig_future, ax_future = plt.subplots(figsize=(10,5))
            ax_future.plot(df['Close'], label="Historical", color='blue')
            ax_future.plot(future_df['Predicted_Price'], label="Future Forecast", linestyle='dashed', color='orange')
            ax_future.legend()
            st.pyplot(fig_future)

            st.subheader("Forecasted Values")
            st.dataframe(future_df.style.format({"Predicted_Price": "{:.2f}"}))

            csv = future_df.to_csv().encode('utf-8')
            st.download_button(label="Download Forecast CSV", data=csv, file_name=f"{stock}_forecast.csv", mime='text/csv')

# --------------------------
# ANALYTICS SECTION
# --------------------------
elif page == "Analytics":
    st.header("Technical Analytics (Indicators)")
    if 'df' not in st.session_state:
        st.warning("Please fetch data from the Dashboard first.")
    else:
        df = compute_indicators(st.session_state['df'])

        st.subheader("MACD vs Signal Line")
        fig_macd, ax_macd = plt.subplots(figsize=(10,5))
        ax_macd.plot(df['MACD'], label='MACD', color='red')
        ax_macd.plot(df['Signal'], label='Signal', color='blue')
        ax_macd.legend()
        st.pyplot(fig_macd)

        st.subheader("Bollinger Bands")
        fig_bb, ax_bb = plt.subplots(figsize=(10,5))
        ax_bb.plot(df['Close'], label='Close', color='blue')
        ax_bb.plot(df['Upper'], label='Upper Band', linestyle='dashed', color='red')
        ax_bb.plot(df['Lower'], label='Lower Band', linestyle='dashed', color='green')
        ax_bb.legend()
        st.pyplot(fig_bb)
