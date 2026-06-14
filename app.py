import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
st.set_page_config(
    page_title="Real Time Stock Market Dashboard",
    layout="wide"
)

st.markdown("""
<h1 style='text-align:center;color:#00cc66;'>
📈 REAL TIME STOCK MARKET DASHBOARD
</h1>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Stock Market Dashboard")

page = st.sidebar.radio(
    "Select Section",
    ["Trending Stocks", "Search Stock", "Analysis"]
)

# ------------------------------
# TRENDING STOCKS PAGE
# ------------------------------

if page == "Trending Stocks":

    st.header("🔥 Trending Stocks")

    trending = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
        "NFLX"
    ]

    cols = st.columns(4)

    for i, symbol in enumerate(trending):
        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            with cols[i % 4]:
                st.metric(
                    label=symbol,
                    value=info.get("currentPrice", "N/A")
                )

        except:
            pass

# ------------------------------
# SEARCH STOCK PAGE
# ------------------------------

elif page == "Search Stock":

    st.header("🔍 Search Stock")

    ticker = st.text_input(
        "Enter Stock Symbol",
        "AAPL"
    )

    if ticker:

        stock = yf.Ticker(ticker)

        try:

            info = stock.info

            st.subheader("Company Information")

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    "Company Name:",
                    info.get("longName", "N/A")
                )

                st.write(
                    "Sector:",
                    info.get("sector", "N/A")
                )

                st.write(
                    "Industry:",
                    info.get("industry", "N/A")
                )

                st.write(
                    "Country:",
                    info.get("country", "N/A")
                )

            with col2:

                st.write(
                    "Market Cap:",
                    info.get("marketCap", "N/A")
                )

                st.write(
                    "PE Ratio:",
                    info.get("trailingPE", "N/A")
                )

                st.write(
                    "Current Price:",
                    info.get("currentPrice", "N/A")
                )

                st.write(
                    "Website:",
                    info.get("website", "N/A")
                )

            data = stock.history(period="1y")

            st.subheader("Historical Data")

            st.dataframe(data.tail(20))

            st.subheader("Candlestick Chart")

            fig = go.Figure()

            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    name="Price"
                )
            )

            fig.update_layout(
                height=600
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except:
            st.error("Invalid Stock Symbol")

            # ------------------------------
# ANALYSIS PAGE
# ------------------------------

elif page == "Analysis":

    st.header("📊 Stock Analysis")

    ticker = st.text_input(
        "Enter Stock Symbol",
        "AAPL",
        key="analysis"
    )

    start_date = st.date_input(
        "Start Date",
        pd.to_datetime("2023-01-01")
    )

    end_date = st.date_input(
        "End Date",
        pd.to_datetime("today")
    )

    if ticker:

        stock = yf.Ticker(ticker)

        data = stock.history(
            start=start_date,
            end=end_date
        )

        if not data.empty:

            # SMA & EMA

            data["SMA50"] = (
                data["Close"]
                .rolling(window=50)
                .mean()
            )

            data["EMA50"] = (
                data["Close"]
                .ewm(span=50, adjust=False)
                .mean()
            )

            st.subheader("SMA & EMA")

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    name="Close Price"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["SMA50"],
                    name="SMA 50"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["EMA50"],
                    name="EMA 50"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # RSI

            delta = data["Close"].diff()

            gain = (
                delta.where(delta > 0, 0)
                .rolling(14)
                .mean()
            )

            loss = (
                -delta.where(delta < 0, 0)
                .rolling(14)
                .mean()
            )

            rs = gain / loss

            data["RSI"] = (
                100 - (100 / (1 + rs))
            )

            st.subheader("RSI Indicator")

            fig_rsi = go.Figure()

            fig_rsi.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["RSI"],
                    name="RSI"
                )
            )

            fig_rsi.add_hline(y=70)

            fig_rsi.add_hline(y=30)

            st.plotly_chart(
                fig_rsi,
                use_container_width=True
            )

            # MACD

            data["EMA12"] = (
                data["Close"]
                .ewm(span=12, adjust=False)
                .mean()
            )

            data["EMA26"] = (
                data["Close"]
                .ewm(span=26, adjust=False)
                .mean()
            )

            data["MACD"] = (
                data["EMA12"] -
                data["EMA26"]
            )

            data["Signal"] = (
                data["MACD"]
                .ewm(span=9, adjust=False)
                .mean()
            )

            st.subheader("MACD")

            fig_macd = go.Figure()

            fig_macd.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["MACD"],
                    name="MACD"
                )
            )

            fig_macd.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["Signal"],
                    name="Signal"
                )
            )

            st.plotly_chart(
                fig_macd,
                use_container_width=True
            )

            # Bollinger Bands

            data["MA20"] = (
                data["Close"]
                .rolling(20)
                .mean()
            )

            data["STD20"] = (
                data["Close"]
                .rolling(20)
                .std()
            )

            data["Upper"] = (
                data["MA20"] +
                (data["STD20"] * 2)
            )

            data["Lower"] = (
                data["MA20"] -
                (data["STD20"] * 2)
            )

            st.subheader(
                "Bollinger Bands"
            )

            fig_bb = go.Figure()

            fig_bb.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    name="Close"
                )
            )

            fig_bb.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["Upper"],
                    name="Upper Band"
                )
            )

            fig_bb.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["Lower"],
                    name="Lower Band"
                )
            )

            st.plotly_chart(
                fig_bb,
                use_container_width=True
            )

            # Cumulative Returns

            data["Daily_Return"] = (
                data["Close"]
                .pct_change()
            )

            data["Cumulative_Return"] = (
                (1 + data["Daily_Return"])
                .cumprod() - 1
            )

            st.subheader(
                "Cumulative Returns"
            )

            fig_return = go.Figure()

            fig_return.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["Cumulative_Return"],
                    name="Returns"
                )
            )

            st.plotly_chart(
                fig_return,
                use_container_width=True
            )
            st.subheader("🤖 Machine Learning Prediction")
            ml_data = data[["Close"]].copy()

            ml_data["Prediction"] = ml_data["Close"].shift(-1)

            ml_data = ml_data.dropna()

            X = ml_data[["Close"]]

            y = ml_data["Prediction"]

            model = LinearRegression()

            model.fit(X, y)

            latest_price = data["Close"].iloc[-1]

            next_day_prediction = model.predict(
                [[latest_price]]
            )[0]

            st.success(
                f"Predicted Next Day Price: ${next_day_prediction:.2f}"
            )


        else:
            st.error(
                "No data available"
            )


st.markdown("---")
st.markdown(
    "Developed by Sameela Rathnagari | Real Time Stock Market Dashboard"
)