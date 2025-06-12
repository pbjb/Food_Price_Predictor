import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv("data/food_prices.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

st.title("🍎 AI-Powered Food Price Predictor")

# Sidebar inputs
items = df["Item"].unique().tolist()
selected_item = st.selectbox("Select Food Item", items)
n_days = st.slider("Days to Predict", 1, 30, 7)

# Filter and prepare data
item_data = df[df["Item"] == selected_item][["Date", "Price"]]
item_data = item_data.rename(columns={"Date": "ds", "Price": "y"})

# Forecasting
model = Prophet()
model.fit(item_data)
future = model.make_future_dataframe(periods=n_days)
forecast = model.predict(future)

# Plot results
fig = px.line(forecast, x='ds', y='yhat', title=f"Price Forecast for {selected_item}")
fig.add_scatter(x=item_data["ds"], y=item_data["y"], mode='markers', name='Actual')

# Output
st.plotly_chart(fig)
st.subheader("Forecast Table")
st.write(forecast[["ds", "yhat"]].tail(n_days).rename(columns={"ds": "Date", "yhat": "Predicted Price"}))
