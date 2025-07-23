import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("⚡ EcoWatts – Smart Home Energy Dashboard")

# Load dataset
df = pd.read_csv("energy_usage_sample.csv")
df.columns = df.columns.str.strip()
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Add time features
df['Hour'] = df['Timestamp'].dt.hour
df['Date'] = df['Timestamp'].dt.date

# --- 📊 Hourly Energy Usage with Plotly ---
st.subheader("📊 Hourly Energy Usage")
hourly_usage = df.groupby('Hour')['Usage_kWh'].sum().reset_index()

# 🔁 Ensure column names are correct
hourly_usage.columns = ['Hour', 'Usage_kWh']

fig_hourly = px.line(
    hourly_usage, x='Hour', y='Usage_kWh',
    markers=True, title="Hourly Energy Consumption",
    labels={'Usage_kWh': 'Energy Used (kWh)', 'Hour': 'Hour of Day'}
)
st.plotly_chart(fig_hourly)

# --- 🔮 Next 10 Days Prediction ---
st.subheader("🔮 Next 10 Days Usage Forecast")

daily_usage = df.groupby('Date')['Usage_kWh'].sum().reset_index()
daily_usage['Day_Index'] = range(len(daily_usage))

X = daily_usage[['Day_Index']]
y = daily_usage['Usage_kWh']
model = LinearRegression().fit(X, y)

future_days = np.arange(len(daily_usage), len(daily_usage) + 10).reshape(-1, 1)
predicted_usage = model.predict(future_days)
future_dates = pd.date_range(start=daily_usage['Date'].iloc[-1] + pd.Timedelta(days=1), periods=10)

forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Usage_kWh': predicted_usage})

fig_forecast = px.line(
    forecast_df, x='Date', y='Predicted_Usage_KWh',
    markers=True, title="Next 10 Days Forecast",
    labels={'Predicted_Usage_KWh': 'Forecasted Usage (kWh)'}
)
st.plotly_chart(fig_forecast)

st.write("📅 Forecast Table")
st.dataframe(forecast_df)
