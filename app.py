import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="EcoWatts – Smart Home Energy Analyzer", layout="wide")
st.title("⚡ EcoWatts – Smart Home Energy Analyzer")

# Load dataset
df = pd.read_csv("energy_usage_sample.csv")
df.columns = df.columns.str.strip()
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Time-based features
df['Hour'] = df['Timestamp'].dt.hour
df['Date'] = df['Timestamp'].dt.date

# --- 📊 Hourly Energy Usage Plot ---
st.subheader("📊 Hourly Energy Usage")
hourly_usage = df.groupby('Hour')['Usage_kWh'].sum().reset_index()
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

# Linear Regression Model
X = daily_usage[['Day_Index']]
y = daily_usage['Usage_kWh']
model = LinearRegression().fit(X, y)

# Predict next 10 days
future_days = np.arange(len(daily_usage), len(daily_usage) + 10).reshape(-1, 1)
predicted_usage = model.predict(future_days)
future_dates = pd.date_range(start=pd.to_datetime(daily_usage['Date'].iloc[-1]) + pd.Timedelta(days=1), periods=10)

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Usage_kWh': predicted_usage
})

# ✅ Fixed column name in plotting
fig_forecast = px.line(
    forecast_df, x='Date', y='Predicted_Usage_kWh',
    markers=True, title="Next 10 Days Forecast",
    labels={'Predicted_Usage_kWh': 'Forecasted Usage (kWh)'}
)
st.plotly_chart(fig_forecast)

# Display Table
st.write("📅 Forecast Table")
st.dataframe(forecast_df)
