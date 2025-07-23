import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="EcoWatts – Energy Analyzer", layout="wide")
st.title("⚡ EcoWatts – Smart Home Energy Dashboard")

# Load and clean data
df = pd.read_csv("energy_usage_sample.csv")
df.columns = df.columns.str.strip()  # Clean column names
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Date'] = df['Timestamp'].dt.date

# -----------------------------------------------
# 📊 Hourly Usage Line Chart with Plotly
# -----------------------------------------------
st.subheader("📈 Hourly Energy Usage Pattern")
hourly_usage = df.groupby('Hour')['Usage_kWh'].sum().reset_index()

fig_hourly = px.line(
    hourly_usage, x='Hour', y='Usage_KWh',
    markers=True, title="Hourly Energy Consumption",
    labels={'Usage_KWh': 'Energy Used (kWh)', 'Hour': 'Hour of Day'}
)
fig_hourly.update_traces(line=dict(color='green'))
st.plotly_chart(fig_hourly, use_container_width=True)

# -----------------------------------------------
# 🔮 Next 10 Days Prediction using Linear Regression
# -----------------------------------------------
st.subheader("🔮 Forecast: Next 10 Days Energy Usage")

# Prepare daily usage
daily_usage = df.groupby('Date')['Usage_KWh'].sum().reset_index()
daily_usage['Day_Index'] = range(len(daily_usage))  # For regression

# Linear Regression model
X = daily_usage[['Day_Index']]
y = daily_usage['Usage_KWh']
model = LinearRegression()
model.fit(X, y)

# Predict next 10 days
future_indices = np.arange(len(daily_usage), len(daily_usage) + 10).reshape(-1, 1)
predicted_usage = model.predict(future_indices)
future_dates = pd.date_range(start=daily_usage['Date'].iloc[-1] + pd.Timedelta(days=1), periods=10)

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Usage_KWh': predicted_usage
})

# Combine actual and forecast for plot
combined = pd.concat([
    daily_usage[['Date', 'Usage_KWh']].rename(columns={'Usage_KWh': 'kWh'}),
    forecast_df.rename(columns={'Predicted_Usage_KWh': 'kWh'})
])
combined['Type'] = ['Actual'] * len(daily_usage) + ['Forecast'] * 10

# Plot
fig_forecast = px.line(
    combined, x='Date', y='kWh', color='Type',
    title="Energy Usage Forecast (Next 10 Days)",
    labels={'kWh': 'Energy Used (kWh)'}
)
fig_forecast.update_traces(mode='lines+markers')
st.plotly_chart(fig_forecast, use_container_width=True)

# Show Forecast Table
st.write("📋 Forecast Table")
st.dataframe(forecast_df)
