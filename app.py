import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("⚡ EcoWatts – Smart Home Energy Dashboard")

# Load dataset
df = pd.read_csv("energy_usage_sample.csv")
df.columns = df.columns.str.strip()  # Strip spaces in column names

# Convert 'Timestamp' to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Date'] = df['Timestamp'].dt.date

# --- 📊 HOURLY ENERGY USAGE ---
st.subheader("📊 Hourly Energy Usage Pattern")
hourly_usage = df.groupby('Hour')['Usage_kWh'].sum()
fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(hourly_usage.index, hourly_usage.values, marker='o', color='green')
ax1.set_title("Hourly Energy Usage")
ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("Usage (kWh)")
ax1.grid(True)
st.pyplot(fig1)

# --- 🔮 DAILY USAGE PREDICTION FOR NEXT 10 DAYS ---
st.subheader("🔮 Next 10 Days Usage Prediction (Linear Regression)")

# Group by date
daily_usage = df.groupby('Date')['Usage_kWh'].sum().reset_index()
daily_usage['Day_Index'] = range(len(daily_usage))  # X-axis

# Train simple linear regression model
X = daily_usage[['Day_Index']]
y = daily_usage['Usage_kWh']
model = LinearRegression()
model.fit(X, y)

# Predict for next 10 days
future_days = np.arange(len(daily_usage), len(daily_usage) + 10).reshape(-1, 1)
predicted_usage = model.predict(future_days)

# Create forecast DataFrame
future_dates = pd.date_range(start=daily_usage['Date'].iloc[-1] + pd.Timedelta(days=1), periods=10)
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Usage_kWh': predicted_usage})

# Plot prediction
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(daily_usage['Date'], daily_usage['Usage_kWh'], label='Actual', color='blue')
ax2.plot(forecast_df['Date'], forecast_df['Predicted_Usage_kWh'], label='Forecast', color='red', linestyle='--')
ax2.set_title("Daily Usage Forecast (Next 10 Days)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Usage (kWh)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# Show forecast table
st.write("📅 Forecast Table")
st.dataframe(forecast_df)



