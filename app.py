# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Smart Home Energy Analyzer", layout="wide")
st.title("🏠 Smart Home Energy Usage Analyzer & Predictor")

# Load data
df = pd.read_csv("energy_usage_sample.csv", parse_dates=["Timestamp"])
df['Hour'] = df['Timestamp'].dt.hour
df['Date'] = df['Timestamp'].dt.date

st.subheader("📋 Sample Data")
st.dataframe(df.head())

# Hourly Usage
hourly_usage = df.groupby('Hour')['Usage_kWh'].sum()
st.subheader("⏰ Hourly Energy Usage")
fig, ax = plt.subplots()
ax.plot(hourly_usage.index, hourly_usage.values, marker='o', color='green')
ax.set_xlabel("Hour")
ax.set_ylabel("Usage (kWh)")
ax.set_title("Hourly Energy Usage")
st.pyplot(fig)

# Appliance Pie Chart
appliance_usage = df.groupby('Appliance')['Usage_kWh'].sum()
st.subheader("📊 Appliance Usage Distribution")
fig, ax = plt.subplots()
ax.pie(appliance_usage, labels=appliance_usage.index, autopct='%1.1f%%', startangle=90)
ax.set_title("Usage by Appliance")
st.pyplot(fig)

# Room vs Hour Heatmap
st.subheader("🧭 Room vs Hourly Heatmap")
heatmap_data = df.pivot_table(values='Usage_kWh', index='Room', columns='Hour', aggfunc='sum')
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(heatmap_data, cmap="YlOrBr", ax=ax)
st.pyplot(fig)

# Daily Usage Bar
daily_usage = df.groupby('Date')['Usage_kWh'].sum()
st.subheader("📅 Daily Energy Usage")
fig, ax = plt.subplots()
ax.bar(daily_usage.index.astype(str), daily_usage.values, color='skyblue')
plt.xticks(rotation=45)
st.pyplot(fig)

# Cost by Appliance
cost_by_appliance = df.groupby('Appliance')['Cost(INR)'].sum()
st.subheader("💰 Cost by Appliance")
fig, ax = plt.subplots()
ax.bar(cost_by_appliance.index, cost_by_appliance.values, color='orange')
st.pyplot(fig)

# Smart Tip Generator
tips = {
    "Air Conditioner": "Set to 25°C & use sleep mode.",
    "Lights": "Use LED & motion sensors.",
    "Geyser": "Limit usage to 10 minutes.",
    "Fridge": "Avoid frequent door opening.",
    "Washing Machine": "Use eco mode & full loads."
}
top_appliance = appliance_usage.idxmax()
tip = tips.get(top_appliance, "Consider unplugging unused appliances.")
st.subheader("💡 Smart Energy Tip")
st.markdown(f"**Top Appliance:** {top_appliance}")
st.info(f"💡 Tip: {tip}")

# Forecasting with Linear Regression
st.subheader("🔮 Energy Forecast (Next 10 Days)")
daily_df = df.groupby('Date')['Usage_kWh'].sum().reset_index()
daily_df['Day_Index'] = np.arange(len(daily_df))
X = daily_df[['Day_Index']]
y = daily_df['Usage_kWh']
model = LinearRegression()
model.fit(X, y)

# Predict next 10 days
future_index = np.arange(len(daily_df), len(daily_df)+10).reshape(-1, 1)
future_usage = model.predict(future_index)
future_dates = pd.date_range(start=daily_df['Date'].max() + pd.Timedelta(days=1), periods=10)

# Plot forecast
fig, ax = plt.subplots()
ax.plot(daily_df['Date'], daily_df['Usage_kWh'], label='Actual', marker='o')
ax.plot(future_dates, future_usage, label='Predicted', linestyle='--', marker='x', color='red')
ax.set_title("Energy Usage Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Usage (kWh)")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Forecast Table
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Usage_kWh": np.round(future_usage, 2)
})
st.dataframe(forecast_df)



