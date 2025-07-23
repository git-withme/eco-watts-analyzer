import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="EcoWatts – Smart Home Energy Analyzer", layout="wide")

st.title("⚡ EcoWatts – Smart Home Energy Analyzer")

# Load dataset
df = pd.read_csv("energy_usage_sample.csv")

# Use actual column names from your dataset
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Date'] = df['Timestamp'].dt.date

# Show sample data
st.subheader("📄 Sample Data")
st.dataframe(df.head())

# Hourly Energy Usage
hourly_usage = df.groupby('Hour')[' Usage_kWh'].sum()
fig1, ax1 = plt.subplots()
ax1.plot(hourly_usage.index, hourly_usage.values, marker='o', color='green')
ax1.set_title("Hourly Energy Usage Pattern")
ax1.set_xlabel("Hour")
ax1.set_ylabel("Usage (kWh)")
st.pyplot(fig1)

# Appliance Usage Pie Chart
appliance_usage = df.groupby('appliance')[' Usage_kWh'].sum()
fig2, ax2 = plt.subplots()
ax2.pie(appliance_usage, labels=appliance_usage.index, autopct='%1.1f%%', startangle=90)
ax2.set_title("Usage Distribution by Appliance")
st.pyplot(fig2)

# Room vs Hour Heatmap
heatmap_data = df.pivot_table(values=' Usage_kWh', index='Room', columns='Hour', aggfunc='sum')
fig3, ax3 = plt.subplots()
sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax3)
ax3.set_title("Room vs Hourly Energy Heatmap")
st.pyplot(fig3)

# Daily Energy Usage
daily_usage = df.groupby('Date')[' Usage_kWh'].sum()
fig4, ax4 = plt.subplots()
ax4.bar(daily_usage.index.astype(str), daily_usage.values, color='skyblue')
ax4.set_title("Total Daily Energy Usage")
ax4.set_xticklabels(daily_usage.index.astype(str), rotation=45)
st.pyplot(fig4)

# Cost by Appliance
cost_by_appliance = df.groupby('appliance')['Cost(INR)'].sum()
fig5, ax5 = plt.subplots()
ax5.bar(cost_by_appliance.index, cost_by_appliance.values, color='orange')
ax5.set_title("Total Cost by Appliance (INR)")
ax5.tick_params(axis='x', rotation=10)
st.pyplot(fig5)

# Smart Energy Saving Tip
tips = {
    "Air Conditioner": "Set to 25°C & use sleep mode.",
    "Lights": "Use LED & motion sensors.",
    "Geyser": "Use timer or thermostat to control Geyser.",
    "Fridge": "Check for door seal leaks.",
    "Washing Machine": "Run full loads and use eco mode."
}
top_appliance = appliance_usage.idxmax()
mean_usage = df[df['appliance'] == top_appliance][' Usage_kWh'].mean()

if top_appliance == "Geyser" and mean_usage > 1.5:
    tip = tips["Geyser"]
else:
    tip = tips.get(top_appliance, "Consider unplugging unused appliances.")

st.markdown(f"### 💡 Smart Tip for Most Used Appliance: `{top_appliance}`")
st.info(f"{tip}")

# Forecasting Future Energy Usage (Next 10 Days)
daily_df = df.groupby('Date')[' Usage_kWh'].sum().reset_index()
daily_df['Day_Index'] = np.arange(len(daily_df))
X = daily_df[['Day_Index']]
y = daily_df[' Usage_kWh']
model = LinearRegression()
model.fit(X, y)

# Predict next 10 days
future_index = np.arange(len(daily_df), len(daily_df) + 10).reshape(-1, 1)
future_usage = model.predict(future_index)
future_dates = pd.date_range(start=daily_df['Date'].max() + pd.Timedelta(days=1), periods=10)

# Forecast plot
fig6, ax6 = plt.subplots()
ax6.plot(daily_df['Date'], daily_df[' Usage_kWh'], label='Actual Usage', marker='o')
ax6.plot(future_dates, future_usage, label='Predicted Usage', linestyle='--', marker='x', color='red')
ax6.set_title("🔮 Forecast: Next 10 Days Energy Usage")
ax6.set_xlabel("Date")
ax6.set_ylabel("Usage (kWh)")
ax6.legend()
st.pyplot(fig6)

# Forecast table
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Usage (kWh)': np.round(future_usage, 2)
})
st.subheader("📊 Forecasted Energy Usage")
st.dataframe(forecast_df)




