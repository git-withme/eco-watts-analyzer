import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import timedelta

st.set_page_config(page_title="EcoWatts – Smart Home Analyzer", layout="wide")
st.title("⚡ EcoWatts – Smart Home Energy Analyzer")

# Fixed dataset loading
df = pd.read_csv("energy_usage_sample.csv", parse_dates=["Timestamp"])
df['Date'] = df['Timestamp'].dt.date

st.write("📊 Energy Usage Data Preview")
st.dataframe(df.head())

# Dashboard View
st.header("📈 Energy Usage Dashboard")
daily_usage = df.groupby('Date')['Usage_kWh'].sum().reset_index()
appliance_usage = df.groupby('Appliance')['Usage_kWh'].sum().reset_index()

st.subheader("Daily Energy Usage")
st.plotly_chart(px.line(daily_usage, x='Date', y='Usage_kWh', title='Daily Energy Usage'))

st.subheader("Top Consuming Appliances")
st.plotly_chart(px.bar(appliance_usage, x='Appliance', y='Usage_kWh', title='Top Consuming Appliances'))

# Forecast View
st.header("🔮 10-Day Energy Usage Forecast")
daily_usage['Day_Index'] = range(len(daily_usage))
X = daily_usage[['Day_Index']]
y = daily_usage['Usage_kWh']

model = LinearRegression()
model.fit(X, y)

future_index = pd.DataFrame({'Day_Index': range(len(daily_usage), len(daily_usage) + 10)})
future_usage = model.predict(future_index)

future_dates = pd.date_range(start=pd.to_datetime(daily_usage['Date'].max()) + timedelta(days=1), periods=10)
forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Usage_kWh": future_usage.round(2)})

combined = pd.concat([daily_usage[['Date', 'Usage_kWh']].rename(columns={'Usage_kWh': 'Predicted_Usage_kWh'}), forecast_df])
st.plotly_chart(px.line(combined, x='Date', y='Predicted_Usage_kWh', title='Actual & Forecasted Energy Usage'))
st.dataframe(forecast_df)


