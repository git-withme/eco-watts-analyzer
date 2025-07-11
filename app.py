import streamlit as st
import pandas as pd
import plotly.express as px

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

