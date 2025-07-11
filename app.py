
import streamlit as st
import pandas as pd

st.title("🔌 EcoWatts – Smart Home Energy Analyzer")

uploaded_file = st.file_uploader("energy_usage_sample.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Timestamp'])
    st.write("📊 Preview of Uploaded Data:")
    st.dataframe(df.head())

    # Example visualization
    df['Date'] = df['Timestamp'].dt.date
    daily_usage = df.groupby('Date')['Usage_kWh'].sum()
    st.line_chart(daily_usage)
