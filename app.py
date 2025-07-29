import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# App title and setup
st.title("EcoWatt - Simple Energy Analyzer")
st.sidebar.header("Settings")

# Sample data in case user doesn't upload
sample_data = {
    "Date": [f"2023-01-{i}" for i in range(1, 31)],
    "Appliance": ["Fridge"]*10 + ["AC"]*10 + ["Lights"]*10,
    "Usage": [5, 5, 6, 5, 4, 6, 5, 7, 6, 5] + 
             [15, 17, 16, 18, 19, 17, 16, 20, 18, 19] + 
             [3, 4, 3, 2, 3, 4, 5, 3, 4, 3]
}
df = pd.DataFrame(sample_data)

# Allow file upload
uploaded_file = st.sidebar.file_uploader("Upload your energy data (CSV)")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")

# Show graph selector
appliance = st.selectbox("Select appliance", df['Appliance'].unique())

# Filter data
filtered_df = df[df['Appliance'] == appliance]

# Basic visualization
st.subheader("Energy Usage")
st.line_chart(filtered_df.set_index('Date')['Usage'])

# Simple stats
st.subheader("Statistics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Usage", f"{filtered_df['Usage'].sum()} kWh")
    st.metric("Average Daily", f"{filtered_df['Usage'].mean():.1f} kWh")
with col2:
    st.metric("Max Daily", f"{filtered_df['Usage'].max()} kWh")
    st.metric("Days Recorded", len(filtered_df))

# Simple forecast
if st.button("Show 7-day forecast"):
    model = LinearRegression()
    dates = [(datetime.strptime(d, "%Y-%m-%d") - datetime(2023,1,1)).days for d in filtered_df['Date']]
    model.fit([[d] for d in dates], filtered_df['Usage'])
    
    future_dates = [datetime(2023,1,1) + timedelta(days=max(dates)+i) for i in range(1,8)]
    forecast = model.predict([[max(dates)+i] for i in range(1,8)])
    
    forecast_df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
        "Forecast": forecast
    })
    
    st.line_chart(forecast_df.set_index('Date'))

# Energy saving tips
st.subheader("Tips to Save Energy")
if appliance == "AC":
    st.write("- Set thermostat to 24°C or higher")
    st.write("- Clean filters monthly")
elif appliance == "Fridge":
    st.write("- Don't leave door open long")
    st.write("- Keep coils clean")
else:
    st.write("- Switch to LED bulbs")
    st.write("- Turn off when not in use")

# Download button
@st.cache_data
def convert_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_to_csv(filtered_df)
st.download_button(
    "Download Report",
    csv,
    "energy_report.csv",
    "text/csv"
)
