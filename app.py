
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import timedelta

st.set_page_config(page_title="EcoWatts – Smart Home Analyzer", layout="wide")
st.title("\U0001F50C EcoWatts – Smart Home Energy Analyzer")

# Sidebar Navigation
page = st.sidebar.selectbox("Select a Page", ["Dashboard", "Forecast", "About"])

# Upload CSV once and cache it
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date
    return df

uploaded_file = st.sidebar.file_uploader("Upload your energy CSV", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)

    if page == "Dashboard":
        st.header("\U0001F4CA Energy Usage Dashboard")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Daily Energy Usage (kWh)")
            daily_usage = df.groupby('Date')['Usage_kWh'].sum()
            st.line_chart(daily_usage)
        
        with col2:
            st.subheader("Top Consuming Appliances")
            appliance_usage = df.groupby('Appliance')['Usage_kWh'].sum().sort_values(ascending=False)
            st.bar_chart(appliance_usage)
        
        st.subheader("Room-wise Usage Heatmap")
        pivot_df = df.pivot_table(index='Room', columns='Mode', values='Usage_kWh', aggfunc='sum').fillna(0)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

    elif page == "Forecast":
        st.header("\U0001F52E 10-Day Energy Usage Forecast")
        daily_df = df.groupby('Date')['Usage_kWh'].sum().reset_index()
        daily_df['Day_Index'] = range(len(daily_df))

        X = daily_df[['Day_Index']]
        y = daily_df['Usage_kWh']

        model = LinearRegression()
        model.fit(X, y)

        future_index = pd.DataFrame({'Day_Index': range(len(daily_df), len(daily_df) + 10)})
        future_usage = model.predict(future_index)

        future_dates = pd.date_range(start=pd.to_datetime(daily_df['Date'].max()) + timedelta(days=1), periods=10)
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Usage_kWh": future_usage.round(2)})
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily_df['Date'], daily_df['Usage_kWh'], label='Actual', marker='o')
        ax.plot(future_dates, future_usage, label='Predicted (Next 10 Days)', linestyle='--', marker='x', color='red')
        ax.set_title("10-Day Energy Usage Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Usage (kWh)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.dataframe(forecast_df)
    
    elif page == "About":
        st.header("About EcoWatts")
        st.write("EcoWatts is a smart home energy analyzer that helps you visualize your energy consumption patterns and predict future usage.")
        st.write("Developed using Python, Streamlit, and machine learning techniques.")

else:
    st.warning("\U0001F4C2 Please upload your energy usage CSV file to get started.")
