
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import timedelta

st.set_page_config(page_title="EcoWatts – Smart Home Analyzer", layout="wide")
st.title("\U0001F50C EcoWatts – Smart Home Energy Analyzer")

page = st.sidebar.selectbox("Select a Page", ["Dashboard", "Forecast", "About"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date
    return df

uploaded_file = st.sidebar.file_uploader("Upload your energy usage CSV", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)

    if page == "Dashboard":
        st.header("\U0001F4CA Energy Usage Dashboard")
        daily_usage = df.groupby('Date')['Usage_kWh'].sum().reset_index()
        appliance_usage = df.groupby('Appliance')['Usage_kWh'].sum().reset_index()

        st.subheader("Daily Energy Usage")
        st.plotly_chart(px.line(daily_usage, x='Date', y='Usage_kWh', title='Daily Usage'))

        st.subheader("Top Consuming Appliances")
        st.plotly_chart(px.bar(appliance_usage, x='Appliance', y='Usage_kWh', title='Appliance Usage'))

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

        combined = pd.concat([daily_df[['Date', 'Usage_kWh']].rename(columns={'Usage_kWh': 'Predicted_Usage_kWh'}), forecast_df])
        st.plotly_chart(px.line(combined, x='Date', y='Predicted_Usage_kWh', title='Actual & Forecasted Energy Usage'))
        st.dataframe(forecast_df)

    elif page == "About":
        st.header("About EcoWatts")
        st.write("EcoWatts helps visualize energy usage and predict future consumption using linear regression.")
        st.write("Developed with Streamlit and Plotly for easy visualization.")
else:
    st.warning("Please upload your energy usage CSV file to get started.")
