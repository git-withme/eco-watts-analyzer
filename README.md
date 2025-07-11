# ⚡ EcoWatts – Smart Home Energy Analyzer

EcoWatts is a Streamlit-powered dashboard that helps homeowners monitor and optimize their electricity usage.  
It visualizes energy consumption by appliance, room, and time, while predicting future energy needs using machine learning.

## ✅ Key Features
- Upload and analyze hourly energy usage CSV
- Interactive visualizations: Pie charts, heatmaps, and line graphs
- Predict next 10–15 days energy usage with Linear Regression
- Eco-friendly usage tips based on consumption patterns

## 🚀 How to Run
1. Install Streamlit and required libraries:
   pip install -r requirements.txt
2. Run the app:
   streamlit run app.py
3. Upload your energy usage CSV file and explore insights.

## 📦 Dataset Columns
- Timestamp
- Appliance
- Usage_kWh
- Room
- Mode
- Temp(C)
- Cost(INR)

## 🔮 Future Enhancements
- IoT smart plug integration
- Dynamic tariff analysis
- Occupancy-based predictions
- Mobile app deployment

