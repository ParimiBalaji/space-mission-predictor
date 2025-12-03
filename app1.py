import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("mission_success_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("🚀 Space Mission Success Prediction App")
st.write("Enter mission details below to predict whether your mission will be successful.")

# User Inputs
country = st.selectbox("Country", ["USA", "Russia", "India", "China", "Japan", "Other"])
mission_type = st.selectbox("Mission Type", ["Orbiter", "Lander", "Rover", "Flyby", "Test"])
launch_site = st.text_input("Launch Site (Example: Cape Canaveral)")
satellite_type = st.text_input("Satellite Type (Example: Communication, Research, Weather)")
budget = st.number_input("Budget (in Billion $)", min_value=0.1, step=0.1)
success_rate = st.slider("Success Rate (%)", 0, 100, 50)
technology_used = st.text_input("Technology Used")
duration = st.number_input("Mission Duration (Days)", min_value=1, step=1)
crew_size = st.number_input("Crew Size", min_value=0, step=1)
risk_level = st.selectbox("Mission Risk Level", ["Low", "Medium", "High"])


# Create dataframe
input_data = pd.DataFrame({
    "Country": [country],
    "Mission Type": [mission_type],
    "Launch Site": [launch_site],
    "Satellite Type": [satellite_type],
    "Budget (in Billion $)": [budget],
    "Success Rate (%)": [success_rate],
    "Technology Used": [technology_used],
    "Duration (in Days)": [duration],
    "Crew Size": [crew_size],
    "Mission Risk Level": [risk_level]
})

# Apply encoding to categorical columns
for col in input_data.select_dtypes(include='object'):
    try:
        input_data[col] = encoder.transform(input_data[col])
    except:
        st.error(f"⚠️ Unknown value in {col}. Please enter valid category.")
        st.stop()

# Prediction
if st.button("Predict Mission Success 🚀"):
    prediction = model.predict(input_data)[0]
    result = "✅ Successful Mission" if prediction == 1 else "❌ Mission Likely to Fail"
    
    st.subheader("📢 Prediction Result:")
    st.success(result) if prediction == 1 else st.error(result)
