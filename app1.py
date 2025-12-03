import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Space Mission Success Predictor",
    page_icon="🚀",
    layout="wide"
)

# ---------------------- LOAD MODEL & ENCODER ----------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("mission_success_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_artifacts()

# ---------------------- STATIC OPTIONS ----------------------
countries = [
    "USA", "Russia", "India", "China", "Japan", "France", "UK",
    "Germany", "Israel", "South Korea", "UAE", "Brazil", "Other"
]

mission_types = [
    "Orbiter", "Lander", "Rover", "Flyby", "Test", "Satellite Deployment",
    "Space Station Mission", "Crewed Mission", "Deep Space Probe", "Other"
]

launch_sites = [
    "Cape Canaveral (USA)", "Vandenberg (USA)", "Baikonur (Russia)",
    "Vostochny (Russia)", "Satish Dhawan (India)", "Wenchang (China)",
    "Jiuquan (China)", "Tanegashima (Japan)", "Guiana Space Centre (EU)",
    "Mahia Peninsula (New Zealand)", "Other"
]

satellite_types = [
    "Communication", "Navigation", "Military / Surveillance", "Weather",
    "Earth Observation", "Deep Space", "Space Telescope", "CubeSat / NanoSat",
    "Science Research", "Other"
]

technology_used_options = [
    "Cryogenic Engine", "Solid Rocket Motor", "Liquid Fuel Engine",
    "Reusable Booster System", "Ion Thruster", "Hybrid Propulsion",
    "Nuclear Thermal Propulsion", "Solar Sail Technology",
    "Autonomous Guidance System",
    "Reusable Spacecraft Technology (Falcon / Starship)",
    "Other"
]

# ---------------------- UI BACKGROUND STYLING ----------------------
st.markdown(
    """
    <style>
    .stApp {
        background: url('background.jpg') no-repeat center fixed;
        background-size: cover;
    }
    .overlay {
        position:absolute;
        top:0; left:0;
        height:100%; width:100%;
        background:rgba(0,0,0,0.50);
        backdrop-filter:blur(8px);
        z-index:-1;
    }
    .main-title {
        font-size: 45px;
        text-align:center;
        font-weight: 800;
        color:white;
        margin-top: 10px;
        animation: glow 2s infinite alternate;
    }
    @keyframes glow {
        0% { text-shadow: 0 0 6px #00eaff; }
        100% { text-shadow: 0 0 20px #00ffe6; }
    }
    .glass-card {
        background: rgba(255,255,255,0.08);
        padding: 22px;
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 14px;
        backdrop-filter: blur(12px);
        margin-bottom: 15px;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #0099ff, #00ffe6);
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
    }
    </style>
    <div class="overlay"></div>
    """,
    unsafe_allow_html=True
)

# ---------------------- HEADER ----------------------
st.markdown("<div class='main-title'>🚀 Space Mission Success Predictor</div>", unsafe_allow_html=True)
st.write("### AI-powered prediction based on global space mission patterns.")

# ---------------------- USER INPUT SECTION ----------------------
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("🌍 Country", countries)
        mission_type = st.selectbox("🛰 Mission Type", mission_types)
        risk_level = st.selectbox("⚠ Mission Risk Level", ["Low", "Medium", "High"])
        crew_size = st.number_input("👨‍🚀 Crew Size", min_value=0, step=1)

    with col2:
        launch_site = st.selectbox("🚀 Launch Site", launch_sites)
        satellite_type = st.selectbox("🛰 Satellite Type", satellite_types)
        technology_used = st.selectbox("🔧 Technology Used", technology_used_options)

    # ---- Custom Input if "Other" ----
    if launch_site == "Other":
        launch_site = st.text_input("✏ Enter Custom Launch Site")

    if satellite_type == "Other":
        satellite_type = st.text_input("✏ Enter Custom Satellite Type")

    if technology_used == "Other":
        technology_used = st.text_input("✏ Enter Custom Technology Used")

    # ---- Sliders ----
    duration = st.slider(
        "⏳ Mission Duration (Days)",
        min_value=1, max_value=1000,
        value=120, step=1
    )

    budget = st.slider(
        "💰 Budget (Billion $)",
        min_value=0.1, max_value=100.0,
        value=1.5, step=0.1
    )

    success_rate = st.slider(
        "📊 Previous Historical Success Rate (%)",
        min_value=0, max_value=100,
        value=70, step=1
    )

    # ---------------------- DATAFRAME for MODEL ----------------------
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

    # ------- FIX ENCODER NEW CATEGORIES -------
    for col in input_data.select_dtypes(include='object'):
        if input_data[col][0] not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, input_data[col][0])
        input_data[col] = encoder.transform(input_data[col])

    # ---------------------- RUN PREDICTION BUTTON ----------------------
    st.write("")  
    run = st.button("🔍 Run Prediction")

    if run:

        with st.spinner("🛰 Running Mission Simulation..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction] * 100

        if prediction == 1:
            st.balloons()
            st.success(f"🟢 Mission Likely Successful — Confidence: {confidence:.2f}%")
        else:
            st.snow()
            st.error(f"🔴 Mission Failure Risk — Confidence: {confidence:.2f}%")

        st.write("")

        with st.expander("📄 Mission Summary Details:"):
            st.dataframe(input_data)

    st.markdown("</div>", unsafe_allow_html=True)
