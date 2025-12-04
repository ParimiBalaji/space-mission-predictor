import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import random


# ---------------------------------------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------------------------------------
st.set_page_config(page_title="🚀 AI Space Mission", layout="wide", page_icon="🛰")


# ---------------------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    return (
        joblib.load("mission_success_model.pkl"),
        joblib.load("label_encoder.pkl")
    )


model, encoder = load_artifacts()


# ---------------------------------------------------------------------------------------
# STATIC MISSION DATA OPTIONS
# ---------------------------------------------------------------------------------------
countries = ["USA", "Russia", "India", "China", "Japan", "France", "UK",
             "Germany", "Israel", "South Korea", "UAE", "Brazil", "Other"]

mission_types = ["Orbiter", "Lander", "Rover", "Flyby", "Test", "Satellite Deployment",
                 "Space Station Mission", "Crewed Mission", "Deep Space Probe", "Other"]

launch_sites = ["Cape Canaveral (USA)", "Vandenberg (USA)", "Baikonur (Russia)",
                "Vostochny (Russia)", "Satish Dhawan (India)", "Wenchang (China)",
                "Jiuquan (China)", "Tanegashima (Japan)", "Guiana Space Centre (EU)",
                "Mahia Peninsula (New Zealand)", "Other"]

satellite_types = ["Communication", "Navigation", "Military / Surveillance", "Weather",
                   "Earth Observation", "Deep Space", "Space Telescope",
                   "CubeSat / NanoSat", "Science Research", "Other"]

technology_used_options = [
    "Cryogenic Engine", "Solid Rocket Motor", "Liquid Fuel Engine",
    "Reusable Booster System", "Ion Thruster", "Hybrid Propulsion",
    "Nuclear Thermal Propulsion", "Solar Sail Technology",
    "Autonomous Guidance System",
    "Reusable Spacecraft Technology (Falcon / Starship)",
    "Other"
]


# ---------------------------------------------------------------------------------------
# RANDOM MISSION GENERATOR
# ---------------------------------------------------------------------------------------
def generate_random_values():
    return {
        "country": random.choice(countries),
        "mission_type": random.choice(mission_types),
        "risk_level": random.choice(["Low", "Medium", "High"]),
        "crew_size": random.randint(0, 10),
        "launch_site": random.choice(launch_sites),
        "satellite_type": random.choice(satellite_types),
        "technology_used": random.choice(technology_used_options),
        "duration": random.randint(1, 1000),
        "budget": round(random.uniform(0.1, 100.0), 2),
        "success_rate": random.randint(0, 100),
    }


if "random_data" not in st.session_state:
    st.session_state.random_data = None


# ---------------------------------------------------------------------------------------
# STYLES (PREMIUM UI)
# ---------------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;800&family=Montserrat:wght@300;500;700&display=swap');

*{font-family:'Montserrat',sans-serif;}

.stApp {
    background: linear-gradient(135deg, rgba(0,0,0,0.92), rgba(10,15,30,0.97)),
                url('background.jpg') no-repeat center fixed;
    background-size: cover;
}

.main-title {
    text-align:center;
    font-family:'Orbitron';
    font-size:55px;
    font-weight:800;
    background:linear-gradient(90deg,#00ffe6,#00c8ff);
    -webkit-background-clip:text;
    color:transparent;
    animation:glow 2s infinite alternate;
}
@keyframes glow{
    0%{text-shadow:0 0 10px cyan;}
    100%{text-shadow:0 0 35px #00ffe6;}
}

.glass-card {
    background:rgba(255,255,255,0.07);
    backdrop-filter:blur(14px);
    border-radius:18px;
    padding:25px;
    border:1px solid rgba(255,255,255,0.15);
    box-shadow:0 0 20px rgba(0,255,255,0.25);
}

div.stButton > button {
    width:100%;
    font-size:20px;
    padding:12px;
    font-weight:700;
    border-radius:50px;
    background:linear-gradient(90deg,#007bff,#00ffe1);
    border:2px solid rgba(255,255,255,0.3);
    box-shadow:0 0 12px cyan;
    transition:0.3s;
}

div.stButton > button:hover {
    transform:scale(1.06);
    box-shadow:0 0 25px #00ffe6;
}

.footer {
    text-align:center;
    margin-top:40px;
    color:#9ffcff;
}
</style>
""", unsafe_allow_html=True)




# ---------------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------------------
page = st.sidebar.radio("📂 Navigation", ["🏠 Home", "🛰 Mission Predictor", "📘 About", "📞 Contact"])


# ---------------------------------------------------------------------------------------
# HOME (3D Earth)
# ---------------------------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown("<div class='main-title'>🌍 AI Space Mission System</div>", unsafe_allow_html=True)
    st.write("### Explore predictions, mission history, and more — using advanced AI 🚀")

    st.components.v1.html(
        """
        <div style='display:flex;justify-content:center;margin-top:20px;'>
        <iframe src="https://earth.nullschool.net/#current/wind/surface/level/orthographic=-90.00,0,620"
        width="900" height="500" style="border:none;border-radius:20px;box-shadow:0 0 30px cyan;"></iframe>
        </div>
        """,
        height=520
    )

    st.markdown("<div class='footer'>🚀 Built for Future Space Engineers</div>", unsafe_allow_html=True)



# ---------------------------------------------------------------------------------------
# MISSION PREDICTOR PAGE
# ---------------------------------------------------------------------------------------
if page == "🛰 Mission Predictor":

    st.markdown("<div class='main-title'>🚀 Space Mission Success Predictor</div>", unsafe_allow_html=True)
    st.write("### AI-powered prediction based on global space mission patterns.")

    st.write("")
    if st.button("🎲 Generate Random Mission"):
        st.session_state.random_data = generate_random_values()

    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

        random_values = st.session_state.random_data
        col1, col2 = st.columns(2)

        with col1:
            country = st.selectbox("🌍 Country", countries,
                                   index=countries.index(random_values["country"]) if random_values else 0)
            mission_type = st.selectbox("🛰 Mission Type", mission_types,
                                        index=mission_types.index(random_values["mission_type"]) if random_values else 0)
            risk_level = st.selectbox("⚠ Mission Risk Level", ["Low", "Medium", "High"],
                                      index=["Low", "Medium", "High"].index(
                                          random_values["risk_level"]
                                      ) if random_values else 0)
            crew_size = st.number_input("👨‍🚀 Crew Size", 0, 10,
                                        value=random_values["crew_size"] if random_values else 0)

        with col2:
            launch_site = st.selectbox("🚀 Launch Site", launch_sites,
                                       index=launch_sites.index(random_values["launch_site"]) if random_values else 0)
            satellite_type = st.selectbox("🛰 Satellite Type", satellite_types,
                                          index=satellite_types.index(random_values["satellite_type"]) if random_values else 0)
            technology_used = st.selectbox("🔧 Technology Used", technology_used_options,
                                           index=technology_used_options.index(
                                               random_values["technology_used"]
                                           ) if random_values else 0)

        duration = st.slider("⏳ Mission Duration (Days)", 1, 1000,
                             value=random_values["duration"] if random_values else 120)

        budget = st.slider("💰 Budget (Billion $)", 0.1, 100.0,
                           value=random_values["budget"] if random_values else 1.5)

        success_rate = st.slider("📊 Historical Success Rate (%)", 0, 100,
                                 value=random_values["success_rate"] if random_values else 70)

        input_data = pd.DataFrame({
            "Country": [country], "Mission Type": [mission_type],
            "Launch Site": [launch_site], "Satellite Type": [satellite_type],
            "Budget (in Billion $)": [budget], "Success Rate (%)": [success_rate],
            "Technology Used": [technology_used], "Duration (in Days)": [duration],
            "Crew Size": [crew_size], "Mission Risk Level": [risk_level]
        })

        for col in input_data.select_dtypes(include='object'):
            if input_data[col][0] not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, input_data[col][0])
            input_data[col] = encoder.transform(input_data[col])

        st.write("")
        if st.button("🔍 Run Prediction"):
            with st.spinner("🛰 Running mission simulation..."):
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

            with st.expander("📄 Mission Summary"):
                st.dataframe(input_data)

        st.markdown("</div>", unsafe_allow_html=True)



# ---------------------------------------------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------------------------------------------
if page == "📘 About":
    st.markdown("<div class='main-title'>📘 About The Project</div>", unsafe_allow_html=True)
    st.write("""
This AI application predicts the success of space missions using real global space mission patterns.

### 🔍 Features:
- AI-based machine learning predictions  
- Mission scenario randomizer  
- Interactive UI with 3D Earth model  
- Future-ready space analytics  
""")


# ---------------------------------------------------------------------------------------
# CONTACT PAGE
# ---------------------------------------------------------------------------------------
if page == "📞 Contact":
    st.markdown("<div class='main-title'>📡 Contact the Developer</div>", unsafe_allow_html=True)
    st.write("""
💡 Developed by: **Parimi Gandhi Balaji**

📧 Email: **parimibalaji@gmail.com**

🌍 GitHub/Portfolio: *https://github.com/ParimiBalaji/space-mission-predictor*

🚀 Available for freelance & collaboration in  
AI • Space Technology • Web Engineering • ML Systems
""")


# ---------------------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------------------
st.markdown("<br><div class='footer'>✨Built by PARIMI GANDHI BALAJI 🚀</div>", unsafe_allow_html=True)
