import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import random
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------------------------------------
st.set_page_config(page_title=" AI Space Mission", layout="wide", page_icon="🛰")

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
# LOAD DATASET
# ---------------------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv("Global_Space_Exploration_Dataset_With_Nulls-checkpoint.csv")
    return df

missions_df = load_dataset()

# ---------------------------------------------------------------------------------------
# STATIC MISSION DATA OPTIONS
# ---------------------------------------------------------------------------------------
countries = missions_df["Country"].dropna().unique().tolist()
mission_types = missions_df["Mission Type"].dropna().unique().tolist()
launch_sites = missions_df["Launch Site"].dropna().unique().tolist()
satellite_types = missions_df["Satellite Type"].dropna().unique().tolist()
technology_used_options = missions_df["Technology Used"].dropna().unique().tolist()

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
# STYLES
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
    font-size:50px;
    font-weight:800;
    background:linear-gradient(90deg,#00ffe6,#00c8ff);
    -webkit-background-clip:text;
    color:transparent;
}

.glass-card {
    background:rgba(255,255,255,0.07);
    backdrop-filter:blur(14px);
    border-radius:18px;
    padding:25px;
    border:1px solid rgba(255,255,255,0.15);
}

.footer {
    text-align:center;
    margin-top:40px;
    color:#9ffcff;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------------------
page = st.sidebar.radio(
    " Navigation",
    [" Home", " Mission Predictor", " Mission Control", " About", " Contact"]
)

# ---------------------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------------------
if page ==  Home":

    st.markdown("<div class='main-title'> AI Space Mission System</div>", unsafe_allow_html=True)
    st.write("### Explore global space missions using AI-powered analytics")

    # KPI METRICS
    total_missions = len(missions_df)
    total_countries = missions_df["Country"].nunique()
    avg_budget = missions_df["Budget (in Billion $)"].mean()
    avg_success = missions_df["Success Probability"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(" Missions", total_missions)
    c2.metric(" Countries", total_countries)
    c3.metric(" Avg Budget", f"{avg_budget:.2f} B$")
    c4.metric(" Avg Success", f"{avg_success:.1f}%")

    st.write("")

    st.subheader(" Global Mission Visualization")

    st.components.v1.html(
    """
    <iframe
    src="https://satellites.pro/3d-earth"
    width="100%"
    height="600"
    style="border:none;border-radius:20px;box-shadow:0 0 30px cyan;">
    </iframe>
    """,
    height=620
    )

    st.write("")

    st.subheader(" Latest Missions")

    recent = missions_df.sort_values("Year", ascending=False).head(10)

    st.dataframe(recent, use_container_width=True)

    st.markdown("<div class='footer'>✨ Built by PARIMI GANDHI BALAJI </div>", unsafe_allow_html=True)
       

# ---------------------------------------------------------------------------------------
# MISSION CONTROL DASHBOARD
# ---------------------------------------------------------------------------------------
if page == " Mission Control":

    st.markdown("<div class='main-title'>🛰 Mission Control Dashboard</div>", unsafe_allow_html=True)

    # KPIs
    total_missions = len(missions_df)
    avg_budget = missions_df["Budget (in Billion $)"].mean()
    avg_duration = missions_df["Duration (in Days)"].mean()
    avg_success = missions_df["Success Probability"].mean()

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(" Total Missions", total_missions)
    col2.metric(" Avg Budget ($B)", round(avg_budget,2))
    col3.metric(" Avg Duration", round(avg_duration,1))
    col4.metric(" Avg Success %", round(avg_success,1))

    st.write("")

    # Budget vs Success
    fig = px.scatter(
        missions_df,
        x="Budget (in Billion $)",
        y="Success Probability",
        color="Mission Risk Level",
        title="Budget vs Success Probability"
    )
    st.plotly_chart(fig,use_container_width=True)

    # Missions by country
    country_counts = missions_df["Country"].value_counts()

    fig2 = px.bar(
        country_counts,
        title="Global Missions by Country"
    )
    st.plotly_chart(fig2,use_container_width=True)

    # Risk distribution
    risk_counts = missions_df["Mission Risk Level"].value_counts()

    fig3 = px.pie(
        names=risk_counts.index,
        values=risk_counts.values,
        title="Mission Risk Distribution"
    )
    st.plotly_chart(fig3,use_container_width=True)

    # Technology usage
    tech_counts = missions_df["Technology Used"].value_counts().head(10)

    fig4 = px.bar(
        tech_counts,
        title="Top Space Technologies"
    )
    st.plotly_chart(fig4,use_container_width=True)

    st.subheader("📡 Recent Missions")
    st.dataframe(missions_df.head(20),use_container_width=True)

# ---------------------------------------------------------------------------------------
# MISSION PREDICTOR
# ---------------------------------------------------------------------------------------
if page == "🛰 Mission Predictor":

    st.markdown("<div class='main-title'> Space Mission Success Predictor</div>", unsafe_allow_html=True)

    if st.button("🎲 Generate Random Mission"):
        st.session_state.random_data = generate_random_values()

    random_values = st.session_state.random_data

    col1,col2 = st.columns(2)

    with col1:
        country = st.selectbox("Country",countries)
        mission_type = st.selectbox("Mission Type",mission_types)
        risk_level = st.selectbox("Risk Level",["Low","Medium","High"])
        crew_size = st.number_input("Crew Size",0,10)

    with col2:
        launch_site = st.selectbox("Launch Site",launch_sites)
        satellite_type = st.selectbox("Satellite Type",satellite_types)
        technology_used = st.selectbox("Technology Used",technology_used_options)

    duration = st.slider("Duration (Days)",1,1000)
    budget = st.slider("Budget (Billion $)",0.1,100.0)
    success_rate = st.slider("Historical Success Rate (%)",0,100)

    input_data = pd.DataFrame({
        "Country":[country],
        "Mission Type":[mission_type],
        "Launch Site":[launch_site],
        "Satellite Type":[satellite_type],
        "Budget (in Billion $)":[budget],
        "Success Rate (%)":[success_rate],
        "Technology Used":[technology_used],
        "Duration (in Days)":[duration],
        "Crew Size":[crew_size],
        "Mission Risk Level":[risk_level]
    })

    for col in input_data.select_dtypes(include="object"):
        if input_data[col][0] not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, input_data[col][0])
        input_data[col] = encoder.transform(input_data[col])

    if st.button(" Run Prediction"):

        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction]*100

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text':"Mission Success Probability"},
            gauge={'axis':{'range':[0,100]}}
        ))

        st.plotly_chart(fig,use_container_width=True)

        if prediction==1:
            st.success(f"Mission Likely Successful ({confidence:.2f}%)")
        else:
            st.error(f"Mission Failure Risk ({confidence:.2f}%)")

# ---------------------------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------------------------
if page == " About":

    st.markdown("<div class='main-title'>📘 About The Project</div>", unsafe_allow_html=True)

    st.write("""
AI system predicting the success probability of space missions using machine learning.

Features:

• Mission success prediction  
• Global mission dataset analytics  
• NASA-style mission control dashboard  
• Interactive visualization
""")

# ---------------------------------------------------------------------------------------
# CONTACT
# ---------------------------------------------------------------------------------------
if page == " Contact":

    st.markdown("<div class='main-title'>📡 Contact the Developer</div>", unsafe_allow_html=True)

    st.write("""
Developer: **Parimi Gandhi Balaji**

Email: **parimigandhibalaji@gmail.com**

GitHub: https://github.com/ParimiBalaji/space-mission-predictor
""")

# ---------------------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------------------
st.markdown("<br><div class='footer'>✨Built by PARIMI GANDHI BALAJI </div>", unsafe_allow_html=True)

