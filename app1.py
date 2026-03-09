import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Space Mission Platform",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("mission_success_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_artifacts()

# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv("Global_Space_Exploration_Dataset_With_Nulls-checkpoint.csv")
    return df

missions_df = load_dataset()

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Mission Predictor", "Mission Control", "About", "Contact"]
)

# ---------------------------------------------------------
# GLOBAL FILTERS
# ---------------------------------------------------------
st.sidebar.markdown("### Data Filters")

country_filter = st.sidebar.multiselect(
    "Country",
    missions_df["Country"].unique(),
    default=missions_df["Country"].unique()
)

risk_filter = st.sidebar.multiselect(
    "Mission Risk Level",
    missions_df["Mission Risk Level"].unique(),
    default=missions_df["Mission Risk Level"].unique()
)

tech_filter = st.sidebar.multiselect(
    "Technology Used",
    missions_df["Technology Used"].unique(),
    default=missions_df["Technology Used"].unique()
)

filtered_df = missions_df[
    (missions_df["Country"].isin(country_filter)) &
    (missions_df["Mission Risk Level"].isin(risk_filter)) &
    (missions_df["Technology Used"].isin(tech_filter))
]

# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
if page == "Home":

    st.title("AI Space Mission Analytics Platform")

    st.write("Explore global space mission data, analytics, and AI predictions.")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Missions", len(filtered_df))
    col2.metric("Countries", filtered_df["Country"].nunique())
    col3.metric("Average Budget (B$)", round(filtered_df["Budget (in Billion $)"].mean(),2))
    col4.metric("Average Success (%)", round(filtered_df["Success Probability"].mean(),1))

    st.divider()

    # -----------------------------------------------------
    # WORLD MAP OF LAUNCH SITES
    # -----------------------------------------------------
    st.subheader("Global Launch Sites")

    map_df = filtered_df.copy()

    if "Latitude" in map_df.columns and "Longitude" in map_df.columns:

        fig = px.scatter_geo(
            map_df,
            lat="Latitude",
            lon="Longitude",
            color="Country",
            hover_name="Mission Name",
            title="Launch Locations Around the World"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Dataset does not contain latitude/longitude columns.")

    st.divider()

    # -----------------------------------------------------
    # MISSIONS BY COUNTRY
    # -----------------------------------------------------
    st.subheader("Missions by Country")

    country_counts = filtered_df["Country"].value_counts()

    fig = px.bar(
        country_counts,
        title="Top Countries Running Space Missions"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # MISSIONS OVER TIME
    # -----------------------------------------------------
    st.subheader("Mission Timeline")

    year_counts = filtered_df["Year"].value_counts().sort_index()

    fig = px.line(
        x=year_counts.index,
        y=year_counts.values,
        labels={"x":"Year","y":"Number of Missions"},
        title="Space Missions Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # TECHNOLOGY USAGE
    # -----------------------------------------------------
    st.subheader("Technology Usage")

    tech_counts = filtered_df["Technology Used"].value_counts().head(10)

    fig = px.bar(
        tech_counts,
        title="Top Technologies Used in Space Missions"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # RECENT MISSIONS
    # -----------------------------------------------------
    st.subheader("Recent Missions")

    recent = filtered_df.sort_values("Year", ascending=False).head(10)

    st.dataframe(recent, use_container_width=True)

# ---------------------------------------------------------
# MISSION CONTROL DASHBOARD
# ---------------------------------------------------------
elif page == "Mission Control":

    st.title("Mission Control Dashboard")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            filtered_df,
            x="Budget (in Billion $)",
            y="Success Probability",
            color="Mission Risk Level",
            title="Budget vs Mission Success"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        risk_counts = filtered_df["Mission Risk Level"].value_counts()

        fig = px.pie(
            names=risk_counts.index,
            values=risk_counts.values,
            title="Mission Risk Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# MISSION PREDICTOR
# ---------------------------------------------------------
elif page == "Mission Predictor":

    st.title("Space Mission Success Predictor")

    countries = missions_df["Country"].dropna().unique()
    mission_types = missions_df["Mission Type"].dropna().unique()
    launch_sites = missions_df["Launch Site"].dropna().unique()
    satellite_types = missions_df["Satellite Type"].dropna().unique()
    technologies = missions_df["Technology Used"].dropna().unique()

    col1, col2 = st.columns(2)

    with col1:

        country = st.selectbox("Country", countries)
        mission_type = st.selectbox("Mission Type", mission_types)
        risk = st.selectbox("Mission Risk Level", ["Low","Medium","High"])
        crew = st.number_input("Crew Size",0,10)

    with col2:

        launch = st.selectbox("Launch Site", launch_sites)
        satellite = st.selectbox("Satellite Type", satellite_types)
        technology = st.selectbox("Technology Used", technologies)

    duration = st.slider("Mission Duration",1,1000)
    budget = st.slider("Mission Budget (B$)",0.1,100.0)
    success_rate = st.slider("Historical Success Rate",0,100)

    input_df = pd.DataFrame({
        "Country":[country],
        "Mission Type":[mission_type],
        "Launch Site":[launch],
        "Satellite Type":[satellite],
        "Budget (in Billion $)":[budget],
        "Success Rate (%)":[success_rate],
        "Technology Used":[technology],
        "Duration (in Days)":[duration],
        "Crew Size":[crew],
        "Mission Risk Level":[risk]
    })

    for col in input_df.select_dtypes(include="object"):
        if input_df[col][0] not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, input_df[col][0])
        input_df[col] = encoder.transform(input_df[col])

    if st.button("Run Prediction"):

        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df)[0][prediction] * 100

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={"text":"Mission Success Probability"},
            gauge={"axis":{"range":[0,100]}}
        ))

        st.plotly_chart(fig, use_container_width=True)

        if prediction == 1:
            st.success(f"Mission Likely Successful ({confidence:.2f}%)")
        else:
            st.error(f"Mission Failure Risk ({confidence:.2f}%)")

# ---------------------------------------------------------
# ABOUT
# ---------------------------------------------------------
elif page == "About":

    st.title("About This Project")

    st.write("""
This platform uses machine learning to predict the probability of success
for space missions using historical mission data.

Features included:

• AI mission success prediction  
• Global mission analytics dashboard  
• Technology trend analysis  
• Mission control visualization  
""")

# ---------------------------------------------------------
# CONTACT
# ---------------------------------------------------------
elif page == "Contact":

    st.title("Contact")

    st.write("""
Developer: Parimi Gandhi Balaji  

Email: parimigandhibalaji@gmail.com  

GitHub:  
https://github.com/ParimiBalaji/space-mission-predictor
""")
