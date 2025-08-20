import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Flask API URL
API_URL = "http://127.0.0.1:8000/predict"

# Unity WebGL Hosted via Ngrok (Ensure it's active)
UNITY_URL = "https://c7c7-128-6-147-101.ngrok-free.app"

# Load dataset
dataset = pd.read_csv("Yonghe_Modal_FDD_with_headers.csv")
mode3_all = dataset.iloc[2].values[1:].astype(float)

# Define session state for storing predictions
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# --- UI SETUP ---
st.title("🚆 Structural Health Monitoring: Portal North Bridge")
st.markdown("### 🌉 Digital Twin & Anomaly Detection for Wind-Induced Vibrations")

# --- UNITY WEBGL SIMULATION ---
st.subheader("🕹️ Live Unity Simulation")
st.markdown(f"""
    <iframe src="{UNITY_URL}" width="100%" height="600px" frameborder="0"></iframe>
""", unsafe_allow_html=True)

# --- INPUT SECTION ---
st.sidebar.header("📡 Input Live Data")
frequency = st.sidebar.slider("🔄 Simulated Cable Frequency (Hz)", 0.8, 1.2, 1.0, 0.01)
wind_strength = st.sidebar.slider("💨 Simulated Wind Strength", 0, 10, 2)

# --- SEND REQUEST TO FLASK SERVER ---
if st.sidebar.button("📊 Predict Anomaly"):
    response = requests.post(API_URL, data={"frequency": frequency, "wind_strength": wind_strength})
    if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        st.session_state.predictions.append((result["frequency"], result["wind_strength"], prediction))
    else:
        st.sidebar.error("⚠️ Error communicating with the server!")

# --- DISPLAY RESULTS ---
st.header("🔍 Live Anomaly Detection")
col1, col2 = st.columns(2)

with col1:
    # Status Box
    if st.session_state.predictions:
        last_prediction = st.session_state.predictions[-1][2]
        if last_prediction == -1:
            st.error("⚠️ **Anomaly Detected!** High risk of structural failure.")
        else:
            st.success("✅ **Structure is stable.** No anomalies detected.")

with col2:
    # Display Wind & Frequency
    st.metric("Current Frequency (Hz)", frequency)
    st.metric("Wind Strength", wind_strength)

# --- HISTORICAL DATA VISUALIZATION ---
st.subheader("📈 Mode 3 Frequency History")
fig = go.Figure()
fig.add_trace(go.Scatter(y=mode3_all[:192], mode="lines", name="Undamaged", line=dict(color="green")))
fig.add_trace(go.Scatter(y=mode3_all[192:], mode="lines", name="Damaged", line=dict(color="red")))
fig.update_layout(title="Mode 3 Frequency Over Time", xaxis_title="Sample", yaxis_title="Frequency (Hz)")
st.plotly_chart(fig)

# --- LIVE DATA LOG ---
st.subheader("📜 Live Data Log")
df = pd.DataFrame(st.session_state.predictions, columns=["Frequency (Hz)", "Wind Strength", "Anomaly (1=Normal, -1=Fault)"])
st.dataframe(df)

st.markdown("📌 **Notes:** If the frequency decreases significantly, the system will flag it as an anomaly.")
