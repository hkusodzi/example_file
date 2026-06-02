import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from joblib import load

st.title("Crop Yield Prediction App")

model = load('my_model.joblib')

df = pd.read_csv("classification_data.csv")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Rainfall",
    f"{df['Rainfall_mm'].mean():.2f}"
)

col2.metric(
    "Average Temperature",
    f"{df['Temperature_C'].mean():.2f}"
)

col3.metric(
    "Total Records",
    len(df)
)
region = st.sidebar.selectbox(
    "Region",
    ["Northern", "Western", "Eastern", "Central"]
)

crop_type = st.sidebar.selectbox(
    "Crop Type",
    ["Maize", "Cassava", "Rice", "Wheat"]
)

rainfall = st.sidebar.slider(
    "Rainfall (mm)",
    0.0, 500.0, 150.0
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    0.0, 50.0, 25.0
)

soil_moisture = st.sidebar.slider(
    "Soil Moisture",
    0.0, 100.0, 40.0
)

fertilizer = st.sidebar.slider(
    "Fertilizer Used",
    0.0, 500.0, 100.0
)

pest_presence = st.sidebar.slider(
    "Pest Presence",
    0, 1, 0
)

day = st.sidebar.slider(
    "Day",
    1, 31, 15
)


input_data = pd.DataFrame({
    'Region': [region],
    'Crop_Type': [crop_type],
    'Rainfall_mm': [rainfall],
    'Temperature_C': [temperature],
    'Soil_Moisture': [soil_moisture],
    'Fertilizer_Used': [fertilizer],
    'Pest_Presence': [pest_presence],
    'Day': [day]
})
if st.button("Predict Yield"):

    prediction = model.predict(input_data)

    st.success(f"Predicted Yield: {prediction[0]:.2f} tonnes")


st.subheader("Dashboard Visualizations")

col1, col2 = st.columns(2)
with col1:

    # Visualization 1
    st.subheader("Crop Type Distribution")

    fig1, ax1 = plt.subplots()

    df['Crop_Type'].value_counts().plot(
        kind='bar',
        ax=ax1
    )

    st.pyplot(fig1)


    # Visualization 2
    st.subheader("Rainfall Distribution")

    fig2, ax2 = plt.subplots()

    ax2.hist(df['Rainfall_mm'])

    ax2.set_xlabel("Rainfall (mm)")
    ax2.set_ylabel("Frequency")

    st.pyplot(fig2)


with col2:

    # Visualization 3
    st.subheader("Regional Distribution")

    fig3, ax3 = plt.subplots()

    df['Region'].value_counts().plot(
        kind='bar',
        ax=ax3
    )

    st.pyplot(fig3)


    # Visualization 4
    st.subheader("Temperature vs Soil Moisture")

    fig4, ax4 = plt.subplots()

    ax4.scatter(
        df['Temperature_C'],
        df['Soil_Moisture']
    )

    ax4.set_xlabel("Temperature")
    ax4.set_ylabel("Soil Moisture")

    st.pyplot(fig4)

