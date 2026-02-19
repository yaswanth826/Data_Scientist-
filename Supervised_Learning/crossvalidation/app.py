import streamlit as st
import joblib
import numpy as np
import random

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")

# Load model & scaler
model = joblib.load("wine_model.pkl")
scaler = joblib.load("wine_scaler.pkl")

st.title("🍷 Wine Quality Prediction App")
st.write("Enter wine chemical properties to predict quality (0–10 scale).")

# Inputs
fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.0)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.5)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.5)
chlorides = st.slider("Chlorides", 0.01, 0.7, 0.08)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 75, 15)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 300, 46)
density = st.slider("Density", 0.9900, 1.0050, 0.9968)
pH = st.slider("pH", 2.5, 4.5, 3.3)
sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6)
alcohol = st.slider("Alcohol %", 8.0, 15.0, 10.0)

if st.button("🔮 Predict Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                            residual_sugar, chlorides, free_sulfur_dioxide,
                            total_sulfur_dioxide, density, pH, sulphates, alcohol]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"🍇 Predicted Wine Quality: {round(prediction[0], 2)} / 10")

# Random test button
if st.button("🎲 Generate Random Values & Predict"):
    rand_data = np.array([[
        random.uniform(4.0, 16.0),
        random.uniform(0.1, 1.6),
        random.uniform(0.0, 1.0),
        random.uniform(0.5, 15.0),
        random.uniform(0.01, 0.7),
        random.randint(1, 75),
        random.randint(6, 300),
        random.uniform(0.9900, 1.0050),
        random.uniform(2.5, 4.5),
        random.uniform(0.3, 2.0),
        random.uniform(8.0, 15.0)
    ]])

    rand_scaled = scaler.transform(rand_data)
    rand_pred = model.predict(rand_scaled)

    st.info("Random Input Used:")
    st.code(rand_data)

    st.success(f"🍷 Predicted Wine Quality (Random): {round(rand_pred[0], 2)} / 10")