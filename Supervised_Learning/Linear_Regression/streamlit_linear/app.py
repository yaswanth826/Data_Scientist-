import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Coffee Sales Prediction App",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Coffee Sales Prediction App")
st.write("Predict coffee sales using Machine Learning")

# ---------------- FILE PATHS ----------------
MODEL_PATH = "coffee_sales_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_SELECTOR_PATH = "feature_selector.pkl"

# ---------------- FILE CHECK ----------------
missing_files = []

for file in [MODEL_PATH, SCALER_PATH, FEATURE_SELECTOR_PATH]:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    st.error(f"❌ Missing file(s): {', '.join(missing_files)}")
    st.stop()

# ---------------- LOAD MODELS ----------------
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_selector = joblib.load(FEATURE_SELECTOR_PATH)
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

st.success("✅ Model loaded successfully!")

# ---------------- USER INPUT ----------------
st.subheader("Enter Input Features")

# ⚠️ CHANGE feature names if your model expects different ones
feature_1 = st.number_input("Advertising Spend", min_value=0.0, step=1.0)
feature_2 = st.number_input("Store Footfall", min_value=0.0, step=1.0)
feature_3 = st.number_input("Price per Cup", min_value=0.0, step=0.5)

# Put inputs into DataFrame
input_df = pd.DataFrame([[feature_1, feature_2, feature_3]],
                        columns=["Advertising", "Footfall", "Price"])

# ---------------- PREDICTION ----------------
if st.button("Predict Coffee Sales ☕"):
    try:
        # Scale input
        scaled_input = scaler.transform(input_df)

        # Feature selection
        selected_input = feature_selector.transform(scaled_input)

        # Prediction
        prediction = model.predict(selected_input)

        st.success(f"📈 Predicted Coffee Sales: **{prediction[0]:.2f} units**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
