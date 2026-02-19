import streamlit as st
import joblib
import numpy as np
import time

# Load Model
model = joblib.load("xgboost_house_price_model.pkl")

# Page Config
st.set_page_config(page_title="Property Valuator", page_icon="🏢", layout="wide")

# Dark Professional CSS - Clean & High Contrast
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF;
    }

    /* Deep Solid Background */
    .stApp {
        background-color: #0b0f1a;
    }

    /* High Visibility Heading */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: -40px;
        padding-bottom: 20px;
        border-bottom: 1px solid #1e293b;
    }

    /* Form Container */
    .stForm {
        background: #161b2c !important;
        border: 1px solid #2d3748 !important;
        border-radius: 12px !important;
        padding: 30px !important;
        margin-top: 20px;
    }

    /* Subheadings */
    h3 {
        color: #6366f1 !important;
        font-size: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 15px !important;
    }

    /* Input Styling */
    label p {
        color: #cbd5e1 !important;
        font-size: 0.95rem !important;
    }
    
    div[data-baseweb="input"] {
        background-color: #0b0f1a !important;
        border: 1px solid #334155 !important;
    }

    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: #6366f1 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: #4f46e5 !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
    }

    /* Result Block */
    .result-container {
        background: #1e293b;
        border-left: 5px solid #6366f1;
        padding: 25px;
        border-radius: 8px;
        margin-top: 25px;
        text-align: left;
    }
    .result-label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        margin: 0;
    }
    .result-value {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<h1 class='main-header'>House Price Prediction</h1>", unsafe_allow_html=True)

with st.form("valuation_engine"):
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("Property Dimensions")
        GrLivArea = st.number_input("Living Area (sq ft)", 100, 5000, 1500)
        TotalBsmtSF = st.number_input("Basement Area (sq ft)", 0, 3000, 800)
        LotArea = st.number_input("Lot Area (sq ft)", 1000, 50000, 8000)
        OverallQual = st.slider("Overall Quality (1–10)", 1, 10, 7)

    with col2:
        st.subheader("Structure & Features")
        BedroomAbvGr = st.number_input("Bedrooms", 1, 10, 3)
        FullBath = st.number_input("Bathrooms", 1, 5, 2)
        GarageCars = st.number_input("Garage Capacity", 0, 5, 2)
        YearBuilt = st.number_input("Year Built", 1800, 2025, 2005)

    submit = st.form_submit_button("Run Prediction")

if submit:
    X = np.array([[GrLivArea, BedroomAbvGr, FullBath,
                   TotalBsmtSF, GarageCars, YearBuilt,
                   LotArea, OverallQual]])
    
    prediction = model.predict(X)[0]

    st.markdown(f"""
        <div class='result-container'>
            <p class='result-label'>Estimated Market Value</p>
            <div class='result-value'>₹ {prediction:,.2f}</div>
            <p style='color: #6366f1; font-size: 0.85rem; margin:0;'>Calculated based on current model parameters</p>
        </div>
    """, unsafe_allow_html=True)