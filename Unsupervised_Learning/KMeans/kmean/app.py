import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Mall Customer Segmentation", page_icon="📊", layout="centered")

# Modern Aesthetic CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ecf0f1;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 20px;
        color: #ecf0f1;
    }

    .sub-title {
        text-align: center;
        font-size: 16px;
        color: #bdc3c7;
        margin-bottom: 40px;
    }

    label {
        color: #ecf0f1 !important;
        font-weight: 600;
    }

    .stNumberInput input {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #ecf0f1;
        padding: 10px;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px;
        font-weight: 600;
        font-size: 16px;
        transition: 0.3s ease;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        background: linear-gradient(135deg, #0072ff, #00c6ff);
    }

    div[data-testid="stSuccess"] {
        background: rgba(46, 204, 113, 0.15);
        border-left: 4px solid #2ecc71;
        border-radius: 10px;
        padding: 16px;
    }

    div[data-testid="stInfo"] {
        background: rgba(52, 152, 219, 0.15);
        border-left: 4px solid #3498db;
        border-radius: 10px;
        padding: 16px;
    }

    .footer {
        text-align: center;
        color: #95a5a6;
        margin-top: 40px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main-title'>Mall Customer Segmentation</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>K-Means Clustering | Customer Behavior Analysis</div>", unsafe_allow_html=True)

# Inputs
col1, col2 = st.columns(2)

with col1:
    annual_income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=60)

with col2:
    spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

st.write("")

# Predict
if st.button("Predict Customer Segment"):
    input_data = np.array([[annual_income, spending_score]])
    input_scaled = scaler.transform(input_data)
    cluster = model.predict(input_scaled)[0]

    cluster_info = {
        0: "Low Spending - High Income",
        1: "High Spending - High Income",
        2: "Low Spending - Low Income",
        3: "High Spending - Low Income",
        4: "Average Customers"
    }

    st.success(f"Predicted Cluster: {cluster}")
    st.info(f"Segment Description: {cluster_info.get(cluster, 'Unknown Segment')}")

# Footer
st.markdown("<div class='footer'>Built with Streamlit | K-Means Clustering</div>", unsafe_allow_html=True)