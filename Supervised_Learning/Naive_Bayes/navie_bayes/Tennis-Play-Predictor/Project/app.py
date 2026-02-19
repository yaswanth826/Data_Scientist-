import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="🎾 Tennis Play Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# ULTRA PREMIUM CSS STYLING
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Premium Dark Background */
    .stApp {
        background: linear-gradient(135deg, #0c0c1e 0%, #1a1a2e 50%, #16213e 100%);
        overflow-x: hidden;
    }
    
    /* Decorative Elements */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(ellipse at 10% 20%, rgba(120, 58, 237, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 90% 80%, rgba(236, 72, 153, 0.12) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(6, 182, 212, 0.08) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* Professional Card */
    .pro-card {
        position: relative;
        background: linear-gradient(145deg, rgba(26, 26, 46, 0.95) 0%, rgba(12, 12, 30, 0.98) 100%);
        border-radius: 28px;
        padding: 40px;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 
            0 25px 50px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        transition: all 0.4s ease;
    }
    
    .pro-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 35px 70px rgba(0, 0, 0, 0.5),
            0 0 60px rgba(120, 58, 237, 0.1);
        border-color: rgba(255, 255, 255, 0.12);
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 40px 0 30px 0;
    }
    
    .brand-title {
        font-size: 4.5rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 12px;
        margin-bottom: 15px;
        background: linear-gradient(135deg, #fff 0%, #a78bfa 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 30px rgba(167, 139, 250, 0.4));
    }
    
    .brand-tagline {
        color: rgba(255, 255, 255, 0.5);
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 6px;
        text-transform: uppercase;
    }
    
    /* About Section */
    .about-section {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(167, 139, 250, 0.05) 100%);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 20px;
        padding: 30px 35px;
        margin: 25px 0;
    }
    
    .about-title {
        color: #a78bfa;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .about-text {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.95rem;
        line-height: 1.8;
        font-weight: 400;
    }
    
    /* Section Header */
    .section-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 30px;
        padding-bottom: 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .section-icon {
        width: 50px;
        height: 50px;
        background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 10px 25px rgba(124, 58, 237, 0.35);
    }
    
    .section-text {
        flex: 1;
    }
    
    .section-title {
        color: #fff;
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .section-sub {
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.85rem;
        margin-top: 4px;
    }
    
    /* Input Labels */
    .stSelectbox label {
        color: rgba(255, 255, 255, 0.85) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover,
    .stSelectbox > div > div:focus-within {
        border-color: #7c3aed !important;
        box-shadow: 0 0 25px rgba(124, 58, 237, 0.25) !important;
    }
    
    /* Premium Button */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 18px !important;
        padding: 20px 45px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 3px !important;
        width: 100% !important;
        margin-top: 25px !important;
        box-shadow: 0 15px 35px rgba(124, 58, 237, 0.4) !important;
        transition: all 0.4s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 20px 45px rgba(124, 58, 237, 0.5) !important;
    }
    
    /* Result Display */
    .result-container {
        text-align: center;
        padding: 45px 25px;
        min-height: 400px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .result-icon {
        font-size: 6rem;
        margin-bottom: 20px;
        animation: gentleBounce 3s ease-in-out infinite;
    }
    
    @keyframes gentleBounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-15px); }
    }
    
    .result-yes {
        font-size: 4.5rem;
        font-weight: 900;
        letter-spacing: 15px;
        text-transform: uppercase;
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 35px rgba(52, 211, 153, 0.5));
    }
    
    .result-no {
        font-size: 4rem;
        font-weight: 900;
        letter-spacing: 12px;
        text-transform: uppercase;
        background: linear-gradient(135deg, #f87171 0%, #ef4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 35px rgba(248, 113, 113, 0.5));
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 12px 28px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 20px;
    }
    
    .badge-success {
        background: rgba(52, 211, 153, 0.12);
        border: 2px solid rgba(52, 211, 153, 0.35);
        color: #34d399;
    }
    
    .badge-danger {
        background: rgba(248, 113, 113, 0.12);
        border: 2px solid rgba(248, 113, 113, 0.35);
        color: #f87171;
    }
    
    .result-message {
        color: rgba(255, 255, 255, 0.5);
        font-size: 1.05rem;
        margin-top: 25px;
        line-height: 1.7;
    }
    
    /* Waiting State */
    .waiting-box {
        text-align: center;
        opacity: 0.5;
    }
    
    .waiting-icon {
        font-size: 5rem;
        margin-bottom: 20px;
        animation: float 4s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-12px); }
    }
    
    .waiting-title {
        color: rgba(255, 255, 255, 0.6);
        font-size: 1.4rem;
        font-weight: 600;
    }
    
    .waiting-sub {
        color: rgba(255, 255, 255, 0.35);
        font-size: 0.9rem;
        margin-top: 10px;
    }
    
    /* Team Section */
    .team-section {
        margin-top: 35px;
        padding-top: 25px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .team-label {
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 12px;
    }
    
    .team-names {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .team-names span {
        color: #a78bfa;
        font-weight: 600;
    }
    
    /* Footer */
    .pro-footer {
        text-align: center;
        padding: 45px 0 25px 0;
        margin-top: 50px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .footer-content {
        color: rgba(255, 255, 255, 0.35);
        font-size: 0.8rem;
        letter-spacing: 2px;
    }
    
    .footer-content span {
        color: #a78bfa;
        font-weight: 600;
    }
    
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL & ENCODERS
# ============================================
@st.cache_resource
def load_artifacts():
    try:
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'tennis_model.pkl')
        encoders_path = os.path.join(current_dir, 'encoders.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        return None, None

model, encoders = load_artifacts()

if not model or not encoders:
    st.error("⚠️ Model not found! Please run 'train_model.py' first.")
    st.stop()

# ============================================
# SESSION STATE
# ============================================
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'show_balloons' not in st.session_state:
    st.session_state.show_balloons = False

# ============================================
# MAIN INTERFACE
# ============================================

# Header
st.markdown("""
<div class="main-header">
    <div class="brand-title">TENNIS AI</div>
    <div class="brand-tagline">Gaussian Naive Bayes Prediction System</div>
</div>
""", unsafe_allow_html=True)

# About Section
st.markdown("""
<div class="about-section">
    <div class="about-title">📋 About This Project</div>
    <div class="about-text">
        This intelligent system uses <strong>Gaussian Naive Bayes</strong> machine learning algorithm to predict 
        whether weather conditions are suitable for playing tennis. The model is trained on a dataset of 2,400 
        weather observations, analyzing factors like outlook, temperature, humidity, and wind conditions to 
        provide accurate play recommendations. Built as an academic project demonstrating practical applications 
        of machine learning in decision support systems.
    </div>
</div>
""", unsafe_allow_html=True)

# Two Columns Layout
col1, col2 = st.columns([1, 1.2], gap="large")

# Left Column - Parameters
with col1:
    st.markdown("""
    <div class="pro-card">
        <div class="section-header">
            <div class="section-icon">⚙️</div>
            <div class="section-text">
                <div class="section-title">Input Parameters</div>
                <div class="section-sub">Configure weather conditions</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    outlook = st.selectbox("🌤️ Weather Outlook", encoders['Outlook'].classes_, key="outlook")
    temperature = st.selectbox("🌡️ Temperature", encoders['Temperature'].classes_, key="temp")
    humidity = st.selectbox("💧 Humidity Level", encoders['Humidity'].classes_, key="humidity")
    wind = st.selectbox("💨 Wind Condition", encoders['Wind'].classes_, key="wind")
    
    if st.button("🎾 PREDICT NOW"):
        with st.spinner("🔮 Analyzing conditions..."):
            time.sleep(1)
            
            input_df = pd.DataFrame({
                'Outlook': [outlook],
                'Temperature': [temperature],
                'Humidity': [humidity],
                'Wind': [wind]
            })
            
            for c in input_df.columns:
                input_df[c] = encoders[c].transform(input_df[c])
            
            pred_idx = model.predict(input_df)[0]
            pred_label = encoders['Play Tennis'].inverse_transform([pred_idx])[0]
            
            # Only show balloons if it's a new "Yes" prediction
            if pred_label == 'Yes' and st.session_state.prediction != 'Yes':
                st.session_state.show_balloons = True
            else:
                st.session_state.show_balloons = False
                
            st.session_state.prediction = pred_label
            st.rerun()
    
    # Team Credits
    st.markdown("""
    <div class="team-section">
        <div class="team-label">Model Created By</div>
        <div class="team-names">
            <span>Partha Sarathi R</span> • <span>Ayyapparaja VJ</span> • <span>Thirupathi Yaswanth</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Right Column - Results
with col2:
    st.markdown("""
    <div class="pro-card">
        <div class="section-header">
            <div class="section-icon">📊</div>
            <div class="section-text">
                <div class="section-title">Prediction Result</div>
                <div class="section-sub">AI-powered analysis output</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.prediction:
        result = st.session_state.prediction
        
        if result == 'Yes':
            st.markdown("""
            <div class="result-container">
                <div class="result-icon">🎾</div>
                <div class="result-yes">PLAY</div>
                <div class="status-badge badge-success">✓ Recommended</div>
                <div class="result-message">
                    Weather conditions are optimal for tennis.<br>
                    Enjoy your game!
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Only show balloons once per new Yes prediction
            if st.session_state.show_balloons:
                st.balloons()
                st.session_state.show_balloons = False
        else:
            st.markdown("""
            <div class="result-container">
                <div class="result-icon">🏠</div>
                <div class="result-no">NO PLAY</div>
                <div class="status-badge badge-danger">✗ Not Recommended</div>
                <div class="result-message">
                    Weather conditions are unfavorable.<br>
                    Consider indoor activities.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-container waiting-box">
            <div class="waiting-icon">🎯</div>
            <div class="waiting-title">Ready to Analyze</div>
            <div class="waiting-sub">Select parameters and click predict</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="pro-footer">
    <div class="footer-content">
        Machine Learning Project • <span>Gaussian Naive Bayes Algorithm</span> • Class of <span>2026</span>
    </div>
</div>
""", unsafe_allow_html=True)
