import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Loan Predictor", layout="centered")

# ------------------ CSS ------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Animated Background */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c1c);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Title */
h1 {
    text-align: center;
    color: white;
    font-weight: 700;
    letter-spacing: 1px;
}

/* Input fields */
.stNumberInput > div > div > input {
    border-radius: 10px;
    padding: 10px;
    border: none;
    transition: 0.3s;
}

.stNumberInput > div > div > input:focus {
    box-shadow: 0px 0px 10px #00f5ff;
    transform: scale(1.03);
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #00f5ff, #0084ff);
    color: white;
    font-weight: 600;
    border-radius: 30px;
    padding: 10px 30px;
    border: none;
    transition: all 0.4s ease;
}

.stButton > button:hover {
    transform: scale(1.08);
    background: linear-gradient(90deg, #0084ff, #00f5ff);
    box-shadow: 0px 0px 15px #00f5ff;
}

/* Result cards */
.result-success {
    background: rgba(0,255,100,0.2);
    padding: 15px;
    border-radius: 12px;
    color: white;
    font-size: 18px;
    text-align: center;
    margin-top: 20px;
    animation: fadeIn 1s ease-in-out;
}

.result-fail {
    background: rgba(255,0,80,0.2);
    padding: 15px;
    border-radius: 12px;
    color: white;
    font-size: 18px;
    text-align: center;
    margin-top: 20px;
    animation: fadeIn 1s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# ------------------ Load Model ------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# ------------------ UI ------------------
st.title("🚀 Smart Loan Approval System")

st.markdown("Enter applicant features below and get instant decision.")

f1 = st.number_input("Income")
f2 = st.number_input("Credit_Score")
f3 = st.number_input("Loan_Amount")

if st.button("Predict"):
    new_data = np.array([[f1, f2, f3]])
    new_scaled = scaler.transform(new_data)
    new_pca = pca.transform(new_scaled)
    prediction = model.predict(new_pca)

    if prediction[0] == 1:
        st.markdown('<div class="result-success">🎉 Loan Approved Successfully!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-fail">❌ Loan Rejected</div>', unsafe_allow_html=True)