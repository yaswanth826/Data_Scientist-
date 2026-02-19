import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Predict the risk of heart disease using Logistic Regression")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("framingham_heart_disease.csv")
    return df

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Handle missing values
df = df.dropna()

X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Accuracy
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {acc:.2f}")

st.subheader("🧑 Enter Patient Details")

user_input = []
for col in X.columns:
    val = st.number_input(f"{col}", value=float(X[col].mean()))
    user_input.append(val)

user_data = np.array(user_input).reshape(1, -1)
user_data_scaled = scaler.transform(user_data)

if st.button("🔍 Predict"):
    prediction = model.predict(user_data_scaled)[0]
    prob = model.predict_proba(user_data_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Probability: {prob:.2f})")