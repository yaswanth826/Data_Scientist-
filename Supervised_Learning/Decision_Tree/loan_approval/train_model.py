# Install if needed
# !pip install scikit-learn joblib

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    "Feature1": [1,2,3,4,5,6,7,8,9,10],
    "Feature2": [50,55,60,65,70,75,80,85,90,95],
    "Feature3": [5,4,3,6,7,8,2,9,1,10],
    "Loan_Status": [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train_pca, y_train)

# Accuracy
y_pred = model.predict(X_test_pca)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save files
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

print("Model, Scaler and PCA saved successfully!")