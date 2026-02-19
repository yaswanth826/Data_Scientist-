import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Sample dataset
data = {
    "Income": [40000, 25000, 50000, 20000, 60000, 30000, 45000],
    "CreditScore": [720, 650, 780, 600, 800, 690, 710],
    "HasJob": [1, 1, 1, 0, 1, 0, 1],  # 1 = Yes, 0 = No
    "Approved": [1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["Income", "CreditScore", "HasJob"]]
y = df["Approved"]

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")