import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# ---------------- Load Dataset ----------------
# Use mushroom dataset CSV file
df = pd.read_csv("mushrooms.csv")

# ---------------- Encode Categorical Data ----------------
label_encoders = {}

for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# ---------------- Split Features & Target ----------------
X = df.drop("class", axis=1)
y = df["class"]

# ---------------- Train Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Train Gradient Boosting ----------------
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# ---------------- Accuracy ----------------
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# ---------------- Save Model ----------------
joblib.dump(model, "gradient_boosting_model.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("✅ Model and feature_columns saved successfully!")