import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "wine_model.pkl")
joblib.dump(scaler, "wine_scaler.pkl")

print("✅ Model & scaler saved successfully!")