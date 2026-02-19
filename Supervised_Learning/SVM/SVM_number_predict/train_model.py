"""
Script to train and save the digit recognition model.
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import numpy as np

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
print("Training SVC model...")
model = SVC(kernel='rbf', C=10, gamma='scale')
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Save the model along with necessary info
model_package = {
    'model': model,
    'target_names': [str(i) for i in range(10)],
    'image_size': (8, 8),
    'n_features': 64
}

joblib.dump(model_package, 'digit_model.pkl')
print("Model saved as 'digit_model.pkl'")

