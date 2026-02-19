import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

csvfile = 'play_tennis_2400.csv'
df = pd.read_csv(csvfile)

if 'Day' in df.columns:
    df = df.drop(columns=['Day'])


encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


X = df.drop(columns=['Play Tennis'])
y = df['Play Tennis']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CategoricalNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


with open('tennis_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("Model and encoders saved successfully.")
print("Classes for 'Outlook':", encoders['Outlook'].classes_)
print("Classes for 'Temperature':", encoders['Temperature'].classes_)
print("Classes for 'Humidity':", encoders['Humidity'].classes_)
print("Classes for 'Wind':", encoders['Wind'].classes_)
print("Classes for 'Play Tennis':", encoders['Play Tennis'].classes_)
