import pandas as pd
import numpy as np
import random

# Set seed
random.seed(42)
np.random.seed(42)

# Create dataset where Outlook ALONE determines the outcome
# This removes the interaction problem for Naive Bayes

data = []

# Overcast -> Yes (100%)
for _ in range(400):
    temp = random.choice(['Hot', 'Mild', 'Cool'])
    humidity = random.choice(['High', 'Normal'])
    wind = random.choice(['Strong', 'Weak'])
    data.append(['Overcast', temp, humidity, wind, 'Yes'])

# Sunny -> Yes (95% of the time)
for _ in range(380):
    temp = random.choice(['Hot', 'Mild', 'Cool'])
    humidity = random.choice(['High', 'Normal'])
    wind = random.choice(['Strong', 'Weak'])
    data.append(['Sunny', temp, humidity, wind, 'Yes'])

# Sunny -> No (5% edge cases)
for _ in range(20):
    temp = random.choice(['Hot', 'Mild', 'Cool'])
    humidity = random.choice(['High', 'Normal'])
    wind = random.choice(['Strong', 'Weak'])
    data.append(['Sunny', temp, humidity, wind, 'No'])

# Rain -> No (95% of the time)
for _ in range(380):
    temp = random.choice(['Hot', 'Mild', 'Cool'])
    humidity = random.choice(['High', 'Normal'])
    wind = random.choice(['Strong', 'Weak'])
    data.append(['Rain', temp, humidity, wind, 'No'])

# Rain -> Yes (5% edge cases)
for _ in range(20):
    temp = random.choice(['Hot', 'Mild', 'Cool'])
    humidity = random.choice(['High', 'Normal'])
    wind = random.choice(['Strong', 'Weak'])
    data.append(['Rain', temp, humidity, wind, 'Yes'])

# Create DataFrame
df = pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play Tennis'])

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add Day column
df.insert(0, 'Day', [f'D{i+1}' for i in range(len(df))])

# Save
csv_path = 'play_tennis_2400.csv'
df.to_csv(csv_path, index=False)

print(f"Dataset created with {len(df)} rows")
print(f"\nClass distribution:")
print(df['Play Tennis'].value_counts())
print(f"\nOutlook vs Play Tennis:")
print(pd.crosstab(df['Outlook'], df['Play Tennis']))
print(f"\nDataset saved to {csv_path}")
