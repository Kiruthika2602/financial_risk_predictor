import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
import pickle
import os

# Load data
df = pd.read_csv('data/dataset.csv')

# Initialize LabelEncoders
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Split features and label
X = df.drop('risk', axis=1)
y = df['risk']

# Train Naive Bayes model
model = CategoricalNB()
model.fit(X, y)

# Save model and encoders together
os.makedirs('model', exist_ok=True)
with open('model/risk_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'encoders': encoders}, f)
