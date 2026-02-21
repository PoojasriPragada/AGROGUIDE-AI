import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# 1. Load the data from your 'data' folder
# Make sure the filename matches exactly!
df = pd.read_csv('data/Crop_recommendation.csv')

# 2. Define Features (X) and Target (y)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# 3. Initialize and Train the Model
# We use RandomForest because it's highly accurate for agriculture data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Save the model into your 'models' folder
model_path = 'models/crop_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

print(f"🎉 Success! The AI model is trained and saved in {model_path}")