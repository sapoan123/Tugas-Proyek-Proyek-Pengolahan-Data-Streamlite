import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os

# Pastikan folder models ada
os.makedirs("models", exist_ok=True)

# Load data
train = pd.read_csv("data/prepared/train.csv")

# Model KMeans
model = KMeans(n_clusters=3, random_state=42)
model.fit(train)

# Simpan model
joblib.dump(model, "models/model.pkl")

print("Training selesai")