import pandas as pd
import joblib
import json
from sklearn.metrics import silhouette_score

# Load data test
test = pd.read_csv("data/prepared/test.csv")

# Load model
model = joblib.load("models/model.pkl")

# Prediksi cluster
labels = model.predict(test)

# Evaluasi (clustering metric)
score = silhouette_score(test, labels)

# Simpan hasil evaluasi
metrics = {
    "silhouette_score": float(score)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation selesai")