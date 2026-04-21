import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ======================
# Setup folder
# ======================
os.makedirs("data/prepared", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ======================
# Load dataset
# ======================
df = pd.read_excel("data/raw/Online Retail.xlsx")

# ======================
# EDA (Exploratory Data Analysis)
# ======================

# 1. Statistik deskriptif
desc = df.describe()
desc.to_csv("plots/statistik.csv")

# 2. Cek missing value
missing = df.isnull().sum()
missing.to_csv("plots/missing_values.csv")

# 3. Ambil data numerik
df_numeric = df.select_dtypes(include='number')

# 4. Korelasi antar fitur numerik
corr = df_numeric.corr()
corr.to_csv("plots/correlation.csv")

# ======================
# Heatmap Korelasi (TAMBAHAN)
# ======================
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# ======================
# Preprocessing
# ======================

# Cleaning sederhana
df = df.dropna()

# Ambil ulang data numerik setelah cleaning
df_numeric = df.select_dtypes(include='number')

# Split data
train, test = train_test_split(df_numeric, test_size=0.2, random_state=42)

# ======================
# Simpan hasil
# ======================
train.to_csv("data/prepared/train.csv", index=False)
test.to_csv("data/prepared/test.csv", index=False)

# ======================
# Visualisasi distribusi
# ======================
df_numeric.hist(figsize=(10, 8))
plt.tight_layout()
plt.savefig("plots/data_distribution.png")
plt.close()

print("✅ Preparation + EDA selesai")