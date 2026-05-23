import pandas as pd
import os

def load_data(path="data/raw/Online Retail.xlsx"):
    print("Loading dataset...")
    df = pd.read_excel(path)
    print(f"Data shape: {df.shape}")
    return df


def preprocess_data(df):
    print("\n=== PREPROCESSING ===")

    # Drop missing value
    df = df.dropna()

    # Hapus data tidak valid
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    # Feature engineering
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    print(f"Data setelah preprocessing: {df.shape}")

    return df


def save_data(df):
    print("\nMenyimpan data hasil preprocessing...")
    os.makedirs("processed", exist_ok=True)
    df.to_csv("processed/data_clean.csv", index=False)
    print("Data disimpan di processed/data_clean.csv")


if __name__ == "__main__":
    df = load_data()
    df_clean = preprocess_data(df)
    save_data(df_clean)