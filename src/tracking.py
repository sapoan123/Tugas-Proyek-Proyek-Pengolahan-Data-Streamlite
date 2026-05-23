import pandas as pd
import os
from datetime import datetime

def load_raw_data():
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, "..", "data/raw/Online Retail.xlsx")
    return pd.read_excel(path)

def load_processed_data():
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, "..", "processed", "data_clean.csv")
    return pd.read_csv(path)

def track_dataset(df_before, df_after):
    print("\n📈 ===== DATASET TRACKING =====")

    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "tracking_log.txt")

    with open(log_file, "a") as f:
        f.write("\n===== TRACKING LOG =====\n")
        f.write(f"Waktu: {datetime.now()}\n")
        f.write(f"Data sebelum: {df_before.shape}\n")
        f.write(f"Data sesudah: {df_after.shape}\n")
        f.write(f"Missing sebelum: {df_before.isnull().sum().sum()}\n")
        f.write(f"Missing sesudah: {df_after.isnull().sum().sum()}\n")

    print("✅ Tracking disimpan di logs/tracking_log.txt")

if __name__ == "__main__":
    df_before = load_raw_data()
    df_after = load_processed_data()

    track_dataset(df_before, df_after)