import pandas as pd
import os

def load_data():
    current_dir = os.path.dirname(__file__)

    #  FIX PATH (lebih aman & fleksibel)
    path = os.path.abspath(
        os.path.join(current_dir, "..", "data", "raw", "Online Retail.xlsx")
    )

    print(" Loading dataset dari:", path)

    if not os.path.exists(path):
        raise FileNotFoundError(f" File tidak ditemukan: {path}")

    return pd.read_excel(path)


def preprocess_data(df):
    print("\n🧹 ===== PREPROCESSING =====")

    # Hapus missing value
    df = df.dropna()

    # Hapus transaksi cancel
    df = df[~df['InvoiceNo'].astype(str).str.contains('C')]

    # Hapus quantity negatif
    df = df[df['Quantity'] > 0]

    # Tambah kolom total harga
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    print(" Data setelah preprocessing:", df.shape)

    return df


def save_data(df):
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "processed")
    )

    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, "data_clean.csv")
    df.to_csv(file_path, index=False)

    print(" Data disimpan di:", file_path)

    return file_path


if __name__ == "__main__":
    df = load_data()
    df_clean = preprocess_data(df)
    save_data(df_clean)