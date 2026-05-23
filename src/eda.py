import pandas as pd
import os
import matplotlib.pyplot as plt

def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(BASE_DIR, "data", "raw", "Online Retail.xlsx")

    print(" Loading dataset:", path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    return pd.read_excel(path)


def save_plot(BASE_DIR, filename):
    output_dir = os.path.join(BASE_DIR, "outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, filename)
    plt.savefig(file_path, bbox_inches='tight')
    print(" Plot disimpan di:", file_path)


def perform_eda(df):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("\n ===== EDA =====")

    print("\n Dimensi data:", df.shape)

    print("\n Missing values:")
    print(df.isnull().sum())

    # =========================
    # 1. Missing Values Plot
    # =========================
    plt.figure()
    df.isnull().sum().plot(kind='bar')
    plt.title("Missing Values per Kolom")
    plt.tight_layout()
    save_plot(BASE_DIR, "missing_values.png")
    plt.close()

    # =========================
    # 2. Quantity Distribution
    # =========================
    if 'Quantity' in df.columns:
        plt.figure()
        df['Quantity'].plot(kind='hist', bins=50)
        plt.title("Distribusi Quantity")
        plt.tight_layout()
        save_plot(BASE_DIR, "quantity_distribution.png")
        plt.close()

    # =========================
    # 3. Top 10 Produk
    # =========================
    if 'Description' in df.columns:
        top_products = df['Description'].value_counts().head(10)

        plt.figure()
        top_products.plot(kind='bar')
        plt.title("Top 10 Produk Terlaris")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_plot(BASE_DIR, "top_products.png")
        plt.close()

    print("\n Semua plot berhasil disimpan!")


if __name__ == "__main__":
    df = load_data()
    perform_eda(df)