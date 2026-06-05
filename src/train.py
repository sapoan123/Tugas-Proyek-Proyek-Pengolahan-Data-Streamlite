import pandas as pd
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import joblib
import os
import json


def load_processed_data():
    print("Loading processed data...")
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, "..", "processed", "data_clean.csv")
    df = pd.read_csv(path)
    return df


# =========================
# KMEANS
# =========================
def train_kmeans(df):
    print("\nTraining KMeans...")

    X = df[['Quantity', 'UnitPrice', 'TotalPrice']]

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    print("Distribusi Cluster:")
    print(df['Cluster'].value_counts())
    
    # Hitung inertia dan silhouette
    inertia = kmeans.inertia_
    print(f"\n KMeans Inertia: {inertia:.4f}")

    return kmeans, df


# =========================
# APRIORI (ARM)
# =========================
def train_apriori(df):
    print("\n\n=== TRAINING ASSOCIATION RULE MINING (ARM) ===\n")

    basket = (df
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().fillna(0))

    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    print(f" Ukuran basket: {basket.shape[0]} transaksi × {basket.shape[1]} produk")

    frequent_items = apriori(basket, min_support=0.02, use_colnames=True)
    print(f" Jumlah frequent itemset: {len(frequent_items)}")
    
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)

    print(f"\n Total Association Rules ditemukan: {len(rules)}")
    
    # Statistik ARM Training
    if len(rules) > 0:
        print(f"\n STATISTIK TRAINING ARM:")
        print(f"   Support      - Min: {rules['support'].min():.4f}, Max: {rules['support'].max():.4f}, Avg: {rules['support'].mean():.4f}")
        print(f"   Confidence   - Min: {rules['confidence'].min():.4f}, Max: {rules['confidence'].max():.4f}, Avg: {rules['confidence'].mean():.4f}")
        print(f"   Lift         - Min: {rules['lift'].min():.4f}, Max: {rules['lift'].max():.4f}, Avg: {rules['lift'].mean():.4f}")
        
        # Hitung rules berkualitas tinggi
        high_confidence = len(rules[rules['confidence'] >= 0.5])
        high_lift = len(rules[rules['lift'] >= 2])
        print(f"\n Rules berkualitas:")
        print(f"   - Confidence >= 0.5: {high_confidence} rules ({high_confidence/len(rules)*100:.1f}%)")
        print(f"   - Lift >= 2: {high_lift} rules ({high_lift/len(rules)*100:.1f}%)")
        
        # Top 3 rules by lift
        print(f"\n Top 3 Rules (by Lift):")
        top_3 = rules.nlargest(3, 'lift')
        for idx, (i, row) in enumerate(top_3.iterrows(), 1):
            print(f"   {idx}. Antecedents: {list(row['antecedents'])} → Consequents: {list(row['consequents'])}")
            print(f"      Support: {row['support']:.4f}, Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.4f}")

    return rules


# =========================
# SAVE MODEL
# =========================
def save_model(kmeans):
    current_dir = os.path.dirname(__file__)
    model_dir = os.path.join(current_dir, "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "kmeans_model.pkl")
    joblib.dump(kmeans, model_path)
    print(f" Model KMeans disimpan di: {model_path}")
    return model_path


# =========================
# SAVE TRAINING RESULTS
# =========================
def save_training_results(kmeans, df_clustered, rules):
    """Simpan hasil training KMEANS dan ARM ke file JSON"""
    current_dir = os.path.dirname(__file__)
    
    # Persiapan data untuk JSON
    kmeans_results = {
        "model": "KMeans",
        "n_clusters": kmeans.n_clusters,
        "inertia": float(kmeans.inertia_),
        "cluster_distribution": df_clustered['Cluster'].value_counts().to_dict()
    }
    
    arm_results = {
        "model": "Association Rule Mining (Apriori)",
        "total_rules": int(len(rules))
    }
    
    if len(rules) > 0:
        arm_results.update({
            "support": {
                "min": float(rules['support'].min()),
                "max": float(rules['support'].max()),
                "average": float(rules['support'].mean())
            },
            "confidence": {
                "min": float(rules['confidence'].min()),
                "max": float(rules['confidence'].max()),
                "average": float(rules['confidence'].mean())
            },
            "lift": {
                "min": float(rules['lift'].min()),
                "max": float(rules['lift'].max()),
                "average": float(rules['lift'].mean())
            },
            "high_confidence_rules": int(len(rules[rules['confidence'] >= 0.5])),
            "high_lift_rules": int(len(rules[rules['lift'] >= 2]))
        })
    
    training_results = {
        "status": "SUCCESS",
        "kmeans": kmeans_results,
        "arm": arm_results
    }
    
    # Simpan ke training_results.json
    results_path = os.path.join(current_dir, "..", "training_results.json")
    try:
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        print(f"\n Hasil training disimpan ke: training_results.json")
    except Exception as e:
        print(f" Error menyimpan training_results.json: {e}")


# =========================
# MAIN TRAIN
# =========================
def main():
    print("="*60)
    print(" STARTING TRAINING PIPELINE")
    print("="*60)
    
    df = load_processed_data()

    print(f"\n Data shape: {df.shape}")
    
    # Training KMEANS
    print("\n" + "="*60)
    print("PHASE 1: KMEANS CLUSTERING TRAINING")
    print("="*60)
    kmeans, df_clustered = train_kmeans(df)
    
    # Training ARM
    print("\n" + "="*60)
    print("PHASE 2: ASSOCIATION RULE MINING TRAINING")
    print("="*60)
    rules = train_apriori(df_clustered)

    # Simpan model dan results
    print("\n" + "="*60)
    print("PHASE 3: SAVING MODELS & RESULTS")
    print("="*60)
    save_model(kmeans)
    save_training_results(kmeans, df_clustered, rules)

    # Simpan hasil cluster
    current_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(current_dir, "..", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    clustered_path = os.path.join(processed_dir, "data_clustered.csv")
    df_clustered.to_csv(clustered_path, index=False)
    print(f" Data dengan cluster disimpan di: data_clustered.csv")

    # Simpan rules
    rules_path = os.path.join(processed_dir, "association_rules.csv")
    rules.to_csv(rules_path, index=False)
    print(f" Rules disimpan di: association_rules.csv")
    
    print("\n" + "="*60)
    print(" TRAINING SELESAI")
    print("="*60)


if __name__ == "__main__":
    main()