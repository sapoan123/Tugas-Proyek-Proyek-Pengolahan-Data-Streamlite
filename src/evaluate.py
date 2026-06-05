import pandas as pd
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import numpy as np
import json


def load_data():
    current_dir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(current_dir, "..", "processed", "data_clustered.csv"))
    rules_path = os.path.join(current_dir, "..", "processed", "association_rules.csv")
    rules = pd.read_csv(rules_path) if os.path.exists(rules_path) else None
    return df, rules


# =========================
# EVALUASI KMEANS (OPTIMIZED)
# =========================
def evaluate_kmeans(df):
    print("\n=== EVALUASI KMEANS ===")

    # cek apakah kolom Cluster ada
    if 'Cluster' not in df.columns:
        print(" Kolom 'Cluster' tidak ditemukan, melakukan clustering otomatis...")
        print(" Kolom yang tersedia:", df.columns.tolist())
        
        # Lakukan clustering otomatis dengan KMeans
        try:
            X = df[['Quantity', 'UnitPrice', 'TotalPrice']].copy()
            X = X.dropna()
            kmeans = KMeans(n_clusters=3, random_state=42)
            df['Cluster'] = -1  # default value
            df.loc[X.index, 'Cluster'] = kmeans.fit_predict(X)
            print(" Clustering selesai dengan k=3")
        except Exception as e:
            print(f" Error saat clustering: {e}")
            return None

    # ambil fitur
    X = df[['Quantity', 'UnitPrice', 'TotalPrice']].copy()
    labels = df['Cluster'].copy()

    print(f" Total data points: {len(X)}")
    print(f" Jumlah cluster unik sebelum filtering: {len(labels.unique())}")
    print(f" Cluster yang ditemukan: {sorted(labels.unique())}")

    # buang NaN biar aman
    valid = X.notnull().all(axis=1)
    X = X[valid]
    labels = labels.loc[valid]

    print(f" Data setelah filtering NaN: {len(X)}")

    #  SAMPLING BIAR CEPAT
    if len(X) > 5000:
        sample_idx = X.sample(5000, random_state=42).index
        X = X.loc[sample_idx]
        labels = labels.loc[sample_idx]
        print(f" Data setelah sampling: {len(X)}")

    # cek cluster valid - PERBAIKAN: jika < 2 cluster, lakukan re-clustering
    unique_clusters = len(labels.unique())
    print(f" Jumlah cluster unik setelah filtering: {unique_clusters}")
    
    if unique_clusters < 2:
        print(f" Hanya {unique_clusters} cluster ditemukan, melakukan re-clustering...")
        try:
            kmeans = KMeans(n_clusters=3, random_state=42)
            labels = kmeans.fit_predict(X)
            print(f" Re-clustering selesai, cluster unik: {len(np.unique(labels))}")
            unique_clusters = len(np.unique(labels))
        except Exception as e:
            print(f" Error saat re-clustering: {e}")
            # Return hasil error (akan ditangani di main)
            return {
                "status": "FAILED",
                "error": f"Tidak bisa melakukan clustering: {str(e)}",
                "total_data_points": len(X),
                "num_clusters": 0
            }

    # Hitung berbagai metrik evaluasi
    print("\n" + "="*50)
    print(" HASIL EVALUASI KMEANS")
    print("="*50)
    
    try:
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        
        print(f" Silhouette Score: {silhouette:.4f}")
        print(f"   (Range: -1 to 1, lebih tinggi lebih baik)")
        print(f"\n Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"   (Lebih rendah lebih baik)")
        print(f"\n Calinski-Harabasz Score: {calinski_harabasz:.4f}")
        print(f"   (Lebih tinggi lebih baik)")
        
        metrics_calculated = True
    except Exception as e:
        print(f" Error menghitung metrics: {e}")
        silhouette = davies_bouldin = calinski_harabasz = None
        metrics_calculated = False
    
    # Distribusi cluster
    from collections import Counter
    cluster_counts = Counter(labels)
    print(f"\n Distribusi Cluster:")
    for cluster in sorted(cluster_counts.keys()):
        count = cluster_counts[cluster]
        percentage = (count / len(labels)) * 100
        print(f"   Cluster {cluster}: {count} data points ({percentage:.1f}%)")
    
    # Statistik fitur per cluster
    print(f"\n Rata-rata Fitur per Cluster:")
    df_eval = X.copy()
    df_eval['Cluster'] = labels
    for cluster in sorted(set(labels)):
        cluster_data = df_eval[df_eval['Cluster'] == cluster]
        print(f"\n   Cluster {cluster}:")
        print(f"      - Quantity: {cluster_data['Quantity'].mean():.2f}")
        print(f"      - UnitPrice: {cluster_data['UnitPrice'].mean():.2f}")
        print(f"      - TotalPrice: {cluster_data['TotalPrice'].mean():.2f}")
    
    # Simpan hasil ke dictionary (TIDAK langsung ke JSON, akan digabung di main)
    # Convert cluster distribution ke native Python types untuk JSON serialization
    cluster_dist_native = {int(k): int(v) for k, v in cluster_counts.items()}
    
    results = {
        "status": "SUCCESS" if metrics_calculated else "PARTIAL",
        "silhouette_score": float(silhouette) if silhouette is not None else None,
        "davies_bouldin_index": float(davies_bouldin) if davies_bouldin is not None else None,
        "calinski_harabasz_score": float(calinski_harabasz) if calinski_harabasz is not None else None,
        "total_data_points": int(len(labels)),
        "num_clusters": int(len(set(labels))),
        "cluster_distribution": cluster_dist_native
    }
    
    return results


# =========================
# EVALUASI APRIORI (ARM)
# =========================
def evaluate_apriori(rules):
    print("\n=== EVALUASI ASSOCIATION RULE MINING (ARM) ===")

    if rules is None or len(rules) == 0:
        print(" Tidak ada rules ditemukan atau file tidak tersedia")
        return {
            "status": "NO_RULES",
            "total_rules": 0,
            "message": "Tidak ada association rules ditemukan"
        }

    print(f" Total rules ditemukan: {len(rules)}")
    
    try:
        # Statistik dasar
        avg_support = float(rules['support'].mean())
        avg_confidence = float(rules['confidence'].mean())
        avg_lift = float(rules['lift'].mean())
        
        min_support = float(rules['support'].min())
        max_support = float(rules['support'].max())
        min_confidence = float(rules['confidence'].min())
        max_confidence = float(rules['confidence'].max())
        min_lift = float(rules['lift'].min())
        max_lift = float(rules['lift'].max())
        
        print(f"\n Statistik Rules:")
        print(f"   Support    - Min: {min_support:.4f}, Max: {max_support:.4f}, Avg: {avg_support:.4f}")
        print(f"   Confidence - Min: {min_confidence:.4f}, Max: {max_confidence:.4f}, Avg: {avg_confidence:.4f}")
        print(f"   Lift       - Min: {min_lift:.4f}, Max: {max_lift:.4f}, Avg: {avg_lift:.4f}")
        
        # Top 5 rules by lift
        print("\n Top 5 Rules (sorted by lift):")
        top_rules = rules.nlargest(5, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        print(top_rules.to_string())
        
        # Konversi top rules ke format JSON-serializable
        top_rules_list = []
        for idx, row in rules.nlargest(5, 'lift').iterrows():
            top_rules_list.append({
                "antecedents": str(list(row['antecedents'])),
                "consequents": str(list(row['consequents'])),
                "support": float(row['support']),
                "confidence": float(row['confidence']),
                "lift": float(row['lift'])
            })
        
        # Hitung rules dengan confidence tinggi (>= 0.5)
        high_confidence_rules = len(rules[rules['confidence'] >= 0.5])
        high_lift_rules = len(rules[rules['lift'] >= 2])
        
        print(f"\n Analisis Lebih Lanjut:")
        print(f"   Rules dengan confidence >= 0.5: {high_confidence_rules}")
        print(f"   Rules dengan lift >= 2: {high_lift_rules}")
        
        results = {
            "status": "SUCCESS",
            "total_rules": int(len(rules)),
            "statistics": {
                "support": {
                    "min": min_support,
                    "max": max_support,
                    "average": avg_support
                },
                "confidence": {
                    "min": min_confidence,
                    "max": max_confidence,
                    "average": avg_confidence
                },
                "lift": {
                    "min": min_lift,
                    "max": max_lift,
                    "average": avg_lift
                }
            },
            "high_confidence_rules": high_confidence_rules,
            "high_lift_rules": high_lift_rules,
            "top_5_rules": top_rules_list
        }
        
        return results
        
    except Exception as e:
        print(f" Error saat evaluasi ARM: {e}")
        return {
            "status": "ERROR",
            "error_message": str(e),
            "total_rules": int(len(rules)) if rules is not None else 0
        }


# =========================
# MAIN
# =========================
def main():
    df, rules = load_data()

    kmeans_results = evaluate_kmeans(df)
    arm_results = evaluate_apriori(rules)
    
    # Gabungkan hasil KMEANS dan ARM ke dalam satu JSON
    print("\n" + "="*50)
    print(" MENYIMPAN HASIL EVALUASI GABUNGAN")
    print("="*50)
    
    combined_results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "kmeans_evaluation": kmeans_results,
        "arm_evaluation": arm_results
    }
    
    current_dir = os.path.dirname(__file__)
    results_path = os.path.join(current_dir, "..", "metrics.json")
    
    try:
        with open(results_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        print(f" Hasil evaluasi KMEANS + ARM disimpan ke: metrics.json")
    except Exception as e:
        print(f" Error menyimpan metrics.json: {e}")
    
    print("\n" + "="*50)
    print(" EVALUASI SELESAI")
    print("="*50)


if __name__ == "__main__":
    main()