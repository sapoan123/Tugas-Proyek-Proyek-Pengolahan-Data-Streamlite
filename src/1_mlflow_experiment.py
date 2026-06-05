# =========================
# MLflow Experiment
# KMeans + Association Rule
# =========================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import fpgrowth, association_rules
import os
import joblib

# =========================
# CONFIG
# =========================
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(
    "KMeans_ARM_proyek_data_maining_4"
)

# =========================
# LOAD DATA
# =========================
def load_data():

    path = "data/raw/Online Retail.xlsx"

    if not os.path.exists(path):
        raise FileNotFoundError(
            f" File tidak ditemukan: {path}"
        )

    df = pd.read_excel(
        path,
        engine="openpyxl"
    )

    print("Data berhasil load dari Excel")

    # =========================
    # CLEANING
    # =========================
    df = df.dropna()

    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    df['TotalPrice'] = (
        df['Quantity'] * df['UnitPrice']
    )

    print(
        f"Data setelah cleaning: {df.shape}"
    )

    return df


# =========================
# KMEANS
# =========================
def run_kmeans(df):

    cols = [
        'Quantity',
        'UnitPrice',
        'TotalPrice'
    ]

    for col in cols:

        if col not in df.columns:
            raise ValueError(
                f"Kolom {col} tidak ditemukan"
            )

    X = df[cols].dropna()

    if len(X) > 3000:
        X = X.sample(
            3000,
            random_state=42
        )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print(f"Data KMeans: {X.shape}")

    best_score = -1
    best_model = None
    best_k = None

    # =========================
    # TRAINING KMEANS
    # =========================
    for k in range(2, 5):

        with mlflow.start_run(
            run_name=f"KMeans_k={k}"
        ):

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            labels = model.fit_predict(
                X_scaled
            )

            unique_labels = np.unique(
                labels
            )

            if len(unique_labels) < 2:

                print(
                    f"KMeans k={k} hanya 1 cluster"
                )

                mlflow.log_metric(
                    "silhouette_score",
                    -1
                )

                continue

            score = silhouette_score(
                X_scaled,
                labels,
                sample_size=min(
                    1000,
                    len(X_scaled)
                )
            )

            # =========================
            # LOG MLFLOW
            # =========================
            mlflow.log_param(
                "method",
                "KMeans"
            )

            mlflow.log_param(
                "n_clusters",
                k
            )

            mlflow.log_metric(
                "silhouette_score",
                float(score)
            )

            mlflow.sklearn.log_model(
                model,
                name=f"model_k_{k}"
            )

            print(
                f"✔️ KMeans k={k} | "
                f"score={score:.4f}"
            )

            # =========================
            # SAVE BEST MODEL
            # =========================
            if score > best_score:

                best_score = score
                best_model = model
                best_k = k

    # =========================
    # BEST RESULT
    # =========================
    print("\n KMEANS TERBAIK")

    print("Model Terbaik : KMeans")

    print(
        f"Parameter Terbaik : "
        f"n_clusters={best_k}"
    )

    print(
        f"Silhouette Score : "
        f"{best_score:.4f}"
    )

    # =========================
    # SAVE BEST MODEL
    # =========================
    if best_model is not None:

        best_model_path = (
            "best_kmeans_model.pkl"
        )

        joblib.dump(
            best_model,
            best_model_path
        )

        with mlflow.start_run(
            run_name="Best_KMeans_Model"
        ):

            mlflow.log_param(
                "best_model",
                "KMeans"
            )

            mlflow.log_param(
                "best_n_clusters",
                best_k
            )

            mlflow.log_metric(
                "best_silhouette_score",
                float(best_score)
            )

            mlflow.log_artifact(
                best_model_path
            )

        print(
            "Best KMeans Model disimpan"
        )


# =========================
# ASSOCIATION RULE
# =========================
def run_association_rule(df):

    with mlflow.start_run(
        run_name="Association_Rule"
    ):

        print("\n RUN ASSOCIATION RULE")

        required_cols = [
            'InvoiceNo',
            'Description',
            'Quantity'
        ]

        for col in required_cols:

            if col not in df.columns:
                raise ValueError(
                    f" Kolom {col} tidak ditemukan"
                )

        df = df[required_cols].dropna()

        # =========================
        # TAMBAH DATA
        # =========================
        df = df.head(10000)

        # =========================
        # FILTER TRANSAKSI
        # =========================
        transaksi = (
            df.groupby('InvoiceNo')
            .size()
        )

        valid_invoice = transaksi[
            transaksi > 1
        ].index

        df = df[
            df['InvoiceNo'].isin(
                valid_invoice
            )
        ]

        # =========================
        # TOP PRODUCT DIPERBANYAK
        # =========================
        top_products = (
            df['Description']
            .value_counts()
            .head(30)
            .index
        )

        df = df[
            df['Description'].isin(
                top_products
            )
        ]

        print(f" Data ARM: {df.shape}")

        # =========================
        # BASKET
        # =========================
        basket = pd.crosstab(
            df['InvoiceNo'],
            df['Description']
        )

        basket = (
            basket > 0
        ).astype(int)

        print(
            f" Basket Shape: "
            f"{basket.shape}"
        )

        if basket.shape[0] == 0:

            print(" Basket kosong")

            mlflow.log_metric(
                "num_rules",
                0
            )

            return

        # =========================
        # FP GROWTH
        # =========================
        freq = fpgrowth(
            basket,
            min_support=0.05,
            use_colnames=True
        )

        print(
            f" Frequent Itemsets: "
            f"{len(freq)}"
        )

        if freq.empty:

            mlflow.log_metric(
                "num_rules",
                0
            )

            mlflow.log_metric(
                "num_itemsets",
                0
            )

            return

        # =========================
        # ASSOCIATION RULE
        # =========================
        rules = association_rules(
            freq,
            metric="confidence",
            min_threshold=0.3
        )

        # =========================
        # SORT RULES
        # =========================
        if not rules.empty:

            rules = rules.sort_values(
                by="confidence",
                ascending=False
            )

        print(
            f"Total Rules: "
            f"{len(rules)}"
        )

        # =========================
        # TAMPILKAN RULES TERBAIK
        # =========================
        if not rules.empty:

            print("\n TOP RULES")

            print(
                rules[
                    [
                        'antecedents',
                        'consequents',
                        'support',
                        'confidence',
                        'lift'
                    ]
                ].head(10)
            )

        # =========================
        # BEST MODEL INFO
        # =========================
        best_model = "FP-Growth"

        best_parameters = {
            "min_support": 0.05,
            "metric": "confidence",
            "min_threshold": 0.3
        }

        best_metric = {
            "num_rules": len(rules),
            "num_itemsets": len(freq)
        }

        print("\n🏆 ASSOCIATION RULE")

        print(
            f" Model Terbaik : "
            f"{best_model}"
        )

        print(
            f" Parameter : "
            f"{best_parameters}"
        )

        print(
            f" Metric : "
            f"{best_metric}"
        )

        # =========================
        # SAVE RULES
        # =========================
        rules_path = (
            "association_rules.csv"
        )

        rules.to_csv(
            rules_path,
            index=False
        )

        # =========================
        # SAVE MODEL
        # =========================
        arm_model_path = (
            "best_association_rule_model.pkl"
        )

        joblib.dump(
            rules,
            arm_model_path
        )

        # =========================
        # LOG MLFLOW
        # =========================
        mlflow.log_param(
            "method",
            "Association Rule"
        )

        mlflow.log_param(
            "best_model",
            best_model
        )

        mlflow.log_param(
            "best_min_support",
            0.05
        )

        mlflow.log_param(
            "best_metric_name",
            "confidence"
        )

        mlflow.log_param(
            "best_min_threshold",
            0.3
        )

        mlflow.log_metric(
            "num_rules",
            len(rules)
        )

        mlflow.log_metric(
            "num_itemsets",
            len(freq)
        )

        mlflow.log_metric(
            "best_num_rules",
            len(rules)
        )

        mlflow.log_metric(
            "best_num_itemsets",
            len(freq)
        )

        mlflow.log_artifact(
            rules_path
        )

        mlflow.log_artifact(
            arm_model_path
        )

        print(
            " Association Rule selesai"
        )


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print(" START EXPERIMENT")

    df = load_data()

    print(
        f" Data awal: {df.shape}"
    )

    run_kmeans(df)

    run_association_rule(df)

    print(
        "\n SELESAI → cek MLflow UI"
    )