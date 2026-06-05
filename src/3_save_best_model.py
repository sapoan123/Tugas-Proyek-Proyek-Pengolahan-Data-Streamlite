# =========================
# FILE 3: save_best_model.py
# (KMEANS + ARM)
# =========================

import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import os

# =========================
# CONFIG
# =========================
mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

EXPERIMENT_NAME = (
    "KMeans_ARM_proyek_data_maining_4"
)


# =========================
# SAVE BEST KMEANS MODEL
# =========================
def save_kmeans():

    exp = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    runs = mlflow.search_runs(
        [exp.experiment_id]
    )

    runs = runs.dropna(
        subset=["metrics.silhouette_score"]
    )

    # =========================
    # BEST MODEL
    # =========================
    best = runs.loc[
        runs[
            "metrics.silhouette_score"
        ].idxmax()
    ]

    print("\n BEST KMEANS")

    print(
        "Run ID :",
        best.run_id
    )

    print(
        "Best Score :",
        best["metrics.silhouette_score"]
    )

    # =========================
    # LOAD MODEL
    # =========================
    model_uri = (
        f"runs:/{best.run_id}/model_k_"
        f"{best['params.n_clusters']}"
    )

    model = mlflow.sklearn.load_model(
        model_uri
    )

    # =========================
    # SAVE MODEL
    # =========================
    os.makedirs(
        "models/final",
        exist_ok=True
    )

    model_path = (
        "models/final/kmeans.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f" Model KMeans disimpan di: "
        f"{model_path}"
    )


# =========================
# SAVE ARM MODEL
# =========================
def save_arm():

    exp = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    runs = mlflow.search_runs(
        [exp.experiment_id]
    )

    arm = runs[
        runs["params.method"]
        == "Association Rule"
    ]

    if arm.empty:

        print(
            " Run Association Rule tidak ditemukan"
        )

        return

    arm_run = arm.iloc[0]

    print("\n ASSOCIATION RULE")

    print(
        "Run ID :",
        arm_run.run_id
    )

    # =========================
    # DOWNLOAD ARTIFACT
    # =========================
    client = mlflow.tracking.MlflowClient()

    local_path = client.download_artifacts(
        arm_run.run_id,
        "association_rules.csv"
    )

    # =========================
    # LOAD RULES
    # =========================
    rules = pd.read_csv(local_path)

    # =========================
    # SAVE MODEL ARM
    # =========================
    arm_model = {
        "rules": rules
    }

    os.makedirs(
        "models/final",
        exist_ok=True
    )

    arm_model_path = (
        "models/final/association_rule.pkl"
    )

    joblib.dump(
        arm_model,
        arm_model_path
    )

    print(
        f" Model ARM disimpan di: "
        f"{arm_model_path}"
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    save_kmeans()

    save_arm()

    print(
        "\n Semua model berhasil disimpan"
    )