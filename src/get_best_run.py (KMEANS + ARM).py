# =========================
# FILE 2: get_best_run.py
# (KMEANS + ARM)
# =========================

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

EXPERIMENT_NAME = "KMeans_ARM_proyek_data_maining_4"


# =========================
# BEST KMEANS
# =========================
def get_best_kmeans():

    exp = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    runs = mlflow.search_runs(
        [exp.experiment_id]
    )

    runs = runs.dropna(
        subset=["metrics.silhouette_score"]
    )

    best = runs.loc[
        runs["metrics.silhouette_score"].idxmax()
    ]

    return best


# =========================
# ASSOCIATION RULE INFO
# =========================
def get_arm_info():

    exp = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    runs = mlflow.search_runs(
        [exp.experiment_id]
    )

    arm = runs[
        runs["params.method"] == "Association Rule"
    ]

    return arm.head(1)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # =========================
    # BEST KMEANS
    # =========================
    best = get_best_kmeans()

    print("\n🏆 BEST KMEANS")
    print("Run ID :", best.run_id)

    print(
        "Best Model :",
        best["params.method"]
    )

    print(
        "Best Parameter (k) :",
        best["params.n_clusters"]
    )

    print(
        "Best Metric (silhouette_score) :",
        best["metrics.silhouette_score"]
    )

    # =========================
    # ASSOCIATION RULE
    # =========================
    arm = get_arm_info()

    print("\n🏆 ASSOCIATION RULE")

    print(
        "ARM Run ID :",
        arm.run_id.values[0]
    )

    print(
        "Model ARM :",
        arm["params.method"].values[0]
    )

    # jika ada parameter tambahan
    if "params.best_model" in arm.columns:

        print(
            "Best ARM Model :",
            arm["params.best_model"].values[0]
        )

    if "params.best_min_support" in arm.columns:

        print(
            "Best Min Support :",
            arm["params.best_min_support"].values[0]
        )

    if "params.best_min_threshold" in arm.columns:

        print(
            "Best Min Threshold :",
            arm["params.best_min_threshold"].values[0]
        )

    # metric ARM
    if "metrics.best_num_rules" in arm.columns:

        print(
            "Best Num Rules :",
            arm["metrics.best_num_rules"].values[0]
        )

    if "metrics.best_num_itemsets" in arm.columns:

        print(
            "Best Num Itemsets :",
            arm["metrics.best_num_itemsets"].values[0]
        )