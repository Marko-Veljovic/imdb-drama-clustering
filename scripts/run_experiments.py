from pathlib import Path
import numpy as np

from src.clustering import (
    load_dense_array,
    run_single_experiment,
    save_metrics_table,
)


def main() -> None:
    processed_dir = Path("data/processed")
    metrics_dir = Path("results/metrics")
    reports_dir = Path("results/cluster_reports")
    models_dir = Path("models/clustering")
    configs_dir = Path("results/configs")

    y_sample = np.load(processed_dir / "y_sample.npy", allow_pickle=True)

    representations = {
        "svd_100": processed_dir / "X_svd_100.npy",
        "svd_50": processed_dir / "X_svd_50.npy",
        "svd_20": processed_dir / "X_svd_20.npy",
    }

    experiments = []

    # Algorithms that use an explicit number of clusters
    for k in [2, 3, 5, 7]:
        experiments.append(("minibatch_kmeans", {"n_clusters": k, "random_state": 42}))
        experiments.append(("agglomerative", {"n_clusters": k}))
        experiments.append(("gmm", {"n_clusters": k, "random_state": 42}))
        experiments.append(("birch", {"n_clusters": k}))

    # DBSCAN parameter grid
    for eps in [0.3, 0.5, 0.7, 1.0, 1.5]:
        for min_samples in [5, 10]:
            experiments.append(("dbscan", {"eps": eps, "min_samples": min_samples}))

    all_results = []

    for rep_name, rep_path in representations.items():
        print(f"\n=== Loading {rep_name} ===")
        X = load_dense_array(rep_path)

        for algorithm_name, params in experiments:
            print(f"Running {algorithm_name} on {rep_name} with params={params}...")

            param_suffix = "_".join(f"{k}-{v}" for k, v in params.items())

            model_path = models_dir / rep_name / f"{algorithm_name}_{param_suffix}.pkl"
            labels_path = reports_dir / f"{algorithm_name}_{rep_name}_{param_suffix}_labels.npy"
            config_path = configs_dir / f"{algorithm_name}_{rep_name}_{param_suffix}.json"

            result = run_single_experiment(
                algorithm_name=algorithm_name,
                representation_name=rep_name,
                X=X,
                y_true=y_sample,
                model_output_path=model_path,
                labels_output_path=labels_path,
                config_output_path=config_path,
                params=params,
            )

            all_results.append(result)

            print(
                f"{algorithm_name} | rep={rep_name} | "
                f"clusters={result['n_clusters_found']} | "
                f"silhouette={result['silhouette']} | "
                f"ari={result['ari']} | "
                f"saved={result['model_saved']}"
            )

    metrics_path = metrics_dir / "clustering_metrics.csv"
    save_metrics_table(all_results, metrics_path)

    print("\nALL DONE")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()