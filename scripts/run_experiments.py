from pathlib import Path
import numpy as np

from src.clustering import (
    load_dense_array,
    load_sparse_matrix,
    run_single_experiment,
    save_metrics_table,
)


def make_param_suffix(params: dict) -> str:
    return "_".join(f"{k}-{v}" for k, v in params.items())


def main() -> None:
    processed_dir = Path("data/processed")
    metrics_dir = Path("results/metrics")
    reports_dir = Path("results/cluster_reports")
    models_dir = Path("models/clustering")
    configs_dir = Path("results/configs")

    y_sample = np.load(processed_dir / "y_sample.npy", allow_pickle=True)

    # Dense reduced representations
    dense_representations = {
        "svd_100": processed_dir / "X_svd_100.npy",
        "svd_50": processed_dir / "X_svd_50.npy",
        "svd_20": processed_dir / "X_svd_20.npy",
    }

    # Full feature-space sparse representation
    # Promeni ime fajla ako je kod tebe drugacije.
    sparse_representations = {
        "full_sparse": processed_dir / "X_sample.npz",
    }

    dense_experiments = []

    # Dense algorithms over reduced representations
    for k in [2, 3, 5, 7]:
        dense_experiments.append(("minibatch_kmeans", {"n_clusters": k, "random_state": 42}))
        dense_experiments.append(("agglomerative", {"n_clusters": k}))
        dense_experiments.append(("gmm", {"n_clusters": k, "random_state": 42}))
        dense_experiments.append(("birch", {"n_clusters": k}))

    for eps in [0.3, 0.5, 0.7, 1.0, 1.5]:
        for min_samples in [5, 10]:
            dense_experiments.append(("dbscan", {"eps": eps, "min_samples": min_samples}))

    sparse_experiments = []

    # Sparse-safe algorithms over full data
    for k in [2, 3, 5, 7]:
        sparse_experiments.append(("minibatch_kmeans", {"n_clusters": k, "random_state": 42}))

    # Kod sparse teksta cosine obicno ima vise smisla od euclidean
    for eps in [0.3, 0.5, 0.7, 1.0]:
        for min_samples in [5, 10]:
            sparse_experiments.append(
                ("dbscan", {"eps": eps, "min_samples": min_samples, "metric": "cosine"})
            )

    all_results = []

    # 1) Reduced dense representations
    for rep_name, rep_path in dense_representations.items():
        print(f"\n=== Loading dense representation: {rep_name} ===")
        X = load_dense_array(rep_path)

        for algorithm_name, params in dense_experiments:
            print(f"Running {algorithm_name} on {rep_name} with params={params}...")

            param_suffix = make_param_suffix(params)

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

    # 2) Full sparse representations
    for rep_name, rep_path in sparse_representations.items():
        print(f"\n=== Loading sparse representation: {rep_name} ===")
        X = load_sparse_matrix(rep_path)

        for algorithm_name, params in sparse_experiments:
            print(f"Running {algorithm_name} on {rep_name} with params={params}...")

            param_suffix = make_param_suffix(params)

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