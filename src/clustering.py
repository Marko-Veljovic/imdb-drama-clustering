from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib
from scipy import sparse

from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def load_dense_array(path: Path) -> np.ndarray:
    return np.load(path)


def load_sparse_matrix(path: Path):
    return sparse.load_npz(path)


def is_sparse_matrix(X) -> bool:
    return sparse.issparse(X)


def save_labels(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, labels)


def save_model(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def evaluate_clustering(X, labels: np.ndarray, y_true: np.ndarray) -> dict:
    unique_labels = np.unique(labels)
    non_noise_labels = [lab for lab in unique_labels if lab != -1]

    result = {
        "n_clusters_found": int(len(non_noise_labels)),
        "n_noise": int(np.sum(labels == -1)) if -1 in unique_labels else 0,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "ari": np.nan,
        "nmi": np.nan,
    }

    valid_for_internal = len(non_noise_labels) >= 2 and len(non_noise_labels) < len(labels)

    if valid_for_internal:
        # silhouette ume da radi i nad sparse matricama
        try:
            result["silhouette"] = float(silhouette_score(X, labels))
        except Exception:
            pass

        # Ove metrike su najbezbednije nad dense reprezentacijama
        if not is_sparse_matrix(X):
            try:
                result["davies_bouldin"] = float(davies_bouldin_score(X, labels))
            except Exception:
                pass

            try:
                result["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
            except Exception:
                pass

    try:
        result["ari"] = float(adjusted_rand_score(y_true, labels))
    except Exception:
        pass

    try:
        result["nmi"] = float(normalized_mutual_info_score(y_true, labels))
    except Exception:
        pass

    return result


def run_minibatch_kmeans(X, n_clusters: int = 2, random_state: int = 42):
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=1024,
        n_init=10,
    )
    labels = model.fit_predict(X)
    return model, labels


def run_agglomerative(X: np.ndarray, n_clusters: int = 2):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return model, labels


def run_gmm(X: np.ndarray, n_clusters: int = 2, random_state: int = 42):
    model = GaussianMixture(n_components=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


def run_dbscan(X, eps: float = 1.5, min_samples: int = 10, metric: str = "euclidean"):
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = model.fit_predict(X)
    return model, labels


def run_birch(X: np.ndarray, n_clusters: int = 2):
    model = Birch(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return model, labels


def algorithm_supports_sparse(algorithm_name: str) -> bool:
    return algorithm_name in {"minibatch_kmeans", "dbscan"}


def algorithm_requires_dense(algorithm_name: str) -> bool:
    return algorithm_name in {"agglomerative", "gmm", "birch"}


def run_single_experiment(
    algorithm_name: str,
    representation_name: str,
    X,
    y_true: np.ndarray,
    model_output_path: Path,
    labels_output_path: Path,
    config_output_path: Path,
    params: dict | None = None,
) -> dict:
    params = params or {}

    if is_sparse_matrix(X) and algorithm_requires_dense(algorithm_name):
        raise ValueError(
            f"Algorithm '{algorithm_name}' does not support sparse input for representation '{representation_name}'."
        )

    start = time.perf_counter()

    if algorithm_name == "minibatch_kmeans":
        model, labels = run_minibatch_kmeans(X, **params)
    elif algorithm_name == "agglomerative":
        model, labels = run_agglomerative(X, **params)
    elif algorithm_name == "gmm":
        model, labels = run_gmm(X, **params)
    elif algorithm_name == "dbscan":
        model, labels = run_dbscan(X, **params)
    elif algorithm_name == "birch":
        model, labels = run_birch(X, **params)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    elapsed = time.perf_counter() - start

    save_labels(labels_output_path, labels)

    config = {
        "algorithm": algorithm_name,
        "representation": representation_name,
        "params": params,
        "input_type": "sparse" if is_sparse_matrix(X) else "dense",
    }
    save_config(config_output_path, config)

    model_saved = True
    model_save_error = ""

    try:
        save_model(model_output_path, model)
    except Exception as e:
        model_saved = False
        model_save_error = str(e)

    metrics = evaluate_clustering(X, labels, y_true)

    metrics["algorithm"] = algorithm_name
    metrics["representation"] = representation_name
    metrics["input_type"] = "sparse" if is_sparse_matrix(X) else "dense"
    metrics["runtime_sec"] = elapsed
    metrics["model_saved"] = model_saved
    metrics["model_save_error"] = model_save_error

    for key, value in params.items():
        metrics[f"param_{key}"] = value

    return metrics


def save_metrics_table(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)