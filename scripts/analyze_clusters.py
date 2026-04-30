from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import sparse


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def make_param_suffix_from_row(row: pd.Series) -> str:
    """
    Reconstruct the parameter suffix exactly as it was created in run_experiments.py.

    This is important because label files are saved using names such as:
    minibatch_kmeans_svd_50_n_clusters-2_random_state-42_labels.npy
    dbscan_full_sparse_eps-0.3_min_samples-5_metric-cosine_labels.npy
    """
    ordered_param_names = [
        "n_clusters",
        "random_state",
        "eps",
        "min_samples",
        "metric",
    ]

    int_like_params = {"n_clusters", "min_samples", "random_state"}
    param_parts = []

    for key in ordered_param_names:
        col = f"param_{key}"

        if col not in row.index:
            continue

        value = row[col]

        if pd.isna(value):
            continue

        if key in int_like_params:
            value_str = str(int(float(value)))
        else:
            value_str = str(value)

        param_parts.append(f"{key}-{value_str}")

    return "_".join(param_parts)


def load_labels_for_result(reports_dir: Path, row: pd.Series) -> np.ndarray:
    algorithm = row["algorithm"]
    representation = row["representation"]
    param_suffix = make_param_suffix_from_row(row)

    if not param_suffix:
        raise FileNotFoundError(
            f"Missing parameter suffix for algorithm={algorithm}, representation={representation}."
        )

    path = reports_dir / f"{algorithm}_{representation}_{param_suffix}_labels.npy"

    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    return np.load(path, allow_pickle=True)


def select_best_by_ari(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work.dropna(subset=["ari"])

    idx = work.groupby(["algorithm", "representation"])["ari"].idxmax()
    best = work.loc[idx].copy()
    best = best.sort_values("ari", ascending=False).reset_index(drop=True)

    return best


def compute_cluster_analysis(
    X,
    y_true: np.ndarray,
    labels: np.ndarray,
    feature_names: np.ndarray,
    algorithm: str,
    representation: str,
    ari: float,
    top_n: int = 15,
) -> tuple[list[dict], list[dict]]:
    if X.shape[0] != len(labels):
        raise ValueError(
            f"X and labels length mismatch: {X.shape[0]} vs {len(labels)}"
        )

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"X and feature_names length mismatch: {X.shape[1]} vs {len(feature_names)}"
        )

    if len(y_true) != len(labels):
        raise ValueError(
            f"y_true and labels length mismatch: {len(y_true)} vs {len(labels)}"
        )

    global_mean = np.asarray(X.mean(axis=0)).ravel()

    top_word_rows = []
    summary_rows = []

    unique_clusters = sorted(np.unique(labels))

    for cluster_id in unique_clusters:
        cluster_mask = labels == cluster_id
        cluster_size = int(cluster_mask.sum())

        if cluster_size == 0:
            continue

        X_cluster = X[cluster_mask]
        y_cluster = y_true[cluster_mask]

        cluster_mean = np.asarray(X_cluster.mean(axis=0)).ravel()
        distinctiveness = cluster_mean - global_mean

        top_indices = np.argsort(distinctiveness)[::-1][:top_n]

        unique_y, counts_y = np.unique(y_cluster, return_counts=True)
        dominant_idx = int(np.argmax(counts_y))
        dominant_label = unique_y[dominant_idx]
        dominant_count = int(counts_y[dominant_idx])
        purity = dominant_count / cluster_size

        summary_rows.append(
            {
                "algorithm": algorithm,
                "representation": representation,
                "ari": ari,
                "cluster": int(cluster_id),
                "cluster_size": cluster_size,
                "dominant_true_label": dominant_label,
                "dominant_true_label_count": dominant_count,
                "purity": purity,
                "top_words": ", ".join(feature_names[top_indices]),
            }
        )

        for rank, feature_idx in enumerate(top_indices, start=1):
            top_word_rows.append(
                {
                    "algorithm": algorithm,
                    "representation": representation,
                    "ari": ari,
                    "cluster": int(cluster_id),
                    "cluster_size": cluster_size,
                    "rank": rank,
                    "feature": feature_names[feature_idx],
                    "cluster_mean": cluster_mean[feature_idx],
                    "global_mean": global_mean[feature_idx],
                    "mean_minus_global": distinctiveness[feature_idx],
                }
            )

    return top_word_rows, summary_rows


def build_pdf_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    """
    Create a compact table suitable for copying into the PDF.
    We keep only a few strongest/most interpretable rows.
    """
    compact = summary_df.copy()

    compact = compact.sort_values(
        ["ari", "algorithm", "representation", "cluster_size"],
        ascending=[False, True, True, False],
    )

    columns = [
        "algorithm",
        "representation",
        "cluster",
        "cluster_size",
        "dominant_true_label",
        "purity",
        "top_words",
    ]

    compact = compact[columns].copy()
    compact["purity"] = compact["purity"].round(4)

    compact.to_csv(output_path, index=False)


def main() -> None:
    processed_dir = Path("data/processed")
    reports_dir = Path("results/cluster_reports")
    metrics_path = Path("results/metrics/clustering_metrics.csv")
    output_dir = Path("results/cluster_analysis")

    ensure_dir(output_dir)

    print("=== LOADING INPUTS ===")
    X_sample = sparse.load_npz(processed_dir / "X_sample.npz")
    y_sample = np.load(processed_dir / "y_sample.npy", allow_pickle=True)
    feature_names = np.load(processed_dir / "feature_names_filtered.npy", allow_pickle=True)
    metrics_df = pd.read_csv(metrics_path)

    print(f"X_sample shape: {X_sample.shape}")
    print(f"y_sample shape: {y_sample.shape}")
    print(f"feature_names length: {len(feature_names)}")

    print("\n=== SELECTING BEST MODELS BY ARI ===")
    selected = select_best_by_ari(metrics_df)
    selected.to_csv(output_dir / "selected_models_by_ari.csv", index=False)

    print(f"Selected rows: {len(selected)}")

    all_top_word_rows = []
    all_summary_rows = []

    print("\n=== ANALYZING CLUSTERS ===")

    for _, row in selected.iterrows():
        algorithm = row["algorithm"]
        representation = row["representation"]
        ari = row["ari"]

        print(f"Analyzing {algorithm} on {representation}, ARI={ari:.4f}")

        labels = load_labels_for_result(reports_dir, row)

        top_word_rows, summary_rows = compute_cluster_analysis(
            X=X_sample,
            y_true=y_sample,
            labels=labels,
            feature_names=feature_names,
            algorithm=algorithm,
            representation=representation,
            ari=ari,
            top_n=15,
        )

        all_top_word_rows.extend(top_word_rows)
        all_summary_rows.extend(summary_rows)

    top_words_df = pd.DataFrame(all_top_word_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    top_words_path = output_dir / "cluster_top_words_all.csv"
    summary_path = output_dir / "cluster_summary_all.csv"
    pdf_table_path = output_dir / "cluster_analysis_for_pdf.csv"

    top_words_df.to_csv(top_words_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    build_pdf_table(summary_df, pdf_table_path)

    print("\n=== SAVED ===")
    print(f" - {top_words_path}")
    print(f" - {summary_path}")
    print(f" - {pdf_table_path}")
    print(f" - {output_dir / 'selected_models_by_ari.csv'}")

    print("\nDONE")


if __name__ == "__main__":
    main()