from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dense_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def load_metrics_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# =========================
# COMMON HELPERS
# =========================

def sanitize_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def make_model_label(df: pd.DataFrame) -> pd.Series:
    return df["algorithm"].astype(str) + "\n" + df["representation"].astype(str)


def make_param_suffix_from_row(row: pd.Series) -> str:
    """
    Reconstruct the parameter suffix exactly as it was created in run_experiments.py.

    Rules:
    - Preserve the CSV column order for param_* columns.
    - Keep float formatting for parameters such as eps (e.g. 1.0).
    - Convert integer-like parameters such as n_clusters, min_samples,
      and random_state from 5.0 to 5.
    """
    int_like_params = {"n_clusters", "min_samples", "random_state"}
    param_parts = []

    for col in row.index:
        if not str(col).startswith("param_"):
            continue

        value = row[col]
        if pd.isna(value):
            continue

        key = str(col).replace("param_", "", 1)

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


# =========================
# GROUP 1 - DATA PLOTS
# =========================

def plot_true_labels_2d(
    X_2d: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    title: str = "True labels in 2D SVD space",
) -> None:
    if X_2d.ndim != 2 or X_2d.shape[1] != 2:
        raise ValueError(f"Expected X_2d shape (n_samples, 2), got {X_2d.shape}")

    ensure_dir(output_path.parent)
    unique_labels = np.unique(y)

    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        mask = y == label
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=12,
            alpha=0.7,
            label=str(label),
        )

    plt.title(title)
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.legend(title="True class", fontsize=8, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_true_labels_3d(
    X_3d: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    title: str = "True labels in 3D SVD space",
) -> None:
    if X_3d.ndim != 2 or X_3d.shape[1] != 3:
        raise ValueError(f"Expected X_3d shape (n_samples, 3), got {X_3d.shape}")

    ensure_dir(output_path.parent)
    unique_labels = np.unique(y)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    for label in unique_labels:
        mask = y == label
        ax.scatter(
            X_3d[mask, 0],
            X_3d[mask, 1],
            X_3d[mask, 2],
            s=12,
            alpha=0.7,
            label=str(label),
        )

    ax.set_title(title)
    ax.set_xlabel("SVD component 1")
    ax.set_ylabel("SVD component 2")
    ax.set_zlabel("SVD component 3")
    ax.legend(title="True class", fontsize=8, markerscale=1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_group_1_plots(processed_dir: Path, plots_dir: Path) -> None:
    data_plots_dir = plots_dir / "data"
    ensure_dir(data_plots_dir)

    x_svd_2_path = processed_dir / "X_svd_2.npy"
    x_svd_3_path = processed_dir / "X_svd_3.npy"
    y_sample_path = processed_dir / "y_sample.npy"

    X_2d = load_dense_array(x_svd_2_path)
    X_3d = load_dense_array(x_svd_3_path)
    y = load_dense_array(y_sample_path)

    if len(X_2d) != len(y):
        raise ValueError(
            f"X_svd_2 and y_sample have different number of rows: {len(X_2d)} vs {len(y)}"
        )

    if len(X_3d) != len(y):
        raise ValueError(
            f"X_svd_3 and y_sample have different number of rows: {len(X_3d)} vs {len(y)}"
        )

    plot_true_labels_2d(
        X_2d=X_2d,
        y=y,
        output_path=data_plots_dir / "true_labels_svd_2.png",
    )

    plot_true_labels_3d(
        X_3d=X_3d,
        y=y,
        output_path=data_plots_dir / "true_labels_svd_3.png",
    )


# =========================
# GROUP 2 - BEST MODEL CLUSTERS
# =========================

def select_best_by_metric(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    work = df.copy()

    if metric not in work.columns:
        raise ValueError(f"Metric column '{metric}' not found in metrics table.")

    work = work.dropna(subset=[metric])

    if work.empty:
        raise ValueError(f"No valid rows for metric '{metric}'.")

    idx = work.groupby(["algorithm", "representation"])[metric].idxmax()
    best = work.loc[idx].copy()
    best["model_label"] = make_model_label(best)
    best = best.sort_values(metric, ascending=False).reset_index(drop=True)
    return best


def select_group_2_models(df: pd.DataFrame) -> pd.DataFrame:
    best_by_ari = select_best_by_metric(df, "ari")

    selected_rows = []

    full_sparse = best_by_ari[best_by_ari["representation"] == "full_sparse"]
    if not full_sparse.empty:
        selected_rows.append(full_sparse.iloc[0])

    reduced = best_by_ari[best_by_ari["representation"] != "full_sparse"]
    if not reduced.empty:
        selected_rows.append(reduced.iloc[0])

    idx_algo = best_by_ari.groupby("algorithm")["ari"].idxmax()
    algo_best = best_by_ari.loc[idx_algo].copy()

    for _, row in algo_best.iterrows():
        selected_rows.append(row)

    if not selected_rows:
        raise ValueError("No models selected for group 2 plots.")

    selected = pd.DataFrame(selected_rows).drop_duplicates().reset_index(drop=True)
    selected = selected.sort_values(["algorithm", "representation"]).reset_index(drop=True)
    return selected


def plot_cluster_labels_2d(
    X_2d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    if X_2d.ndim != 2 or X_2d.shape[1] != 2:
        raise ValueError(f"Expected X_2d shape (n_samples, 2), got {X_2d.shape}")

    if len(X_2d) != len(labels):
        raise ValueError(f"X_2d and labels length mismatch: {len(X_2d)} vs {len(labels)}")

    ensure_dir(output_path.parent)
    unique_labels = np.unique(labels)

    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        mask = labels == label
        legend_name = "noise (-1)" if label == -1 else str(label)
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=12,
            alpha=0.7,
            label=legend_name,
        )

    plt.title(title)
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.legend(title="Cluster", fontsize=8, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_labels_3d(
    X_3d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    if X_3d.ndim != 2 or X_3d.shape[1] != 3:
        raise ValueError(f"Expected X_3d shape (n_samples, 3), got {X_3d.shape}")

    if len(X_3d) != len(labels):
        raise ValueError(f"X_3d and labels length mismatch: {len(X_3d)} vs {len(labels)}")

    ensure_dir(output_path.parent)
    unique_labels = np.unique(labels)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    for label in unique_labels:
        mask = labels == label
        legend_name = "noise (-1)" if label == -1 else str(label)
        ax.scatter(
            X_3d[mask, 0],
            X_3d[mask, 1],
            X_3d[mask, 2],
            s=12,
            alpha=0.7,
            label=legend_name,
        )

    ax.set_title(title)
    ax.set_xlabel("SVD component 1")
    ax.set_ylabel("SVD component 2")
    ax.set_zlabel("SVD component 3")
    ax.legend(title="Cluster", fontsize=8, markerscale=1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_cluster_plot_title(row: pd.Series) -> str:
    algorithm = row["algorithm"]
    representation = row["representation"]
    ari = row["ari"]

    parts = [f"{algorithm} on {representation}", f"ARI={ari:.4f}"]

    if "n_clusters_found" in row and not pd.isna(row["n_clusters_found"]):
        parts.append(f"clusters={int(row['n_clusters_found'])}")

    if "n_noise" in row and not pd.isna(row["n_noise"]):
        parts.append(f"noise={int(row['n_noise'])}")

    return " | ".join(parts)


def generate_group_2_plots(
    processed_dir: Path,
    reports_dir: Path,
    metrics_csv_path: Path,
    plots_dir: Path,
) -> None:
    clusters_2d_dir = plots_dir / "clusters_2d"
    clusters_3d_dir = plots_dir / "clusters_3d"
    tables_dir = plots_dir / "tables"

    ensure_dir(clusters_2d_dir)
    ensure_dir(clusters_3d_dir)
    ensure_dir(tables_dir)

    X_2d = load_dense_array(processed_dir / "X_svd_2.npy")
    X_3d = load_dense_array(processed_dir / "X_svd_3.npy")
    df = load_metrics_csv(metrics_csv_path)

    selected = select_group_2_models(df)
    selected.to_csv(tables_dir / "group_2_selected_models.csv", index=False)

    for _, row in selected.iterrows():
        labels = load_labels_for_result(reports_dir, row)
        title = build_cluster_plot_title(row)

        base_name = sanitize_filename(f"{row['algorithm']}_{row['representation']}_ari")

        plot_cluster_labels_2d(
            X_2d=X_2d,
            labels=labels,
            output_path=clusters_2d_dir / f"{base_name}.png",
            title=title,
        )

        plot_cluster_labels_3d(
            X_3d=X_3d,
            labels=labels,
            output_path=clusters_3d_dir / f"{base_name}.png",
            title=title,
        )


# =========================
# GROUP 3 - METRICS PLOTS
# =========================

def plot_best_metric_bar(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    best = select_best_by_metric(df, metric)

    ensure_dir(output_path.parent)

    plt.figure(figsize=(14, 8))
    plt.bar(best["model_label"], best[metric])
    plt.title(title)
    plt.xlabel("Algorithm + representation")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_runtime_comparison(
    df: pd.DataFrame,
    output_path: Path,
    selection_metric: str = "ari",
) -> None:
    best = select_best_by_metric(df, selection_metric)

    if "runtime_sec" not in best.columns:
        raise ValueError("Column 'runtime_sec' not found in metrics table.")

    best = best.dropna(subset=["runtime_sec"]).copy()
    best = best.sort_values("runtime_sec", ascending=False).reset_index(drop=True)

    ensure_dir(output_path.parent)

    plt.figure(figsize=(14, 8))
    plt.bar(best["model_label"], best["runtime_sec"])
    plt.title(f"Runtime comparison (best models selected by {selection_metric.upper()})")
    plt.xlabel("Algorithm + representation")
    plt.ylabel("Runtime (seconds)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_group_3_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)

    summaries = {
        "best_by_ari.csv": select_best_by_metric(df, "ari"),
        "best_by_nmi.csv": select_best_by_metric(df, "nmi"),
        "best_by_silhouette.csv": select_best_by_metric(df, "silhouette"),
    }

    for filename, summary_df in summaries.items():
        summary_df.to_csv(output_dir / filename, index=False)


def generate_group_3_plots(metrics_csv_path: Path, plots_dir: Path) -> None:
    metrics_plots_dir = plots_dir / "metrics"
    metrics_tables_dir = plots_dir / "tables"
    ensure_dir(metrics_plots_dir)
    ensure_dir(metrics_tables_dir)

    df = load_metrics_csv(metrics_csv_path)

    plot_best_metric_bar(
        df=df,
        metric="ari",
        output_path=metrics_plots_dir / "best_ari_by_model.png",
        title="Best ARI by algorithm and representation",
        ylabel="ARI",
    )

    plot_best_metric_bar(
        df=df,
        metric="nmi",
        output_path=metrics_plots_dir / "best_nmi_by_model.png",
        title="Best NMI by algorithm and representation",
        ylabel="NMI",
    )

    plot_best_metric_bar(
        df=df,
        metric="silhouette",
        output_path=metrics_plots_dir / "best_silhouette_by_model.png",
        title="Best silhouette by algorithm and representation",
        ylabel="Silhouette",
    )

    plot_runtime_comparison(
        df=df,
        output_path=metrics_plots_dir / "runtime_comparison.png",
        selection_metric="ari",
    )

    build_group_3_summary_tables(df, metrics_tables_dir)


# =========================
# GROUP 4A - DBSCAN PLOTS
# =========================

def prepare_dbscan_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work[work["algorithm"] == "dbscan"].copy()

    required_columns = {"representation", "param_eps", "param_min_samples"}
    missing = required_columns - set(work.columns)
    if missing:
        raise ValueError(f"Missing required DBSCAN columns: {sorted(missing)}")

    work = work.dropna(subset=["param_eps", "param_min_samples"]).copy()
    work["param_eps"] = work["param_eps"].astype(float)
    work["param_min_samples"] = work["param_min_samples"].astype(int)

    return work


def plot_dbscan_heatmap(
    df: pd.DataFrame,
    representation: str,
    value_column: str,
    output_path: Path,
    title: str,
) -> None:
    rep_df = df[df["representation"] == representation].copy()

    if rep_df.empty:
        return

    rep_df = rep_df.dropna(subset=[value_column]).copy()
    if rep_df.empty:
        return

    pivot = rep_df.pivot_table(
        index="param_min_samples",
        columns="param_eps",
        values=value_column,
        aggfunc="max",
    )

    pivot = pivot.sort_index().sort_index(axis=1)

    if pivot.empty:
        return

    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("eps")
    ax.set_ylabel("min_samples")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns])

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(idx) for idx in pivot.index])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]

            if pd.isna(value):
                text = "nan"
            else:
                if value_column == "n_noise":
                    text = str(int(value))
                else:
                    text = f"{value:.4f}"

            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_column)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_dbscan_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)

    for representation in sorted(df["representation"].dropna().unique()):
        rep_df = df[df["representation"] == representation].copy()
        rep_df = rep_df.sort_values(["param_min_samples", "param_eps"]).reset_index(drop=True)
        rep_df.to_csv(output_dir / f"dbscan_{representation}_grid.csv", index=False)


def generate_dbscan_plots(metrics_csv_path: Path, plots_dir: Path) -> None:
    dbscan_dir = plots_dir / "dbscan"
    tables_dir = plots_dir / "tables"
    ensure_dir(dbscan_dir)
    ensure_dir(tables_dir)

    df = load_metrics_csv(metrics_csv_path)
    dbscan_df = prepare_dbscan_dataframe(df)

    if dbscan_df.empty:
        raise ValueError("No DBSCAN rows found in metrics CSV.")

    representations = sorted(dbscan_df["representation"].dropna().unique())

    for representation in representations:
        safe_rep = sanitize_filename(representation)

        plot_dbscan_heatmap(
            df=dbscan_df,
            representation=representation,
            value_column="ari",
            output_path=dbscan_dir / f"dbscan_{safe_rep}_ari_heatmap.png",
            title=f"DBSCAN ARI heatmap - {representation}",
        )

        plot_dbscan_heatmap(
            df=dbscan_df,
            representation=representation,
            value_column="nmi",
            output_path=dbscan_dir / f"dbscan_{safe_rep}_nmi_heatmap.png",
            title=f"DBSCAN NMI heatmap - {representation}",
        )

        plot_dbscan_heatmap(
            df=dbscan_df,
            representation=representation,
            value_column="n_noise",
            output_path=dbscan_dir / f"dbscan_{safe_rep}_noise_heatmap.png",
            title=f"DBSCAN noise heatmap - {representation}",
        )

    build_dbscan_summary_tables(dbscan_df, tables_dir)


# =========================
# GROUP 4B - K-SELECTION PLOTS
# =========================

def prepare_k_selection_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    algorithms = {"minibatch_kmeans", "agglomerative", "gmm", "birch"}

    work = df.copy()
    work = work[work["algorithm"].isin(algorithms)].copy()

    required_columns = {"algorithm", "representation", "param_n_clusters"}
    missing = required_columns - set(work.columns)
    if missing:
        raise ValueError(f"Missing required K-selection columns: {sorted(missing)}")

    work = work.dropna(subset=["param_n_clusters"]).copy()
    work["param_n_clusters"] = work["param_n_clusters"].astype(int)

    return work


def plot_k_selection_metric(
    df: pd.DataFrame,
    algorithm: str,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    algo_df = df[df["algorithm"] == algorithm].copy()

    if algo_df.empty:
        return

    algo_df = algo_df.dropna(subset=[metric]).copy()
    if algo_df.empty:
        return

    summary = (
        algo_df.groupby(["representation", "param_n_clusters"], as_index=False)[metric]
        .max()
        .sort_values(["representation", "param_n_clusters"])
    )

    ensure_dir(output_path.parent)

    plt.figure(figsize=(10, 7))

    for representation in sorted(summary["representation"].unique()):
        rep_df = summary[summary["representation"] == representation].copy()
        plt.plot(
            rep_df["param_n_clusters"],
            rep_df[metric],
            marker="o",
            label=representation,
        )

    plt.title(title)
    plt.xlabel("n_clusters")
    plt.ylabel(metric.upper() if metric != "silhouette" else "Silhouette")
    plt.xticks(sorted(summary["param_n_clusters"].unique()))
    plt.legend(title="Representation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_k_selection_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)

    for algorithm in sorted(df["algorithm"].dropna().unique()):
        algo_df = df[df["algorithm"] == algorithm].copy()
        algo_df = algo_df.sort_values(["representation", "param_n_clusters"]).reset_index(drop=True)
        algo_df.to_csv(output_dir / f"{algorithm}_k_grid.csv", index=False)


def generate_k_selection_plots(metrics_csv_path: Path, plots_dir: Path) -> None:
    k_selection_dir = plots_dir / "k_selection"
    tables_dir = plots_dir / "tables"
    ensure_dir(k_selection_dir)
    ensure_dir(tables_dir)

    df = load_metrics_csv(metrics_csv_path)
    k_df = prepare_k_selection_dataframe(df)

    if k_df.empty:
        raise ValueError("No K-selection rows found in metrics CSV.")

    algorithms = sorted(k_df["algorithm"].dropna().unique())
    metrics = ["ari", "nmi", "silhouette"]

    for algorithm in algorithms:
        for metric in metrics:
            plot_k_selection_metric(
                df=k_df,
                algorithm=algorithm,
                metric=metric,
                output_path=k_selection_dir / f"{algorithm}_k_vs_{metric}.png",
                title=f"{algorithm} - n_clusters vs {metric}",
            )

    build_k_selection_summary_tables(k_df, tables_dir)