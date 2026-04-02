from pathlib import Path

from src.visualization import (
    generate_group_1_plots,
    generate_group_2_plots,
    generate_group_3_plots,
    generate_dbscan_plots,
    generate_k_selection_plots,
)


def main() -> None:
    processed_dir = Path("data/processed")
    plots_dir = Path("results/plots")
    reports_dir = Path("results/cluster_reports")
    metrics_csv_path = Path("results/metrics/clustering_metrics.csv")

    print("=== GENERATING VISUALIZATIONS ===")
    print(f"Processed dir: {processed_dir}")
    print(f"Reports dir: {reports_dir}")
    print(f"Plots dir: {plots_dir}")
    print(f"Metrics CSV: {metrics_csv_path}")

    print("\n--- Group 1: data visualization (2D/3D) ---")
    generate_group_1_plots(
        processed_dir=processed_dir,
        plots_dir=plots_dir,
    )
    print(f"Saved group 1 plots to: {plots_dir / 'data'}")

    print("\n--- Group 2: best model cluster visualizations ---")
    generate_group_2_plots(
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        metrics_csv_path=metrics_csv_path,
        plots_dir=plots_dir,
    )
    print(f"Saved group 2 plots to: {plots_dir / 'clusters_2d'}")
    print(f"Saved group 2 plots to: {plots_dir / 'clusters_3d'}")

    print("\n--- Group 3: metrics comparison ---")
    generate_group_3_plots(
        metrics_csv_path=metrics_csv_path,
        plots_dir=plots_dir,
    )
    print(f"Saved group 3 plots to: {plots_dir / 'metrics'}")
    print(f"Saved group 2/3 tables to: {plots_dir / 'tables'}")

    print("\n--- Group 4A: DBSCAN parameter analysis ---")
    generate_dbscan_plots(
        metrics_csv_path=metrics_csv_path,
        plots_dir=plots_dir,
    )
    print(f"Saved DBSCAN plots to: {plots_dir / 'dbscan'}")

    print("\n--- Group 4B: K-selection analysis ---")
    generate_k_selection_plots(
        metrics_csv_path=metrics_csv_path,
        plots_dir=plots_dir,
    )
    print(f"Saved K-selection plots to: {plots_dir / 'k_selection'}")

    print("\nDONE")


if __name__ == "__main__":
    main()