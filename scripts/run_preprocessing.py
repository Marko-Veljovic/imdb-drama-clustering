from pathlib import Path
from src.preprocessing import (
    load_data,
    load_feature_names,
    remove_empty_columns,
    scale_data,
    stratified_sample,
    apply_svd,
    save_sparse,
    save_dense,
    save_model,
)


def main():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    models_dir = Path("models/preprocessing")

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=== LOADING DATA ===")
    X, y = load_data(raw_dir)
    feature_names = load_feature_names(raw_dir)

    print(f"Original number of features: {X.shape[1]}")
    print(f"Original number of feature names: {len(feature_names)}")

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"Feature names length mismatch: {len(feature_names)} names, "
            f"but X has {X.shape[1]} columns."
        )

    print("=== REMOVING EMPTY COLUMNS ===")
    X, mask = remove_empty_columns(X)
    feature_names_filtered = feature_names[mask]

    print(f"Remaining features: {X.shape[1]}")
    print(f"Remaining feature names: {len(feature_names_filtered)}")

    save_dense(processed_dir / "feature_names_filtered.npy", feature_names_filtered)
    save_dense(processed_dir / "feature_mask.npy", mask)

    print("=== SCALING DATA ===")
    X_scaled, scaler = scale_data(X)

    save_sparse(processed_dir / "X_scaled.npz", X_scaled)
    save_model(models_dir / "scaler.pkl", scaler)

    print("=== SAMPLING DATA ===")
    X_sample, y_sample = stratified_sample(X_scaled, y, n_samples=15000)

    save_sparse(processed_dir / "X_sample.npz", X_sample)
    save_dense(processed_dir / "y_sample.npy", y_sample)

    print("=== APPLYING SVD ===")

    components_list = [100, 50, 20, 3, 2]

    for n in components_list:
        print(f"Reducing to {n} components...")

        X_reduced, svd = apply_svd(X_sample, n)

        save_dense(processed_dir / f"X_svd_{n}.npy", X_reduced)
        save_model(models_dir / f"svd_{n}.pkl", svd)

    print("=== DONE ===")


if __name__ == "__main__":
    main()