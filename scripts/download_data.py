from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml
from scipy import sparse


def main() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading IMDB.drama from OpenML...")
    dataset = fetch_openml(data_id=273, as_frame=False)

    X = dataset.data      # sparse matrix
    y = dataset.target    # array

    feature_names = np.array(dataset.feature_names, dtype=str)

    if len(feature_names) != X.shape[1]:
        print("WARNING: Number of feature names does not match number of columns.")
        print("Using fallback feature names.")
        feature_names = np.array([f"feature_{i}" for i in range(X.shape[1])], dtype=str)


    print("\n=== BASIC INFO ===")
    print(f"X type: {type(X)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of feature names: {len(feature_names)}")

    print("\n=== SPARSITY INFO ===")
    if sparse.issparse(X):
        nnz = X.nnz
        total = X.shape[0] * X.shape[1]
        sparsity = 1 - (nnz / total)
        print(f"Non-zero elements: {nnz}")
        print(f"Sparsity: {sparsity:.4f}")
    else:
        print("X is not sparse")

    print("\n=== TARGET DISTRIBUTION ===")
    unique, counts = np.unique(y, return_counts=True)
    for val, cnt in zip(unique, counts):
        print(f"{val}: {cnt}")

    sparse.save_npz(raw_dir / "X_sparse.npz", X)
    np.save(raw_dir / "y.npy", y)
    np.save(raw_dir / "feature_names.npy", feature_names)

    print("\nSaved:")
    print(" - X_sparse.npz")
    print(" - y.npy")
    print(" - feature_names.npy")


if __name__ == "__main__":
    main()