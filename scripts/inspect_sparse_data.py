from pathlib import Path
import numpy as np
from scipy import sparse


def main() -> None:
    raw_dir = Path("data/raw")

    x_path = raw_dir / "X_sparse.npz"
    y_path = raw_dir / "y.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing file: {y_path}")

    X = sparse.load_npz(x_path)
    y = np.load(y_path, allow_pickle=True)

    print("=== BASIC INFO ===")
    print(f"X type: {type(X)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\n=== SPARSITY INFO ===")
    nnz = X.nnz
    total = X.shape[0] * X.shape[1]
    sparsity = 1 - (nnz / total)
    print(f"Non-zero elements: {nnz}")
    print(f"Sparsity: {sparsity:.6f}")

    print("\n=== SAMPLE-BASED DUPLICATE ROW CHECK ===")
    # Full duplicate detection may be expensive for a large sparse matrix.
    # As a lightweight diagnostic, we inspect a smaller sample of rows.
    sample_size = min(5000, X.shape[0])

    # Converting only a small sample to dense is acceptable and simplifies
    # the duplicate check with NumPy.
    X_sample = X[:sample_size].toarray()
    unique_rows = np.unique(X_sample, axis=0).shape[0]

    print(f"Sampled rows checked: {sample_size}")
    print(f"Unique rows in sample: {unique_rows}")
    print(f"Duplicate rows in sample: {sample_size - unique_rows}")

    print("\n=== COLUMN ACTIVITY CHECK ===")
    # CSC format is more convenient for column-wise analysis.
    X_csc = X.tocsc()

    # For each column, compute how many rows contain a non-zero value.
    nonzero_per_col = np.diff(X_csc.indptr)

    # Columns with zero non-zero entries are completely empty and carry no information.
    zero_only_cols = np.sum(nonzero_per_col == 0)
    print(f"Columns with all zeros: {zero_only_cols}")

    # Columns that are non-zero in every row may also be weakly informative,
    # because they do not help distinguish between instances.
    always_nonzero_cols = np.sum(nonzero_per_col == X.shape[0])
    print(f"Columns non-zero in all rows: {always_nonzero_cols}")

    print("\n=== TARGET DISTRIBUTION ===")
    unique, counts = np.unique(y, return_counts=True)
    for val, cnt in zip(unique, counts):
        print(f"{val}: {cnt}")


if __name__ == "__main__":
    main()