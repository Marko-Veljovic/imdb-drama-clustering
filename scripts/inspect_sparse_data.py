from pathlib import Path
from collections import Counter

import numpy as np
from scipy import sparse


def count_unique_sparse_rows(X):
    """
    Count unique rows in a sparse matrix

    Each row is represented by:
    - indices of non-zero columns
    - corresponding non-zero values

    This checks exact duplicate sparse rows.
    """
    X_csr = X.tocsr(copy=True)
    X_csr.sort_indices()

    indptr = X_csr.indptr
    indices = X_csr.indices
    data = X_csr.data

    row_counts = Counter()

    for i in range(X_csr.shape[0]):
        start = indptr[i]
        end = indptr[i + 1]

        # A sparse row is uniquely determined by its non-zero column indices
        # and values. Convert them to bytes so they can be used as a Counter key.
        row_key = (
            indices[start:end].tobytes(),
            data[start:end].tobytes(),
        )

        row_counts[row_key] += 1

    unique_rows = len(row_counts)
    duplicate_rows = X_csr.shape[0] - unique_rows
    duplicate_patterns = sum(1 for count in row_counts.values() if count > 1)
    max_duplicate_count = max(row_counts.values()) if row_counts else 0

    return unique_rows, duplicate_rows, duplicate_patterns, max_duplicate_count


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

    print("\n=== FULL DUPLICATE ROW CHECK ===")
    unique_rows, duplicate_rows, duplicate_patterns, max_duplicate_count = (
        count_unique_sparse_rows(X)
    )

    print(f"Rows checked: {X.shape[0]}")
    print(f"Unique rows: {unique_rows}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Duplicate row patterns: {duplicate_patterns}")
    print(f"Maximum repetitions of one row: {max_duplicate_count}")

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