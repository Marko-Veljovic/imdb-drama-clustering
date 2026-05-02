from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import joblib


def load_data(raw_dir: Path) -> tuple[sparse.spmatrix, np.ndarray]:
    x_path = raw_dir / "X_sparse.npz"
    y_path = raw_dir / "y.npy"

    if not x_path.exists():
        raise FileNotFoundError(
            f"Missing file: {x_path}. Run python -m scripts.download_data first."
        )

    if not y_path.exists():
        raise FileNotFoundError(
            f"Missing file: {y_path}. Run python -m scripts.download_data first."
        )

    X = sparse.load_npz(x_path)
    y = np.load(y_path, allow_pickle=True)

    return X, y


def load_feature_names(raw_dir: Path) -> np.ndarray:
    path = raw_dir / "feature_names.npy"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}. Run python -m scripts.download_data first."
        )

    return np.load(path, allow_pickle=True)


def remove_empty_columns(X: sparse.spmatrix) -> tuple[sparse.spmatrix, np.ndarray]:
    """
    Remove columns that are entirely zero.
    """
    X_csc = X.tocsc()
    nonzero_per_col = np.diff(X_csc.indptr)
    mask = nonzero_per_col > 0
    return X[:, mask], mask


def scale_data(X: sparse.spmatrix) -> tuple[sparse.spmatrix, MaxAbsScaler]:
    """
    Scale sparse data using MaxAbsScaler.
    This preserves sparsity.
    """
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def stratified_sample(
    X: sparse.spmatrix,
    y: np.ndarray,
    n_samples: int = 15000,
    random_state: int = 42,
) -> tuple[sparse.spmatrix, np.ndarray]:   
    """
    Create a stratified sample based on target labels.
    """
    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=n_samples,
        stratify=y,
        random_state=random_state,
    )
    return X_sample, y_sample


def apply_svd(
    X: sparse.spmatrix,
    n_components: int,
) -> tuple[np.ndarray, TruncatedSVD]:
    """
    Apply Truncated SVD to reduce dimensionality.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    return X_reduced, svd


def save_sparse(path: Path, X: sparse.spmatrix) -> None:
    sparse.save_npz(path, X)


def save_dense(path: Path, X: np.ndarray) -> None:
    np.save(path, X)


def save_model(path: Path, model: object) -> None:
    joblib.dump(model, path)