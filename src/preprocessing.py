from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import joblib


def load_data(raw_dir: Path):
    X = sparse.load_npz(raw_dir / "X_sparse.npz")
    y = np.load(raw_dir / "y.npy", allow_pickle=True)
    return X, y


def remove_empty_columns(X):
    """
    Remove columns that are entirely zero.
    """
    X_csc = X.tocsc()
    nonzero_per_col = np.diff(X_csc.indptr)
    mask = nonzero_per_col > 0
    return X[:, mask], mask


def scale_data(X):
    """
    Scale sparse data using MaxAbsScaler.
    This preserves sparsity.
    """
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def stratified_sample(X, y, n_samples=15000, random_state=42):
    """
    Create a stratified sample based on target labels.
    """
    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=n_samples,
        stratify=y,
        random_state=random_state
    )
    return X_sample, y_sample


def apply_svd(X, n_components):
    """
    Apply Truncated SVD to reduce dimensionality.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    return X_reduced, svd


def save_sparse(path, X):
    sparse.save_npz(path, X)


def save_dense(path, X):
    np.save(path, X)


def save_model(path, model):
    joblib.dump(model, path)