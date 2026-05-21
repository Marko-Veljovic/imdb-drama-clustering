from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_feature_name(feature_name: str) -> str:
    """
    Remove quotation marks and white spaces from feature name and lower all letters
    """
    return str(feature_name).strip().lower().strip("'\"`")


def extract_candidate_word(feature_name: str) -> str:
    """
    Example: "word_the", "tfidf_the", "attr_the" -> "the"
    """
    normalized = normalize_feature_name(feature_name)

    if normalized in ENGLISH_STOP_WORDS:
        return normalized

    parts = re.split(r"[\s:=/|,;\[\]\(\)\{\}_\-]+", normalized)
    parts = [part for part in parts if part]

    known_prefixes = {
        "word",
        "token",
        "term",
        "tfidf",
        "count",
        "feature",
        "attr",
        "attribute",
    }

    if len(parts) >= 2 and parts[0] in known_prefixes:
        return parts[-1]

    return normalized


def find_stop_word_columns(feature_names: np.ndarray) -> pd.DataFrame:
    rows = []

    for column_index, feature_name in enumerate(feature_names):
        candidate_word = extract_candidate_word(feature_name)

        if candidate_word in ENGLISH_STOP_WORDS:
            rows.append(
                {
                    "column_index": column_index,
                    "feature_name": str(feature_name),
                    "matched_stop_word": candidate_word,
                }
            )

    return pd.DataFrame(rows)


def add_stop_word_column_statistics(
    X: sparse.csr_matrix,
    stop_word_columns_df: pd.DataFrame,
) -> pd.DataFrame:
    if stop_word_columns_df.empty:
        return stop_word_columns_df

    X_csc = X.tocsc()
    stop_indices = stop_word_columns_df["column_index"].to_numpy(dtype=int)

    document_frequency_total = np.diff(X_csc.indptr)
    term_frequency_total = np.asarray(X.sum(axis=0)).ravel()

    result = stop_word_columns_df.copy()

    result["document_frequency_total"] = document_frequency_total[stop_indices]
    result["document_frequency_total_pct"] = (
        result["document_frequency_total"] / X.shape[0]
    )
    result["term_frequency_total"] = term_frequency_total[stop_indices]

    result = result.sort_values(
        by=["document_frequency_total", "term_frequency_total"],
        ascending=False,
    )

    return result


def remove_stop_word_columns(
    X: sparse.csr_matrix,
    feature_names: np.ndarray,
    stop_word_columns_df: pd.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    stop_indices = set(stop_word_columns_df["column_index"].to_numpy(dtype=int))

    keep_mask = np.array(
        [column_index not in stop_indices for column_index in range(X.shape[1])],
        dtype=bool,
    )

    X_without_stop_words = X[:, keep_mask].tocsr()
    feature_names_without_stop_words = feature_names[keep_mask]

    return X_without_stop_words, feature_names_without_stop_words, keep_mask


def write_summary(
    output_path: Path,
    x_path: Path,
    feature_names_path: Path,
    X: sparse.csr_matrix,
    stop_word_columns_df: pd.DataFrame,
    X_without_stop_words: sparse.csr_matrix,
    stop_word_columns_csv_path: Path,
    x_without_stop_words_path: Path,
    feature_names_without_stop_words_path: Path,
    feature_mask_without_stop_words_path: Path,
) -> None:
    n_rows = X.shape[0]
    n_features = X.shape[1]
    n_stop_word_columns = len(stop_word_columns_df)

    if n_stop_word_columns > 0:
        stop_indices = stop_word_columns_df["column_index"].to_numpy(dtype=int)
        X_stop_words = X[:, stop_indices]

        rows_with_at_least_one_stop_word = int(
            (X_stop_words.getnnz(axis=1) > 0).sum()
        )
        stop_word_nonzero_values = int(X_stop_words.nnz)
        stop_word_value_sum = float(X_stop_words.sum())
    else:
        rows_with_at_least_one_stop_word = 0
        stop_word_nonzero_values = 0
        stop_word_value_sum = 0.0

    lines = []

    lines.append("=== STOP WORD ANALYSIS ===")
    lines.append("")
    lines.append("Input files:")
    lines.append(f"- X: {x_path}")
    lines.append(f"- feature names: {feature_names_path}")

    lines.append("")
    lines.append("Dataset shape:")
    lines.append(f"- Number of rows: {n_rows}")
    lines.append(f"- Number of columns/features: {n_features}")
    lines.append(f"- Number of non-zero values: {X.nnz}")

    lines.append("")
    lines.append("Stop word columns:")
    lines.append(f"- Number of stop word columns: {n_stop_word_columns}")
    lines.append(f"- Percentage of all columns: {n_stop_word_columns / n_features:.4%}")
    lines.append(f"- Rows with at least one stop word: {rows_with_at_least_one_stop_word}")
    lines.append(
        "- Percentage of rows with at least one stop word: "
        f"{rows_with_at_least_one_stop_word / n_rows:.4%}"
    )
    lines.append(f"- Non-zero values in stop word columns: {stop_word_nonzero_values}")
    lines.append(f"- Sum of values in stop word columns: {stop_word_value_sum:.4f}")

    lines.append("")
    lines.append("Shape after optional stop word removal:")
    lines.append(f"- Original X shape: {X.shape}")
    lines.append(f"- X without stop words shape: {X_without_stop_words.shape}")
    lines.append(f"- Removed columns: {X.shape[1] - X_without_stop_words.shape[1]}")
    lines.append(f"- Non-zero values after removal: {X_without_stop_words.nnz}")
    lines.append(f"- Removed non-zero values: {X.nnz - X_without_stop_words.nnz}")

    if n_stop_word_columns > 0:
        lines.append("")
        lines.append("Stop word columns sorted by document frequency:")

        for _, row in stop_word_columns_df.iterrows():
            lines.append(
                "- "
                f"column={int(row['column_index'])}, "
                f"feature='{row['feature_name']}', "
                f"matched='{row['matched_stop_word']}', "
                f"document_frequency={int(row['document_frequency_total'])}, "
                f"document_frequency_pct={row['document_frequency_total_pct']:.4%}, "
                f"term_frequency={row['term_frequency_total']:.4f}"
            )

    lines.append("")
    lines.append("Saved outputs:")
    lines.append(f"- {output_path}")
    lines.append(f"- {stop_word_columns_csv_path}")
    lines.append(f"- {x_without_stop_words_path}")
    lines.append(f"- {feature_names_without_stop_words_path}")
    lines.append(f"- {feature_mask_without_stop_words_path}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    raw_dir = Path("data/raw")
    output_dir = Path("results/stop_words_analysis")
    optional_data_dir = Path("data/optional")

    x_path = raw_dir / "X_sparse.npz"
    feature_names_path = raw_dir / "feature_names.npy"

    stop_word_columns_csv_path = output_dir / "stop_word_columns.csv"
    summary_path = output_dir / "summary.txt"

    x_without_stop_words_path = optional_data_dir / "X_without_stop_words.npz"
    feature_names_without_stop_words_path = (
        optional_data_dir / "feature_names_without_stop_words.npy"
    )
    feature_mask_without_stop_words_path = (
        optional_data_dir / "feature_mask_without_stop_words.npy"
    )

    ensure_dir(output_dir)
    ensure_dir(optional_data_dir)

    if not x_path.exists():
        raise FileNotFoundError(f"Missing file: {x_path}")
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Missing file: {feature_names_path}")

    print("=== STOP WORD ANALYSIS ===")

    print("\n=== LOADING DATA ===")
    X = sparse.load_npz(x_path).tocsr()
    feature_names = np.load(feature_names_path, allow_pickle=True)

    print(f"X shape: {X.shape}")
    print(f"Number of feature names: {len(feature_names)}")

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"X and feature names mismatch: X has {X.shape[1]} columns, "
            f"but feature_names has {len(feature_names)} values."
        )

    print("\n=== FINDING STOP WORD COLUMNS ===")
    stop_word_columns_df = find_stop_word_columns(feature_names)
    stop_word_columns_df = add_stop_word_column_statistics(
        X=X,
        stop_word_columns_df=stop_word_columns_df,
    )

    print(f"Stop word columns found: {len(stop_word_columns_df)}")

    print("\n=== SAVING OPTIONAL DATA WITHOUT STOP WORD COLUMNS ===")
    X_without_stop_words, feature_names_without_stop_words, keep_mask = (
        remove_stop_word_columns(
            X=X,
            feature_names=feature_names,
            stop_word_columns_df=stop_word_columns_df,
        )
    )

    sparse.save_npz(x_without_stop_words_path, X_without_stop_words)
    np.save(feature_names_without_stop_words_path, feature_names_without_stop_words)
    np.save(feature_mask_without_stop_words_path, keep_mask)

    print(f"Original X shape: {X.shape}")
    print(f"X without stop words shape: {X_without_stop_words.shape}")
    print(f"Removed columns: {X.shape[1] - X_without_stop_words.shape[1]}")

    print("\n=== SAVING REPORTS ===")
    stop_word_columns_df.to_csv(stop_word_columns_csv_path, index=False)

    write_summary(
        output_path=summary_path,
        x_path=x_path,
        feature_names_path=feature_names_path,
        X=X,
        stop_word_columns_df=stop_word_columns_df,
        X_without_stop_words=X_without_stop_words,
        stop_word_columns_csv_path=stop_word_columns_csv_path,
        x_without_stop_words_path=x_without_stop_words_path,
        feature_names_without_stop_words_path=feature_names_without_stop_words_path,
        feature_mask_without_stop_words_path=feature_mask_without_stop_words_path,
    )

    print(f"Saved: {summary_path}")
    print(f"Saved: {stop_word_columns_csv_path}")
    print(f"Saved: {x_without_stop_words_path}")
    print(f"Saved: {feature_names_without_stop_words_path}")
    print(f"Saved: {feature_mask_without_stop_words_path}")

    print("\nDONE")


if __name__ == "__main__":
    main()