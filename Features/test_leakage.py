from pathlib import Path

import pandas as pd


LABEL_COLS = [
    "atr_swing_label",
    "long_swing_label",
    "short_swing_label",
    "atr_cont_label",
    "long_cont_label",
    "short_cont_label",
]


def load_base():
    base_dir = Path(__file__).resolve().parent.parent / "Data" / "processed" / "base"
    X = pd.read_parquet(base_dir / "X_spy_daily.parquet")
    labels = pd.read_parquet(base_dir / "labels_spy_daily.parquet")
    features_txt = base_dir / "features_spy_daily.txt"
    feature_cols = (
        [line.strip() for line in features_txt.read_text().splitlines() if line.strip()]
        if features_txt.exists()
        else list(X.columns)
    )
    return X, labels, feature_cols


def find_identical_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    return [c for c in df.columns if c != label_col and df[c].equals(df[label_col])]


def high_corr_with_label(
    df: pd.DataFrame, label_col: str, threshold: float = 0.98
) -> pd.Series:
    corr = df.corr()[label_col].drop(label_col)
    return corr[abs(corr) >= threshold].sort_values(ascending=False)


def main():
    X_df, labels_df, feature_cols = load_base()
    df = pd.concat([X_df, labels_df], axis=1).reset_index(drop=True)

    print(f"Rows: {len(df)}, features: {X_df.shape[1]}, labels: {labels_df.shape[1]}")

    # 1) Verify labels are not inside the feature set
    overlap = set(feature_cols) & set(labels_df.columns)
    if overlap:
        print("\n[LEAK] Label columns still present in feature list:", overlap)
    else:
        print("\n[OK] No label columns present in feature list.")

    # 2) Exact-duplicate columns of each label
    for label in LABEL_COLS:
        if label not in df.columns:
            continue
        identical = find_identical_columns(df, label)
        if identical:
            print(f"[LEAK] Columns identical to {label}: {identical}")

    # 3) High correlation with each label
    for label in LABEL_COLS:
        if label not in df.columns:
            continue
        corr_hits = high_corr_with_label(df, label, threshold=0.99)
        if not corr_hits.empty:
            print(f"\n[WARN] High |corr| (>=0.99) with {label}:")
            print(corr_hits.head(20))

    # 4) Low cardinality features that probably should not be scaled or may be flags
    low_card = [c for c in X_df.columns if X_df[c].nunique(dropna=True) <= 5]
    if low_card:
        print("\n[Info] Low-cardinality (possible flag) features:")
        print(low_card[:50])  # limit output


if __name__ == "__main__":
    main()
