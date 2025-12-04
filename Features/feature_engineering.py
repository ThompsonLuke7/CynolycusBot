import pandas_ta as ta
import pandas as pd
import numpy as np
import os
from ta_all_features import add_all_pandasta_indicators

global_file_path = "C:/Users/luket/CynolycusBot"

def load_spy_csv() -> pd.DataFrame:
    path = os.path.join(global_file_path, "Data", "spy_data.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=[0])

    # 1) Keep only the columns we care about initially (drop any 'Symbol', etc.)
    keep_cols = [c for c in df.columns if c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df = df[keep_cols]

    # 2) Rename to lowercase OHLCV for pandas_ta
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # 3) Force OHLCV columns to numeric
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def make_labels(df: pd.DataFrame) -> pd.Series:
    # Shift close by -1 to represent "tomorrow's" close
    future_close = df["Close"].shift(-1)
    direction = (future_close > df["Close"]).astype(int).iloc[:-1].values
    return direction

def main():
    df = load_spy_csv()

    # Add ALL pandas_ta indicators except statistics & performance
    df = add_all_pandasta_indicators(df, verbose=True)

    # Drop NaNs introduced by indicators
    df = df.dropna()

    numeric_df = df.select_dtypes(include=["number"])

    # label: next-day direction based on close
    future_close = numeric_df["close"].shift(-1)
    numeric_df["target"] = (future_close > numeric_df["close"]).astype(int)

    numeric_df = numeric_df.dropna()

    # Features = all numeric columns except target
    feature_cols = [c for c in numeric_df.columns if c != "target"]
    X = numeric_df[feature_cols].to_numpy(dtype=np.float32)
    y = numeric_df["target"].to_numpy(dtype=np.int64)

    np.save("X_spy_daily.npy", X)
    np.save("y_spy_daily.npy", y)

    # Optional: keep feature names for reference
    with open("features_spy_daily.txt", "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

if __name__ == "__main__":
    main()