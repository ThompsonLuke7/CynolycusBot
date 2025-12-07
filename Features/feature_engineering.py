import pandas_ta as ta
import pandas as pd
import numpy as np
import os
from pandas_ta_indicators import add_all_pandasta_indicators
from custom_indicators import add_tmo, add_rsilg_fe_gauss, add_fractal_pivots

global_file_path = "C:/Users/luket/CynolycusBot"

def load_spy_csv() -> pd.DataFrame:
    path = os.path.join(global_file_path, "Data", "spy_data.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=[0], date_format="%Y-%m-%d")
    df.index = pd.to_datetime(df.index, errors="coerce", format="%Y-%m-%d")
    # 2) Drop rows where index is NaT (these are 'Ticker', 'Date', etc.)
    df = df[df.index.notna()]
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
    future_close = df["close"].shift(-1)
    direction = (future_close > df["close"]).astype(int).iloc[:-1].values
    return direction

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df.index.dayofweek        # 0 = Monday … 4 = Friday
    df["day_of_month"] = df.index.day            # sometimes helps
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["is_month_end"] = df.index.is_month_end.astype(int)
    df["is_month_start"] = df.index.is_month_start.astype(int)
    return df

def add_all_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_tmo(df)
    df = add_rsilg_fe_gauss(df)
    df = add_fractal_pivots(df)
    return df

def main():
    df = load_spy_csv()
    print(df.head())
    print(df.info())
    
    # Add ALL pandas_ta indicators except statistics & performance
    df = add_all_pandasta_indicators(df, verbose=True)
    print(df.head())
    # Drop NaNs introduced by indicators
    # 1) Drop columns that are completely NaN
    df = df.dropna(axis=1, how="all")
    # Add date features
    df = add_all_custom_indicators(df)
    df = add_date_features(df)

    # 2) Define your label BEFORE cleaning rows
    future_close = df["close"].shift(-1)
    df["target"] = (future_close > df["close"]).astype(float)  # or int

    # 3) Drop rows that don't have a label or core price info
    df = df.dropna(subset=["target", "close", "open", "high", "low", "volume"])
    print(df.iloc[200:205])
    # Features = all numeric columns except target
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["target"].to_numpy(dtype=np.int64)

    close = df["close"].to_numpy(dtype=float)

    np.save("close_spy_daily.npy", close)
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