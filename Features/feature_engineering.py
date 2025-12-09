import pandas_ta as ta
import pandas as pd
import numpy as np
import os
from Features.pandas_ta_indicators import add_all_pandasta_indicators
from Features.custom_indicators import add_tmo, add_rsilg_fe_gauss, add_fractal_pivots
import matplotlib.pyplot as plt

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

def add_atr_pivot_swing_labels(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    pivot_up_col: str = "pivot_up",
    pivot_down_col: str = "pivot_down",
    atr_length: int = 14,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    max_holding: int = 20,
):
    """
    ATR-based swing labeling anchored on fractal pivots.

    At each pivot:
      - For a pivot_down (local low): treat as potential LONG entry.
        * Entry price = close at pivot.
        * TP = entry + tp_mult * ATR
        * SL = entry - sl_mult * ATR
        * Look ahead up to max_holding bars:
            - If price hits TP before SL -> label +1 (good long pivot).
            - If price hits SL first or neither -> label 0.

      - For a pivot_up (local high): treat as potential SHORT entry.
        * Entry price = close at pivot.
        * TP = entry - tp_mult * ATR
        * SL = entry + sl_mult * ATR
        * Same logic; if TP hit first -> label -1 (good short pivot).

    Returns df with added columns:
        atr                      - ATR series
        atr_swing_label          - {-1, 0, +1} (short / none / long)
        atr_entry_price
        atr_exit_price
        atr_holding_bars
        atr_realized_return      - (exit / entry - 1)
    """

    high = df[high_col].to_numpy(dtype=float)
    low = df[low_col].to_numpy(dtype=float)
    close = df[close_col].to_numpy(dtype=float)
    pivot_up = df[pivot_up_col].to_numpy(dtype=int)
    pivot_down = df[pivot_down_col].to_numpy(dtype=int)

    n = len(df)

    # --- ATR ---
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=atr_length)
    atr = df["atr"].to_numpy()


    # --- outputs ---
    labels = np.zeros(n, dtype=float)          # -1, 0, +1
    entry_price = np.full(n, np.nan)
    exit_price = np.full(n, np.nan)
    holding_bars = np.full(n, np.nan)
    realized_ret = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        # LONG setup at pivot_down
        if pivot_down[i] == 1:
            side = 1
            ep = close[i]
            tp = ep + tp_mult * atr[i]
            sl = ep - sl_mult * atr[i]

            entry_price[i] = ep

            hit_label = 0
            hit_exit = ep
            hit_bars = 0

            for j in range(i + 1, min(i + 1 + max_holding, n)):
                # did we hit stop or target?
                if low[j] <= sl:
                    # stop first -> bad pivot
                    hit_label = 0
                    hit_exit = sl
                    hit_bars = j - i
                    break
                if high[j] >= tp:
                    # target first -> good long
                    hit_label = 1
                    hit_exit = tp
                    hit_bars = j - i
                    break

            labels[i] = hit_label * side   # 1 if good long, 0 otherwise
            exit_price[i] = hit_exit
            holding_bars[i] = hit_bars
            realized_ret[i] = (hit_exit / ep - 1.0)

        # SHORT setup at pivot_up
        elif pivot_up[i] == 1:
            side = -1
            ep = close[i]
            tp = ep - tp_mult * atr[i]   # profit target BELOW
            sl = ep + sl_mult * atr[i]   # stop ABOVE

            entry_price[i] = ep

            hit_label = 0
            hit_exit = ep
            hit_bars = 0

            for j in range(i + 1, min(i + 1 + max_holding, n)):
                # For shorts: TP is when low <= tp, SL when high >= sl
                if high[j] >= sl:
                    hit_label = 0        # stopped out
                    hit_exit = sl
                    hit_bars = j - i
                    break
                if low[j] <= tp:
                    hit_label = 1        # good short
                    hit_exit = tp
                    hit_bars = j - i
                    break

            labels[i] = hit_label * side   # -1 if good short, 0 otherwise
            exit_price[i] = hit_exit
            holding_bars[i] = hit_bars
            realized_ret[i] = (hit_exit / ep - 1.0)

    df["atr"] = atr
    df["atr_swing_label"] = labels          # -1 / 0 / +1
    df["atr_entry_price"] = entry_price
    df["atr_exit_price"] = exit_price
    df["atr_holding_bars"] = holding_bars
    df["atr_realized_return"] = realized_ret

    return df

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

    df = add_atr_pivot_swing_labels(
    df,
    high_col="high",
    low_col="low",
    close_col="close",
    pivot_up_col="pivot_up",
    pivot_down_col="pivot_down",
    atr_length=14,   # tweak later
    tp_mult=2,     # 2x ATR target
    sl_mult=1.0,     # 1x ATR stop
    max_holding=18,  # bars to look ahead
    )

    #drop any rows that have NA in these columns
    df = df.dropna(subset=["atr_swing_label", "close", "open", "high", "low", "volume"])

    # --- NEW: binary labels for two-model swing detector ---
    df["long_swing_label"] = (df["atr_swing_label"] == 1.0).astype(np.int64)
    df["short_swing_label"] = (df["atr_swing_label"] == -1.0).astype(np.int64)

    labels = ["atr_swing_label","atr_entry_price","atr_exit_price","atr_holding_bars","atr_realized_return" ]
    # Features = everything except labels
    feature_cols = [
        c for c in df.columns
        if c not in labels
    ]
    
    print()
    X = df[feature_cols].to_numpy(dtype=np.float32)

    y_long = df["long_swing_label"].to_numpy(dtype=np.int64)
    y_short = df["short_swing_label"].to_numpy(dtype=np.int64)

    close = df["close"].to_numpy(dtype=float)

    # Save outputs to a dedicated directory under the project path
    output_dir = os.path.join(global_file_path, "Data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "close_spy_daily.npy"), close)
    np.save(os.path.join(output_dir, "X_spy_daily.npy"), X)
    np.save(os.path.join(output_dir, "y_spy_daily_long.npy"), y_long)
    np.save(os.path.join(output_dir, "y_spy_daily_short.npy"), y_short)

    # Optional: keep feature names for reference
    with open(os.path.join(output_dir, "features_spy_daily.txt"), "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    print("X shape:", X.shape)
    print("y_long shape:", y_long.shape)
    print("y_short shape:", y_short.shape)
    
    print("describe:")
    print(df.describe().T)
    print("corr:")
    print(df.corr().abs())
    print("high_corr:")
    corr = df.corr().abs()
    high_corr = corr[corr > 0.98]  # overly similar features
    print(high_corr)
    print("atr_swing_label value counts:")
    df["atr_swing_label"].value_counts()
    print(df["atr_swing_label"].value_counts())
    print("close pct change:")
    print(df["close"].pct_change().abs() > 0.2)
    print("close pct change value counts:")
    print((df["close"].pct_change().abs() > 0.2).value_counts())
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    print("constant cols:")
    print(constant_cols)
    df = df.drop(columns=constant_cols)
    plt.plot(df["close"])
    plt.scatter(df.index[df["atr_swing_label"] == 1], df["close"][df["atr_swing_label"] == 1], c="green")
    plt.scatter(df.index[df["atr_swing_label"] == -1], df["close"][df["atr_swing_label"] == -1], c="red")
    plt.show()



if __name__ == "__main__":
    main()