import pandas_ta as ta
import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Momentum
    df["rsi_14"]   = ta.rsi(df["Close"], length=14)
    df["stoch_k"]  = ta.stoch(df["High"], df["Low"], df["Close"]).iloc[:, 0]  # %K
    df["stoch_d"]  = ta.stoch(df["High"], df["Low"], df["Close"]).iloc[:, 1]  # %D
    df["macd"]     = ta.macd(df["Close"]).iloc[:, 0]
    df["macd_sig"] = ta.macd(df["Close"]).iloc[:, 1]
    df["macd_hist"]= ta.macd(df["Close"]).iloc[:, 2]

    # Trend
    df["ema_10"]   = ta.ema(df["Close"], length=10)
    df["ema_20"]   = ta.ema(df["Close"], length=20)
    df["ema_50"]   = ta.ema(df["Close"], length=50)
    df["sma_20"]   = ta.sma(df["Close"], length=20)
    df["sma_50"]   = ta.sma(df["Close"], length=50)

    # Volatility
    bbands = ta.bbands(df["Close"], length=20, std=2)
    df["bb_lower"]  = bbands.iloc[:, 0]   # first column
    df["bb_middle"] = bbands.iloc[:, 1]   # second
    df["bb_upper"]  = bbands.iloc[:, 2]   # third
    df["bb_pct"]   = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["atr_14"]   = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Volume / money flow
    df["mfi_14"]   = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14)
    df["obv"]      = ta.obv(df["Close"], df["Volume"])

    # Returns (simple + log)
    df["ret_1"]    = df["Close"].pct_change(1)
    df["ret_5"]    = df["Close"].pct_change(5)
    df["ret_10"]   = df["Close"].pct_change(10)

    # Candle structure
    df["body"]     = df["Close"] - df["Open"]
    df["range"]    = df["High"] - df["Low"]
    df["upper_wick"]= df["High"] - df[["Close","Open"]].max(axis=1)
    df["lower_wick"]= df[["Close","Open"]].min(axis=1) - df["Low"]

    return df

def make_labels(df: pd.DataFrame) -> pd.Series:
    # Shift close by -1 to represent "tomorrow's" close
    future_close = df["Close"].shift(-1)
    direction = (future_close > df["Close"]).astype(int)
    return direction

def main():
    df = pd.read_csv("spy_data.csv", index_col=0, parse_dates=[0])
    # Force numeric columns
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_indicators(df)
     # Create label
    df["target"] = make_labels(df)

    # Drop last row (no future close) and any rows with NaNs from indicators
    df = df.dropna()

    # Choose which columns are features
    feature_cols = [
        "rsi_14", "stoch_k", "stoch_d",
        "macd", "macd_sig", "macd_hist",
        "ema_10", "ema_20", "ema_50",
        "sma_20", "sma_50",
        "bb_upper", "bb_middle", "bb_lower", "bb_pct",
        "atr_14", "mfi_14", "obv",
        "ret_1", "ret_5", "ret_10",
        "body", "range", "upper_wick", "lower_wick",
    ]

    X = df[feature_cols].values
    y = df["target"].values

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