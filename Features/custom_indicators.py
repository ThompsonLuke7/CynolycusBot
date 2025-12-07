import numpy as np
import pandas as pd
from tqdm import tqdm

def add_fractal_pivots(
    df,
    sequence_count=5,   # same as ThinkScript default in your version
    vol_length=60,
    num_dev=1.0,
    allow_negative=False,
):
    """
    Mobius / AlphaInvestor-style fractal pivots + SuperPivots
    translated from the RSIL_SelfAdj_wFE_wSPivots code.

    Requires columns: 'high', 'low', 'volume'

    Adds:
        pivot_up          : 1 if fractal high at bar, else 0
        pivot_down        : 1 if fractal low at bar, else 0
        super_pivot_up    : 1 if pivot_up and relative volume > num_dev
        super_pivot_down  : 1 if pivot_down and relative volume > num_dev
        rel_vol_z         : relative volume z-score (clipped at 0 if allow_negative=False)
    """

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    vols = df["volume"].to_numpy(dtype=float)

    n = len(df)
    max_side_length = sequence_count + 10  # same as script

    pivot_up = np.zeros(n, dtype=int)
    pivot_down = np.zeros(n, dtype=int)

    # --- helper to compute "side" count like the fold logic ---
    def side_count(data, idx, direction, is_up):
        """
        direction: -1 = right side (past), +1 = left side (future)
        is_up: True for upData logic, False for downData logic
        Returns count or -1 if invalid per script.
        """
        count = 0
        for k in range(1, max_side_length + 1):
            j = idx + direction * k
            if j < 0 or j >= n:
                break

            val = data[idx]
            other = data[j]

            if is_up:
                # UpFractal side logic
                if other > val or (other == val and count == 0):
                    return -1
                elif other < val:
                    count += 1
            else:
                # DownFractal side logic
                if other < val or (other == val and count == 0):
                    return -1
                elif other > val:
                    count += 1

            if count == sequence_count:
                break
        return count

    # --- compute fractal pivots ---
    for t in tqdm(range(n), desc="Computing Fractal Pivots"):
        up_right = side_count(highs, t, direction=-1, is_up=True)
        up_left = side_count(highs, t, direction=+1, is_up=True)
        if up_right == sequence_count and up_left == sequence_count:
            pivot_up[t] = 1

        dn_right = side_count(lows, t, direction=-1, is_up=False)
        dn_left = side_count(lows, t, direction=+1, is_up=False)
        if dn_right == sequence_count and dn_left == sequence_count:
            pivot_down[t] = 1

    # --- relative volume z-score, same formula as script ---
    vol_s = pd.Series(vols, index=df.index)
    vol_mean = vol_s.rolling(vol_length, min_periods=vol_length).mean()
    vol_std = vol_s.rolling(vol_length, min_periods=vol_length).std()

    raw_rel_vol = (vol_s - vol_mean) / vol_std
    if not allow_negative:
        rel_vol = raw_rel_vol.clip(lower=0)
    else:
        rel_vol = raw_rel_vol

    # --- SuperPivots: pivot + rel_vol > num_dev ---
    super_up = ((rel_vol > num_dev) & (pivot_up == 1)).astype(int)
    super_down = ((rel_vol > num_dev) & (pivot_down == 1)).astype(int)

    # --- attach to df ---
    df["rel_vol_z"] = rel_vol.to_numpy()
    df["pivot_up"] = pivot_up
    df["pivot_down"] = pivot_down
    df["super_pivot_up"] = super_up
    df["super_pivot_down"] = super_down

    return df

def add_rsilg_fe_gauss(df, nFE=8, glength=13, beta_dev=8):
    """
    RSI-Laguerre Self Adjusting With Fractal Energy (Gaussian price filter version)

    Translated from ThinkScript "RSILg_FE_Gssn1".

    Requires df to have columns: 'open', 'high', 'low', 'close'.
    Adds columns:
        rsilg_gamma      - Fractal Energy (0 = linear/trending, 1 = non-linear/compressed)
        rsilg_rsi        - Laguerre RSI (0..1)
        rsilg_os         - constant oversold level (0.2)
        rsilg_ob         - constant overbought level (0.8)
        rsilg_mid        - 0.5 midline
        rsilg_fe_high    - 1 if gamma >= 0.618, else 0
        rsilg_fe_low     - 1 if gamma <= 0.382, else 0
        rsilg_rsi_hi     - 1 if RSI >= 0.8
        rsilg_rsi_lo     - 1 if RSI <= 0.2
    """

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    n = len(df)

    # ---- Gaussian filter coefficients ----
    w = 2.0 * np.pi / glength
    beta = (1 - np.cos(w)) / (np.power(1.414, 2.0 / beta_dev) - 1.0)
    alpha = -beta + np.sqrt(beta * beta + 2 * beta)

    def gauss_filter(x):
        g = np.zeros_like(x)
        a = alpha
        one_minus = 1.0 - a
        for t in range(n):
            g0 = (a ** 4) * x[t]
            if t >= 1:
                g0 += 4 * one_minus * g[t - 1]
            if t >= 2:
                g0 += -6 * (one_minus ** 2) * g[t - 2]
            if t >= 3:
                g0 += 4 * (one_minus ** 3) * g[t - 3]
            if t >= 4:
                g0 += -(one_minus ** 4) * g[t - 4]
            g[t] = g0
        return g

    Go = gauss_filter(o)
    Gh = gauss_filter(h)
    Gl = gauss_filter(l)
    Gc = gauss_filter(c)

    # ---- Smoothed OHLC used in RSI/FE logic ----
    Gc_shift = np.roll(Gc, 1)
    Gc_shift[0] = Gc[0]

    o2 = (Go + Gc_shift) / 2.0
    h2 = np.maximum(Gh, Gc_shift)
    l2 = np.minimum(Gl, Gc_shift)
    c2 = (o2 + h2 + l2 + Gc) / 4.0

    # ---- Fractal Energy gamma ----
    rng_bar = h2 - l2  # per-bar range
    rng_bar_s = pd.Series(rng_bar, index=df.index)

    sum_range = rng_bar_s.rolling(nFE, min_periods=nFE).sum()
    highest_high = pd.Series(Gh, index=df.index).rolling(nFE, min_periods=nFE).max()
    lowest_low = pd.Series(Gl, index=df.index).rolling(nFE, min_periods=nFE).min()

    denom = highest_high - lowest_low
    ratio = sum_range / denom.replace(0, np.nan)
    gamma = np.log(ratio) / np.log(nFE)
    gamma = gamma.to_numpy()

    # ---- Laguerre RSI with gamma ----
    L0 = np.zeros(n)
    L1 = np.zeros(n)
    L2 = np.zeros(n)
    L3 = np.zeros(n)
    rsi = np.zeros(n)

    for t in tqdm(range(n), desc="Computing RSI-Laguerre"):
        g = gamma[t]
        if not np.isfinite(g):
            g = gamma[t - 1] if t > 0 else 0.0

        # L0..L3 recursion
        prev_L0 = L0[t - 1] if t > 0 else 0.0
        prev_L1 = L1[t - 1] if t > 0 else 0.0
        prev_L2 = L2[t - 1] if t > 0 else 0.0
        prev_L3 = L3[t - 1] if t > 0 else 0.0

        L0[t] = (1 - g) * Gc[t] + g * prev_L0
        L1[t] = -g * L0[t] + prev_L0 + g * prev_L1
        L2[t] = -g * L1[t] + prev_L1 + g * prev_L2
        L3[t] = -g * L2[t] + prev_L2 + g * prev_L3

        # CU/CD logic
        if L0[t] >= L1[t]:
            CU1 = L0[t] - L1[t]
            CD1 = 0.0
        else:
            CD1 = L1[t] - L0[t]
            CU1 = 0.0

        if L1[t] >= L2[t]:
            CU2 = CU1 + L1[t] - L2[t]
            CD2 = CD1
        else:
            CD2 = CD1 + L2[t] - L1[t]
            CU2 = CU1

        if L2[t] >= L3[t]:
            CU = CU2 + L2[t] - L3[t]
            CD = CD2
        else:
            CU = CU2
            CD = CD2 + L3[t] - L2[t]

        rsi[t] = CU / (CU + CD) if (CU + CD) != 0 else 0.0

    # ---- Add to DataFrame ----
    df["rsilg_gamma"] = gamma
    df["rsilg_rsi"] = rsi
    df["rsilg_os"] = 0.2
    df["rsilg_ob"] = 0.8
    df["rsilg_mid"] = 0.5

    # Helpful binary features (you can let GA decide if they matter)
    df["rsilg_fe_high"] = (df["rsilg_gamma"] >= 0.618).astype(int)
    df["rsilg_fe_low"] = (df["rsilg_gamma"] <= 0.382).astype(int)
    df["rsilg_rsi_hi"] = (df["rsilg_rsi"] >= 0.8).astype(int)
    df["rsilg_rsi_lo"] = (df["rsilg_rsi"] <= 0.2).astype(int)

    return df

def add_tmo(df, length=14, calc_length=5, smooth_length=3):
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()
    n = len(df)

    # True Momentum Oscillator base "data" calculation
    data = np.full(n, np.nan)
    for t in tqdm(range(n), desc="Computing TMO"):
        s = 0.0
        for i in range(length):
            j = t - i
            if j < 0:
                break
            if close[t] > open_[j]:
                s += 1.0
            elif close[t] < open_[j]:
                s -= 1.0
        data[t] = s

    df["tmo_data"] = data

    # EMA layers
    ema1 = df["tmo_data"].ewm(span=calc_length, adjust=False).mean()
    main = ema1.ewm(span=smooth_length, adjust=False).mean()
    signal = main.ewm(span=smooth_length, adjust=False).mean()

    df["tmo_main"] = main
    df["tmo_signal"] = signal

    # Cross signals
    prev_main = df["tmo_main"].shift(1)
    prev_signal = df["tmo_signal"].shift(1)

    df["tmo_buy"] = ((prev_main <= prev_signal) & (df["tmo_main"] > df["tmo_signal"])).astype(int)
    df["tmo_sell"] = ((prev_main >= prev_signal) & (df["tmo_main"] < df["tmo_signal"])).astype(int)

    return df

