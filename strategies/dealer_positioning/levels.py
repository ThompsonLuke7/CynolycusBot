from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from strategies.dealer_positioning.models import GammaLevels, OptionContractRow


LADDER_COLUMNS = [
    "timestamp",
    "symbol",
    "strike",
    "spot",
    "call_oi",
    "put_oi",
    "call_volume",
    "put_volume",
    "call_gamma",
    "put_gamma",
    "call_delta",
    "put_delta",
    "call_iv",
    "put_iv",
    "call_gex",
    "put_gex",
    "net_gex",
    "abs_net_gex",
    "total_abs_gex",
]


def rows_to_frame(rows: Iterable[OptionContractRow]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append(
            {
                "timestamp": row.timestamp,
                "symbol": row.symbol,
                "expiration": row.expiration,
                "dte": row.dte,
                "expiry_bucket": row.expiry_bucket,
                "strike": float(row.strike),
                "option_type": row.option_type,
                "open_interest": float(row.open_interest),
                "volume": float(row.volume),
                "gamma": float(row.gamma),
                "delta": row.delta,
                "iv": row.iv,
            }
        )
    return pd.DataFrame.from_records(records)


def build_gamma_ladder(rows: Iterable[OptionContractRow] | pd.DataFrame, *, spot: float) -> pd.DataFrame:
    df = rows_to_frame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if df.empty:
        return pd.DataFrame(columns=LADDER_COLUMNS)

    df["option_type"] = df["option_type"].astype(str).str.upper().str[0]
    numeric_cols = ["strike", "open_interest", "volume", "gamma", "delta", "iv"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["strike"])
    calls = df[df["option_type"] == "C"].copy()
    puts = df[df["option_type"] == "P"].copy()
    calls["call_gex"] = calls["open_interest"].fillna(0.0) * calls["gamma"].fillna(0.0) * 100.0 * float(spot)
    puts["put_gex"] = -1.0 * puts["open_interest"].fillna(0.0) * puts["gamma"].fillna(0.0) * 100.0 * float(spot)

    call_group = calls.groupby("strike", as_index=False).agg(
        call_oi=("open_interest", "sum"),
        call_volume=("volume", "sum"),
        call_gamma=("gamma", "mean"),
        call_delta=("delta", "mean"),
        call_iv=("iv", "mean"),
        call_gex=("call_gex", "sum"),
    )
    put_group = puts.groupby("strike", as_index=False).agg(
        put_oi=("open_interest", "sum"),
        put_volume=("volume", "sum"),
        put_gamma=("gamma", "mean"),
        put_delta=("delta", "mean"),
        put_iv=("iv", "mean"),
        put_gex=("put_gex", "sum"),
    )
    ladder = pd.merge(call_group, put_group, on="strike", how="outer").fillna(0.0)
    ladder["spot"] = float(spot)
    ladder["net_gex"] = ladder["call_gex"] + ladder["put_gex"]
    ladder["abs_net_gex"] = ladder["net_gex"].abs()
    ladder["total_abs_gex"] = ladder["call_gex"].abs() + ladder["put_gex"].abs()
    ladder["timestamp"] = _latest_timestamp(df)
    ladder["symbol"] = str(df["symbol"].dropna().iloc[0]).upper() if "symbol" in df and not df["symbol"].dropna().empty else ""
    return ladder.reindex(columns=LADDER_COLUMNS).sort_values("strike").reset_index(drop=True)


def compute_gamma_levels(
    rows: Iterable[OptionContractRow] | pd.DataFrame,
    *,
    symbol: str,
    spot: float,
    magnet_quantile: float = 0.90,
) -> tuple[pd.DataFrame, GammaLevels]:
    ladder = build_gamma_ladder(rows, spot=spot)
    if ladder.empty:
        levels = GammaLevels(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol.upper(),
            spot=float(spot),
            total_gex=0.0,
            call_wall=None,
            put_wall=None,
            nearest_magnet=None,
            next_magnet_above=None,
            next_magnet_below=None,
            gamma_flip=None,
            air_gap_above_score=0.0,
            air_gap_below_score=0.0,
            magnet_threshold_abs_net_gex=0.0,
            expirations=[],
            per_dte_levels={},
        )
        return ladder, levels

    timestamp = str(ladder["timestamp"].iloc[-1])
    core = _core_levels_from_ladder(ladder, float(spot), float(magnet_quantile))
    frame = rows_to_frame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    per_dte_levels = _per_dte_levels(frame, symbol=symbol, spot=float(spot), magnet_quantile=float(magnet_quantile))

    levels = GammaLevels(
        timestamp=timestamp,
        symbol=symbol.upper(),
        spot=float(spot),
        total_gex=core["total_gex"],
        call_wall=core["call_wall"],
        put_wall=core["put_wall"],
        nearest_magnet=core["nearest_magnet"],
        next_magnet_above=core["next_magnet_above"],
        next_magnet_below=core["next_magnet_below"],
        gamma_flip=core["gamma_flip"],
        air_gap_above_score=core["air_gap_above_score"],
        air_gap_below_score=core["air_gap_below_score"],
        magnet_threshold_abs_net_gex=core["magnet_threshold_abs_net_gex"],
        expirations=sorted(set(str(x) for x in _expirations_from_rows(rows))),
        per_dte_levels=per_dte_levels,
    )
    return ladder, levels


def _core_levels_from_ladder(ladder: pd.DataFrame, spot: float, magnet_quantile: float) -> dict[str, float | None]:
    threshold = float(ladder["abs_net_gex"].quantile(float(magnet_quantile)))
    if threshold <= 0.0:
        threshold = float(ladder["abs_net_gex"].max())
    magnets = ladder[ladder["abs_net_gex"] >= threshold].copy()
    if magnets.empty and not ladder.empty:
        magnets = ladder.nlargest(1, "abs_net_gex").copy()

    call_wall = _strike_of_max(ladder[ladder["strike"] > spot], "call_gex")
    put_wall = _strike_of_min(ladder[ladder["strike"] < spot], "put_gex")
    nearest_magnet = _nearest_strike(magnets, spot)
    next_above = _next_strike(magnets, spot, "above")
    next_below = _next_strike(magnets, spot, "below")
    return {
        "total_gex": float(ladder["net_gex"].sum()),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "nearest_magnet": nearest_magnet,
        "next_magnet_above": next_above,
        "next_magnet_below": next_below,
        "gamma_flip": _gamma_flip(ladder, spot),
        "air_gap_above_score": _air_gap_score(ladder, spot, next_above, threshold, "above"),
        "air_gap_below_score": _air_gap_score(ladder, spot, next_below, threshold, "below"),
        "magnet_threshold_abs_net_gex": threshold,
    }


def _per_dte_levels(
    frame: pd.DataFrame,
    *,
    symbol: str,
    spot: float,
    magnet_quantile: float,
) -> dict[str, dict[str, float | str | list[str] | None]]:
    bucket_col = "expiry_bucket" if "expiry_bucket" in frame.columns and frame["expiry_bucket"].notna().any() else "dte"
    if frame.empty or bucket_col not in frame.columns:
        return {}
    out: dict[str, dict[str, float | str | list[str] | None]] = {}
    frame[bucket_col] = pd.to_numeric(frame[bucket_col], errors="coerce")
    for dte_value in sorted(x for x in frame[bucket_col].dropna().unique() if x in {0, 1}):
        dte_int = int(dte_value)
        sliced = frame[frame[bucket_col] == dte_value].copy()
        ladder = build_gamma_ladder(sliced, spot=spot)
        if ladder.empty:
            continue
        core = _core_levels_from_ladder(ladder, float(spot), float(magnet_quantile))
        out[f"D{dte_int}"] = {
            "label": f"D{dte_int}",
            "spot": float(spot),
            "total_gex": core["total_gex"],
            "call_wall": core["call_wall"],
            "put_wall": core["put_wall"],
            "nearest_magnet": core["nearest_magnet"],
            "next_magnet_above": core["next_magnet_above"],
            "next_magnet_below": core["next_magnet_below"],
            "gamma_flip": core["gamma_flip"],
            "air_gap_above_score": core["air_gap_above_score"],
            "air_gap_below_score": core["air_gap_below_score"],
            "magnet_threshold_abs_net_gex": core["magnet_threshold_abs_net_gex"],
            "expirations": sorted(set(str(x) for x in sliced["expiration"].dropna().unique())),
        }
    return out


def _latest_timestamp(df: pd.DataFrame) -> str:
    if "timestamp" not in df or df["timestamp"].dropna().empty:
        return datetime.now(timezone.utc).isoformat()
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
    if ts.empty:
        return datetime.now(timezone.utc).isoformat()
    return ts.max().isoformat()


def _expirations_from_rows(rows: Iterable[OptionContractRow] | pd.DataFrame) -> list[str]:
    if isinstance(rows, pd.DataFrame):
        if "expiration" not in rows:
            return []
        return [str(x) for x in rows["expiration"].dropna().unique()]
    return [row.expiration for row in rows if row.expiration]


def _strike_of_max(df: pd.DataFrame, col: str) -> float | None:
    if df.empty:
        return None
    idx = df[col].idxmax()
    val = float(df.loc[idx, col])
    if val <= 0.0:
        return None
    return float(df.loc[idx, "strike"])


def _strike_of_min(df: pd.DataFrame, col: str) -> float | None:
    if df.empty:
        return None
    idx = df[col].idxmin()
    val = float(df.loc[idx, col])
    if val >= 0.0:
        return None
    return float(df.loc[idx, "strike"])


def _nearest_strike(df: pd.DataFrame, spot: float) -> float | None:
    if df.empty:
        return None
    idx = (df["strike"] - float(spot)).abs().idxmin()
    return float(df.loc[idx, "strike"])


def _next_strike(df: pd.DataFrame, spot: float, side: str) -> float | None:
    if df.empty:
        return None
    if side == "above":
        candidates = df[df["strike"] > float(spot)]
        if candidates.empty:
            return None
        return float(candidates["strike"].min())
    candidates = df[df["strike"] < float(spot)]
    if candidates.empty:
        return None
    return float(candidates["strike"].max())


def _gamma_flip(ladder: pd.DataFrame, spot: float) -> float | None:
    ordered = ladder.sort_values("strike").reset_index(drop=True)
    candidates: list[tuple[float, float]] = []
    for i in range(1, len(ordered)):
        prev = float(ordered.loc[i - 1, "net_gex"])
        cur = float(ordered.loc[i, "net_gex"])
        if prev == 0.0:
            strike = float(ordered.loc[i - 1, "strike"])
        elif cur == 0.0:
            strike = float(ordered.loc[i, "strike"])
        elif (prev < 0 < cur) or (prev > 0 > cur):
            s0 = float(ordered.loc[i - 1, "strike"])
            s1 = float(ordered.loc[i, "strike"])
            weight = abs(prev) / (abs(prev) + abs(cur))
            strike = s0 + (s1 - s0) * weight
        else:
            continue
        candidates.append((abs(strike - float(spot)), strike))
    if candidates:
        return float(min(candidates)[1])

    ordered["cum_net_gex"] = ordered["net_gex"].cumsum()
    for i in range(1, len(ordered)):
        prev = float(ordered.loc[i - 1, "cum_net_gex"])
        cur = float(ordered.loc[i, "cum_net_gex"])
        if (prev < 0 < cur) or (prev > 0 > cur):
            return float(ordered.loc[i, "strike"])
    return None


def _air_gap_score(ladder: pd.DataFrame, spot: float, magnet: float | None, threshold: float, side: str) -> float:
    if magnet is None or threshold <= 0.0:
        return 0.0
    lo, hi = sorted((float(spot), float(magnet)))
    between = ladder[(ladder["strike"] > lo) & (ladder["strike"] < hi)]
    distance = abs(float(magnet) - float(spot))
    if between.empty:
        density_ratio = 0.0
    else:
        density_ratio = float(between["abs_net_gex"].mean()) / float(threshold)
    low_density = max(0.0, 1.0 - min(1.0, density_ratio))
    one_atrish_point = max(0.25, float(spot) * 0.001)
    score = (distance / one_atrish_point) * low_density
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return float(score)
