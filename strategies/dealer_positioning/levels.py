from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from strategies.dealer_positioning.models import GammaLevels, GammaStructure, OptionContractRow
from strategies.dealer_positioning.topology import EXPIRY_BUCKETS, expiry_bucket_shares


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
    "call_gamma_mean_by_expiry",
    "put_gamma_mean_by_expiry",
    "call_delta",
    "put_delta",
    "call_vega",
    "put_vega",
    "call_iv",
    "put_iv",
    "call_gex",
    "put_gex",
    "call_vex",
    "put_vex",
    "net_gex",
    "abs_net_gex",
    "total_abs_gex",
    "total_vex",
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
                "vega": row.vega,
                "iv": row.iv,
            }
        )
    return pd.DataFrame.from_records(records)


def build_gamma_ladder(rows: Iterable[OptionContractRow] | pd.DataFrame, *, spot: float) -> pd.DataFrame:
    df = rows_to_frame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if df.empty:
        return pd.DataFrame(columns=LADDER_COLUMNS)

    df["option_type"] = df["option_type"].astype(str).str.upper().str[0]
    numeric_cols = ["strike", "open_interest", "volume", "gamma", "delta", "vega", "iv"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["strike"])
    df = df[df["strike"] > 0.0]
    if df.empty:
        return pd.DataFrame(columns=LADDER_COLUMNS)
    calls = df[df["option_type"] == "C"].copy()
    puts = df[df["option_type"] == "P"].copy()
    calls["call_gex"] = calls["open_interest"].fillna(0.0) * calls["gamma"].fillna(0.0) * 100.0 * float(spot)
    puts["put_gex"] = -1.0 * puts["open_interest"].fillna(0.0) * puts["gamma"].fillna(0.0) * 100.0 * float(spot)
    calls["call_vex"] = calls["open_interest"].fillna(0.0) * calls["vega"].fillna(0.0) * 100.0
    puts["put_vex"] = puts["open_interest"].fillna(0.0) * puts["vega"].fillna(0.0) * 100.0

    call_group = calls.groupby("strike", as_index=False).agg(
        call_oi=("open_interest", "sum"),
        call_volume=("volume", "sum"),
        call_gamma=("gamma", "mean"),
        call_delta=("delta", "mean"),
        call_vega=("vega", "mean"),
        call_iv=("iv", "mean"),
        call_gex=("call_gex", "sum"),
        call_vex=("call_vex", "sum"),
    )
    put_group = puts.groupby("strike", as_index=False).agg(
        put_oi=("open_interest", "sum"),
        put_volume=("volume", "sum"),
        put_gamma=("gamma", "mean"),
        put_delta=("delta", "mean"),
        put_vega=("vega", "mean"),
        put_iv=("iv", "mean"),
        put_gex=("put_gex", "sum"),
        put_vex=("put_vex", "sum"),
    )
    ladder = pd.merge(call_group, put_group, on="strike", how="outer").fillna(0.0)
    # `call_gamma`/`put_gamma` are the MEAN per-contract gamma across every
    # expiry present at that strike, so the call and put values at one strike
    # are not comparable to each other and neither is "the gamma at that
    # strike". They are kept only because artifacts on disk carry the name.
    # The GEX columns are unaffected: exposure is computed per contract above
    # and summed, so `call_gex`/`put_gex` remain correct.
    ladder["call_gamma_mean_by_expiry"] = ladder["call_gamma"]
    ladder["put_gamma_mean_by_expiry"] = ladder["put_gamma"]
    ladder["spot"] = float(spot)
    ladder["net_gex"] = ladder["call_gex"] + ladder["put_gex"]
    ladder["abs_net_gex"] = ladder["net_gex"].abs()
    ladder["total_abs_gex"] = ladder["call_gex"].abs() + ladder["put_gex"].abs()
    ladder["total_vex"] = ladder["call_vex"].fillna(0.0) + ladder["put_vex"].fillna(0.0)
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
            vega_wall=None,
            next_vega_wall_above=None,
            next_vega_wall_below=None,
            gamma_flip=None,
            air_gap_above_score=0.0,
            air_gap_below_score=0.0,
            magnet_threshold_abs_net_gex=0.0,
            vega_threshold_total_vex=0.0,
            expirations=[],
            per_dte_levels={},
            per_bucket_levels={},
            term_structure={},
        )
        return ladder, levels

    timestamp = str(ladder["timestamp"].iloc[-1])
    core = _core_levels_from_ladder(ladder, float(spot), float(magnet_quantile))
    frame = rows_to_frame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    per_dte_levels = _per_dte_levels(frame, symbol=symbol, spot=float(spot), magnet_quantile=float(magnet_quantile))
    per_bucket_levels = _per_bucket_levels(frame, spot=float(spot), magnet_quantile=float(magnet_quantile))
    term_structure = _term_structure(frame, spot=float(spot))

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
        vega_wall=core["vega_wall"],
        next_vega_wall_above=core["next_vega_wall_above"],
        next_vega_wall_below=core["next_vega_wall_below"],
        gamma_flip=core["gamma_flip"],
        air_gap_above_score=core["air_gap_above_score"],
        air_gap_below_score=core["air_gap_below_score"],
        magnet_threshold_abs_net_gex=core["magnet_threshold_abs_net_gex"],
        vega_threshold_total_vex=core["vega_threshold_total_vex"],
        expirations=sorted(set(str(x) for x in _expirations_from_rows(rows))),
        per_dte_levels=per_dte_levels,
        per_bucket_levels=per_bucket_levels,
        term_structure=term_structure,
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
    vega_threshold = _positive_quantile(ladder.get("total_vex"), magnet_quantile)
    vega_walls = ladder[ladder["total_vex"] >= vega_threshold].copy() if vega_threshold > 0.0 else ladder.iloc[0:0].copy()
    return {
        "total_gex": float(ladder["net_gex"].sum()),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "nearest_magnet": nearest_magnet,
        "next_magnet_above": next_above,
        "next_magnet_below": next_below,
        "vega_wall": _nearest_strike(vega_walls, spot),
        "next_vega_wall_above": _next_strike(vega_walls, spot, "above"),
        "next_vega_wall_below": _next_strike(vega_walls, spot, "below"),
        "gamma_flip": _gamma_flip(ladder, spot),
        "air_gap_above_score": _air_gap_score(ladder, spot, next_above, threshold, "above"),
        "air_gap_below_score": _air_gap_score(ladder, spot, next_below, threshold, "below"),
        "magnet_threshold_abs_net_gex": threshold,
        "vega_threshold_total_vex": vega_threshold,
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
            "vega_wall": core["vega_wall"],
            "next_vega_wall_above": core["next_vega_wall_above"],
            "next_vega_wall_below": core["next_vega_wall_below"],
            "gamma_flip": core["gamma_flip"],
            "air_gap_above_score": core["air_gap_above_score"],
            "air_gap_below_score": core["air_gap_below_score"],
            "magnet_threshold_abs_net_gex": core["magnet_threshold_abs_net_gex"],
            "vega_threshold_total_vex": core["vega_threshold_total_vex"],
            "expirations": sorted(set(str(x) for x in sliced["expiration"].dropna().unique())),
        }
    return out


def _per_bucket_levels(
    frame: pd.DataFrame,
    *,
    spot: float,
    magnet_quantile: float,
) -> dict[str, dict[str, float | str | list[str] | None]]:
    """Recompute the core levels inside each expiry bucket.

    Unlike ``_per_dte_levels`` (which reports single days and only D0/D1), this
    covers the whole term so a consumer can ask "where is the near-dated
    structure" separately from "where is the monthly structure".
    """
    if frame.empty or "dte" not in frame.columns:
        return {}
    work = frame.copy()
    work["dte"] = pd.to_numeric(work["dte"], errors="coerce")
    work = work.dropna(subset=["dte"])
    if work.empty:
        return {}
    out: dict[str, dict[str, float | str | list[str] | None]] = {}
    for label, low, high in EXPIRY_BUCKETS:
        sliced = work[(work["dte"] >= low) & (work["dte"] <= high)].copy()
        if sliced.empty:
            continue
        ladder = build_gamma_ladder(sliced, spot=spot)
        if ladder.empty:
            continue
        core = _core_levels_from_ladder(ladder, float(spot), float(magnet_quantile))
        out[label] = {
            "label": label,
            "spot": float(spot),
            "estimated_net_gex": core["total_gex"],
            "call_wall": core["call_wall"],
            "put_wall": core["put_wall"],
            "nearest_magnet": core["nearest_magnet"],
            "gamma_flip": core["gamma_flip"],
            "contract_count": int(len(sliced)),
            "expirations": sorted(set(str(x) for x in sliced["expiration"].dropna().unique()))
            if "expiration" in sliced.columns
            else [],
        }
    return out


def _term_structure(frame: pd.DataFrame, *, spot: float) -> dict[str, float | None]:
    """Gamma share by expiry bucket, plus a near-minus-far slope."""
    shares = expiry_bucket_shares(frame, spot=spot)
    if not shares:
        return {}
    short = float(shares.get("d0", 0.0) + shares.get("d1_2", 0.0))
    long_dated = float(shares.get("d8_30", 0.0) + shares.get("d30_plus", 0.0))
    return {
        **{f"gamma_share_{k}": v for k, v in shares.items()},
        "zero_dte_gamma_share": shares.get("d0"),
        "short_gamma_share": short,
        "weekly_gamma_share": shares.get("d3_7"),
        "gamma_term_slope": short - long_dated,
    }


def _positive_quantile(series: pd.Series | None, quantile: float) -> float:
    if series is None:
        return 0.0
    clean = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if clean.empty or float(clean.max()) <= 0.0:
        return 0.0
    threshold = float(clean.quantile(float(quantile)))
    if threshold <= 0.0:
        threshold = float(clean.max())
    return threshold


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


def compute_gamma_structure(
    rows: Iterable[OptionContractRow] | pd.DataFrame,
    *,
    symbol: str,
    spot: float,
    magnet_quantile: float = 0.90,
    age_days: float | None = 0.0,
    avg_dollar_volume_20d: float | None = None,
    market_cap: float | None = None,
    with_stability: bool = True,
) -> "GammaStructure":
    """The full structural view: levels, unsigned topology, sign estimate, confidence.

    ``compute_gamma_levels`` is unchanged and still returns its ``(ladder,
    levels)`` pair -- four consumers depend on that shape. This is the richer
    entry point for anything that needs to know how much to trust what it just
    read.
    """
    from strategies.dealer_positioning.confidence import build_confidence
    from strategies.dealer_positioning.topology import (
        build_estimated_signed_exposure,
        build_gamma_topology,
    )

    ladder, levels = compute_gamma_levels(
        rows, symbol=symbol, spot=spot, magnet_quantile=magnet_quantile
    )
    frame = rows_to_frame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()

    strike_coverage = None
    if not ladder.empty and float(spot) > 0.0:
        strikes = pd.to_numeric(ladder["strike"], errors="coerce").dropna()
        if not strikes.empty:
            # Did the captured window actually bracket spot? A window entirely
            # on one side describes half a structure.
            strike_coverage = 1.0 if (strikes.min() < spot < strikes.max()) else 0.5

    confidence = build_confidence(
        symbol=symbol,
        frame=frame,
        age_days=age_days,
        avg_dollar_volume_20d=avg_dollar_volume_20d,
        market_cap=market_cap,
        strike_coverage=strike_coverage,
    )
    topology = build_gamma_topology(
        ladder, symbol=symbol.upper(), spot=float(spot), timestamp=levels.timestamp, frame=frame
    )
    signed = build_estimated_signed_exposure(
        ladder,
        symbol=symbol.upper(),
        spot=float(spot),
        timestamp=levels.timestamp,
        call_wall=levels.call_wall,
        put_wall=levels.put_wall,
        gamma_flip=levels.gamma_flip,
    )

    stability = None
    if with_stability and not frame.empty:
        from strategies.dealer_positioning.stability import assess_stability

        stability = assess_stability(frame, spot=float(spot), magnet_quantile=magnet_quantile)

    return GammaStructure(
        ladder=ladder,
        levels=levels,
        topology=topology,
        signed=signed,
        confidence=confidence,
        stability=stability,
    )
