"""Capture daily dealer-positioning snapshots for research.

This is a forward-collection module, not a historical backfill.  It fetches the
current option chain, computes the same floor/ceiling/magnet/VEX levels used by
Amethyst, and persists both summary rows and strike rows partitioned by snapshot
date.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from UI.dealer_positioning_dashboard import DealerDashboardApp

_ET = ZoneInfo("America/New_York")
UNIVERSE = REPO / "Data" / "shared" / "universe" / "shared_universe.csv"
OUT_ROOT = REPO / "Data" / "dealer_positioning" / "historical_snapshots"
SCOPES = ("daily_week", "through_month", "two_months")
DEFAULT_MIN_TOTAL_OI = 5_000.0
DEFAULT_MIN_OPTION_VOLUME = 2_000.0
DEFAULT_MIN_ADV = 50_000_000.0
DEFAULT_MIN_MARKET_CAP = 2_000_000_000.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureResult:
    symbols: int
    candidate_symbols: int
    skipped_symbols: int
    scopes: int
    summary_rows: int
    strike_rows: int
    errors: int
    output_dir: Path


def load_symbols(
    *,
    universe_path: Path = UNIVERSE,
    tickers: Iterable[str] | None = None,
    eligible_only: bool = True,
    liquidity_prefilter: bool = True,
    min_adv: float = DEFAULT_MIN_ADV,
    min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
    limit: int | None = None,
) -> list[str]:
    if tickers:
        symbols = [str(t).strip().upper() for t in tickers if str(t).strip()]
    else:
        frame = _load_universe_frame(
            universe_path=universe_path,
            eligible_only=eligible_only,
            liquidity_prefilter=liquidity_prefilter,
            min_adv=min_adv,
            min_market_cap=min_market_cap,
        )
        symbols = frame["ticker"].dropna().astype(str).str.upper().tolist()
    symbols = sorted(dict.fromkeys(symbols))
    if limit is not None:
        symbols = symbols[: max(0, int(limit))]
    return symbols


def _load_universe_frame(
    *,
    universe_path: Path,
    eligible_only: bool,
    liquidity_prefilter: bool,
    min_adv: float,
    min_market_cap: float,
) -> pd.DataFrame:
    frame = pd.read_csv(universe_path)
    if eligible_only and "is_eligible" in frame.columns:
        mask = frame["is_eligible"].astype(str).str.lower().isin({"true", "1", "1.0", "yes"})
        frame = frame[mask].copy()
    if liquidity_prefilter:
        frame = frame[_equity_liquidity_mask(frame, min_adv=min_adv, min_market_cap=min_market_cap)].copy()
    return frame


def _equity_liquidity_mask(frame: pd.DataFrame, *, min_adv: float, min_market_cap: float) -> pd.Series:
    adv = pd.to_numeric(frame.get("avg_dollar_volume_20d", 0.0), errors="coerce").fillna(0.0)
    market_cap = pd.to_numeric(frame.get("market_cap", 0.0), errors="coerce").fillna(0.0)
    cap_bucket = frame.get("market_cap_bucket", "").astype(str).str.lower()
    cap_tier = frame.get("cap_tier", "").astype(str).str.lower()
    liquid_cap_proxy = cap_bucket.isin({"large", "mega"}) | cap_tier.isin({"large", "mega"})
    stock_only = frame.get("stock_only", False)
    if not isinstance(stock_only, pd.Series):
        stock_only = pd.Series(False, index=frame.index)
    stock_only_mask = stock_only.astype(str).str.lower().isin({"true", "1", "1.0", "yes"})
    return ((adv >= float(min_adv)) | (market_cap >= float(min_market_cap)) | liquid_cap_proxy) & ~stock_only_mask


def count_candidate_symbols(
    *,
    universe_path: Path = UNIVERSE,
    eligible_only: bool = True,
    liquidity_prefilter: bool = True,
    min_adv: float = DEFAULT_MIN_ADV,
    min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
) -> dict[str, int]:
    all_frame = pd.read_csv(universe_path)
    eligible_frame = all_frame
    if eligible_only and "is_eligible" in eligible_frame.columns:
        eligible_frame = eligible_frame[
            eligible_frame["is_eligible"].astype(str).str.lower().isin({"true", "1", "1.0", "yes"})
        ].copy()
    selected = _load_universe_frame(
        universe_path=universe_path,
        eligible_only=eligible_only,
        liquidity_prefilter=liquidity_prefilter,
        min_adv=min_adv,
        min_market_cap=min_market_cap,
    )
    return {
        "universe_rows": int(len(all_frame)),
        "eligible_rows": int(len(eligible_frame)),
        "candidate_rows": int(len(selected)),
        "adv_pass": int(
            (pd.to_numeric(eligible_frame.get("avg_dollar_volume_20d", 0.0), errors="coerce").fillna(0.0) >= float(min_adv)).sum()
        ),
        "market_cap_pass": int(
            (pd.to_numeric(eligible_frame.get("market_cap", 0.0), errors="coerce").fillna(0.0) >= float(min_market_cap)).sum()
        ),
        "large_or_mega_proxy_pass": int(
            (
                eligible_frame.get("market_cap_bucket", "").astype(str).str.lower().isin({"large", "mega"})
                | eligible_frame.get("cap_tier", "").astype(str).str.lower().isin({"large", "mega"})
            ).sum()
        ),
    }


def capture_snapshots(
    *,
    symbols: list[str],
    scopes: tuple[str, ...] = SCOPES,
    output_root: Path = OUT_ROOT,
    strike_window_pct: float = 0.50,
    sleep_seconds: float = 0.25,
    min_total_oi: float = DEFAULT_MIN_TOTAL_OI,
    min_option_volume: float = DEFAULT_MIN_OPTION_VOLUME,
    min_adv: float = DEFAULT_MIN_ADV,
    min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
    universe_path: Path = UNIVERSE,
    snapshot_date: str | None = None,
    ref_date: str | None = None,
    workers: int = 1,
    publish_states: bool = True,
) -> CaptureResult:
    now = datetime.now(tz=_ET)
    snapshot_date_value = snapshot_date or now.date().isoformat()
    ref_date_value = _parse_capture_date(ref_date or snapshot_date_value)
    output_dir = output_root / snapshot_date_value.replace("-", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    strike_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    skipped_symbols: set[str] = set()
    universe_meta = _universe_meta(universe_path)
    prior_summary = _load_prior_summary(output_root=output_root, before_date=snapshot_date_value)

    total = max(1, len(symbols) * len(scopes))
    done = 0

    def _capture_one(symbol: str, scope: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, str | None]:
        app = DealerDashboardApp()
        grid = app.chain_grid(
            symbol=symbol,
            strike_window_pct=float(strike_window_pct),
            expiration_scope=scope,
            ref_date=ref_date_value,
        )
        levels = grid.get("levels") or {}
        row_meta = universe_meta.get(symbol.upper(), {})
        liquidity = _option_liquidity(grid.get("rows") or [], row_meta=row_meta)
        if not _passes_liquidity(
            liquidity,
            min_total_oi=min_total_oi,
            min_option_volume=min_option_volume,
            min_adv=min_adv,
            min_market_cap=min_market_cap,
        ):
            return [], [], None, symbol.upper()
        summary = _summary_row(
            grid=grid,
            levels=levels,
            scope=scope,
            captured_at=now,
            snapshot_date=snapshot_date_value,
            row_meta=row_meta,
            liquidity=liquidity,
        )
        summary.update(_matrix_features(grid.get("rows") or [], levels=levels))
        strikes = [
            _strike_row(grid=grid, row=row, scope=scope, captured_at=now, snapshot_date=snapshot_date_value)
            for row in grid.get("rows") or []
        ]
        return [summary], strikes, None, None

    def _record_progress() -> None:
        if done % 25 == 0 or done == total:
            logger.info(
                "dealer snapshot capture %d/%d done: summaries=%d strikes=%d errors=%d",
                done,
                total,
                len(summary_rows),
                len(strike_rows),
                len(error_rows),
            )

    tasks = [(symbol, scope) for symbol in symbols for scope in scopes]
    if int(workers or 1) <= 1:
        for symbol, scope in tasks:
            done += 1
            try:
                summaries, strikes, _err, skipped = _capture_one(symbol, scope)
                summary_rows.extend(summaries)
                strike_rows.extend(strikes)
                if skipped:
                    skipped_symbols.add(skipped)
            except Exception as exc:  # noqa: BLE001 - each symbol/scope should fail independently
                error_rows.append(
                    {
                        "captured_at": now.isoformat(),
                        "snapshot_date": snapshot_date_value,
                        "symbol": symbol,
                        "scope": scope,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            _record_progress()
            if sleep_seconds > 0:
                time.sleep(float(sleep_seconds))
    else:
        max_workers = max(1, int(workers))
        logger.info("dealer snapshot capture: parallel workers=%d tasks=%d", max_workers, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_capture_one, symbol, scope): (symbol, scope) for symbol, scope in tasks}
            for fut in as_completed(futs):
                symbol, scope = futs[fut]
                done += 1
                try:
                    summaries, strikes, _err, skipped = fut.result()
                    summary_rows.extend(summaries)
                    strike_rows.extend(strikes)
                    if skipped:
                        skipped_symbols.add(skipped)
                except Exception as exc:  # noqa: BLE001
                    error_rows.append(
                        {
                            "captured_at": now.isoformat(),
                            "snapshot_date": snapshot_date_value,
                            "symbol": symbol,
                            "scope": scope,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                _record_progress()
                if sleep_seconds > 0:
                    time.sleep(float(sleep_seconds))

    summary_rows = _with_dynamics_and_ranks(summary_rows, prior_summary)
    _write_daily_frame(output_dir / "dealer_level_summary.parquet", summary_rows, ["symbol", "scope", "snapshot_date"])
    _write_daily_frame(output_dir / "dealer_strike_ladder.parquet", strike_rows, ["symbol", "scope", "strike", "snapshot_date"])
    if error_rows:
        _write_jsonl(output_dir / "errors.jsonl", error_rows)

    if publish_states and summary_rows:
        _publish_dealer_states(output_dir / "dealer_level_summary.parquet")

    return CaptureResult(
        symbols=len(symbols),
        candidate_symbols=len(symbols),
        skipped_symbols=len(skipped_symbols),
        scopes=len(scopes),
        summary_rows=len(summary_rows),
        strike_rows=len(strike_rows),
        errors=len(error_rows),
        output_dir=output_dir,
    )


def _publish_dealer_states(summary_path: Path) -> None:
    """Hand the captured summary to the governed path.

    Without this call the policy's dealer modifier stays dark and every governed
    decision records CONTEXT_DEALER_UNAVAILABLE. Publication never raises -- an
    unreachable state store leaves the decision exactly where it already was --
    so a capture is never failed by a downstream store being down.
    """
    try:
        import pandas as _pd

        from strategies.dealer_positioning.state_publication import publish_dealer_states

        frame = _pd.read_parquet(summary_path)
        captured_at = _pd.to_datetime(frame["captured_at"]).max().to_pydatetime()
        result = publish_dealer_states(
            frame, captured_at=captured_at, snapshot_path=summary_path
        )
        logger.info(
            "dealer states: %s (published=%s)", result.get("status"), result.get("published")
        )
    except Exception as exc:  # noqa: BLE001 - publication is downstream of a completed capture
        logger.warning("dealer-state publication skipped: %s", type(exc).__name__)


def _summary_row(
    *,
    grid: dict[str, Any],
    levels: dict[str, Any],
    scope: str,
    captured_at: datetime,
    snapshot_date: str | None = None,
    row_meta: dict[str, Any],
    liquidity: dict[str, float],
) -> dict[str, Any]:
    spot = _num(grid.get("spot") or levels.get("spot"))
    gamma_flip = _num(levels.get("gamma_flip"))
    call_wall = _num(levels.get("call_wall"))
    put_wall = _num(levels.get("put_wall"))
    magnet = _num(levels.get("nearest_magnet"))
    vega_wall = _num(levels.get("vega_wall"))
    vega_above = _num(levels.get("next_vega_wall_above"))
    vega_below = _num(levels.get("next_vega_wall_below"))
    ceiling = call_wall
    floor = put_wall
    return {
        "captured_at": captured_at.isoformat(),
        "snapshot_date": snapshot_date or captured_at.date().isoformat(),
        "symbol": grid.get("symbol"),
        "scope": scope,
        "scope_label": grid.get("expiration_scope_label"),
        "ref_date_et": grid.get("ref_date_et"),
        "chain_query_date_et": grid.get("chain_query_date_et"),
        "fetched_at": grid.get("fetched_at"),
        "spot": spot,
        "spot_price": spot,
        "expirations": json.dumps(grid.get("expirations") or []),
        "available_expirations": json.dumps(grid.get("available_expirations") or []),
        "strike_window_pct": grid.get("strike_window_pct"),
        "row_count": len(grid.get("rows") or []),
        "total_gex": levels.get("total_gex"),
        "put_wall": put_wall,
        "call_wall": call_wall,
        "nearest_magnet": magnet,
        "magnet": magnet,
        "next_magnet_above": levels.get("next_magnet_above"),
        "next_magnet_below": levels.get("next_magnet_below"),
        "vega_wall": vega_wall,
        "vega_above": vega_above,
        "vega_below": vega_below,
        "next_vega_wall_above": vega_above,
        "next_vega_wall_below": vega_below,
        "gamma_flip": gamma_flip,
        "ceiling": ceiling,
        "floor": floor,
        "air_gap_above_score": levels.get("air_gap_above_score"),
        "air_gap_below_score": levels.get("air_gap_below_score"),
        "magnet_threshold_abs_net_gex": levels.get("magnet_threshold_abs_net_gex"),
        "vega_threshold_total_vex": levels.get("vega_threshold_total_vex"),
        "avg_dollar_volume_20d": _num(row_meta.get("avg_dollar_volume_20d")),
        "market_cap": _num(row_meta.get("market_cap")),
        "market_cap_bucket": row_meta.get("market_cap_bucket"),
        "cap_tier": row_meta.get("cap_tier"),
        "total_option_oi": liquidity["total_option_oi"],
        "total_option_volume": liquidity["total_option_volume"],
        "liquidity_pass_reason": liquidity["liquidity_pass_reason"],
        "pct_to_gamma_flip": _pct_to(gamma_flip, spot),
        "pct_to_call_wall": _pct_to(call_wall, spot),
        "pct_to_put_wall": _pct_to(put_wall, spot),
        "pct_to_magnet": _pct_to(magnet, spot),
        "pct_to_vega_wall": _pct_to(vega_wall, spot),
        "pct_to_vega_above": _pct_to(vega_above, spot),
        "pct_to_vega_below": _pct_to(vega_below, spot),
        "above_gamma_flip": _above(spot, gamma_flip),
        "above_magnet": _above(spot, magnet),
        "above_call_wall": _above(spot, call_wall),
        "below_vega_wall": _below(spot, vega_wall),
        "inside_call_wall_zone": _inside(spot, put_wall, call_wall),
        "distance_callwall_putwall": _pct_distance(call_wall, put_wall, spot),
        "distance_gammaflip_callwall": _pct_distance(gamma_flip, call_wall, spot),
        "distance_gammaflip_magnet": _pct_distance(gamma_flip, magnet, spot),
        "distance_magnet_putwall": _pct_distance(magnet, put_wall, spot),
        "distance_magnet_callwall": _pct_distance(magnet, call_wall, spot),
        "distance_floor_ceiling": _pct_distance(floor, ceiling, spot),
    }


def _strike_row(
    *,
    grid: dict[str, Any],
    row: dict[str, Any],
    scope: str,
    captured_at: datetime,
    snapshot_date: str | None = None,
) -> dict[str, Any]:
    out = {
        "captured_at": captured_at.isoformat(),
        "snapshot_date": snapshot_date or captured_at.date().isoformat(),
        "symbol": grid.get("symbol"),
        "scope": scope,
        "spot": grid.get("spot"),
        "expirations": json.dumps(grid.get("expirations") or []),
    }
    for key, value in row.items():
        out[key] = json.dumps(value) if isinstance(value, (list, dict, tuple)) else value
    return out


def _universe_meta(universe_path: Path) -> dict[str, dict[str, Any]]:
    if not universe_path.exists():
        return {}
    frame = pd.read_csv(universe_path)
    if "ticker" not in frame.columns:
        return {}
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    return {str(row["ticker"]): dict(row) for row in frame.to_dict(orient="records")}


def _parse_capture_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _option_liquidity(rows: list[dict[str, Any]], *, row_meta: dict[str, Any]) -> dict[str, float]:
    total_oi = 0.0
    total_volume = 0.0
    for row in rows:
        total_oi += _nz(row.get("call_oi")) + _nz(row.get("put_oi"))
        total_volume += _nz(row.get("call_volume")) + _nz(row.get("put_volume"))
    adv = _nz(row_meta.get("avg_dollar_volume_20d"))
    market_cap = _nz(row_meta.get("market_cap"))
    cap_proxy = str(row_meta.get("market_cap_bucket") or row_meta.get("cap_tier") or "").lower()
    reasons: list[str] = []
    if total_oi >= DEFAULT_MIN_TOTAL_OI:
        reasons.append("total_oi")
    if total_volume >= DEFAULT_MIN_OPTION_VOLUME:
        reasons.append("option_volume")
    if adv >= DEFAULT_MIN_ADV:
        reasons.append("adv")
    if market_cap >= DEFAULT_MIN_MARKET_CAP or cap_proxy in {"large", "mega"}:
        reasons.append("market_cap_proxy")
    return {
        "total_option_oi": float(total_oi),
        "total_option_volume": float(total_volume),
        "avg_dollar_volume_20d": float(adv),
        "market_cap": float(market_cap),
        "liquidity_pass_reason": ",".join(reasons),
    }


def _passes_liquidity(
    liquidity: dict[str, float],
    *,
    min_total_oi: float,
    min_option_volume: float,
    min_adv: float,
    min_market_cap: float,
) -> bool:
    return (
        float(liquidity["total_option_oi"]) >= float(min_total_oi)
        or float(liquidity["total_option_volume"]) >= float(min_option_volume)
        or float(liquidity["avg_dollar_volume_20d"]) >= float(min_adv)
        or float(liquidity["market_cap"]) >= float(min_market_cap)
        or "market_cap_proxy" in str(liquidity["liquidity_pass_reason"])
    )


def _matrix_features(rows: list[dict[str, Any]], *, levels: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    for col in [
        "strike",
        "spot",
        "net_gex",
        "abs_net_gex",
        "total_abs_gex",
        "call_gex",
        "put_gex",
        "call_oi",
        "put_oi",
        "call_volume",
        "put_volume",
        "total_vex",
    ]:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    spot = _num(levels.get("spot")) or _num(frame["spot"].iloc[0])
    total_abs_gex = float(frame["total_abs_gex"].abs().sum())
    net_gex = float(frame["net_gex"].sum())
    positive = frame[frame["net_gex"] > 0]
    negative = frame[frame["net_gex"] < 0]
    above = frame[frame["strike"] > float(spot or 0.0)]
    below = frame[frame["strike"] < float(spot or 0.0)]
    weights = frame["total_abs_gex"].abs()
    avg_spacing = _average_strike_spacing(frame["strike"])
    weighted_strike = _weighted_average(frame["strike"], weights)
    call_weighted = _weighted_average(frame["strike"], frame["call_gex"].abs())
    put_weighted = _weighted_average(frame["strike"], frame["put_gex"].abs())
    median_gex_strike = _weighted_median(frame["strike"], weights)
    concentration = _concentration_index(weights)
    entropy = _entropy(weights)
    skew, kurtosis = _weighted_moments(frame["strike"], weights)
    vacuum = _vacuum_score(frame, spot=spot)
    compression = _compression_score(frame, spot=spot)
    magnet = _num(levels.get("nearest_magnet"))
    magnet_strength, magnet_relative, competing = _magnet_features(frame, magnet=magnet)
    gamma_density_5 = _gamma_density(frame, spot=spot, pct=0.05)
    gamma_density_10 = _gamma_density(frame, spot=spot, pct=0.10)
    call_wall_strength = _level_strength(frame, levels.get("call_wall"), "call_gex")
    put_wall_strength = abs(_level_strength(frame, levels.get("put_wall"), "put_gex"))
    total_call_gex = float(frame["call_gex"].sum())
    total_put_gex = float(frame["put_gex"].sum())
    dealer_imbalance = net_gex / total_abs_gex if total_abs_gex else None
    nearest_type, nearest_distance = _nearest_level(levels, spot=spot)
    return {
        "largest_positive_gex": _max_or_none(positive["net_gex"]),
        "largest_negative_gex": _min_or_none(negative["net_gex"]),
        "largest_call_gex": _max_or_none(frame["call_gex"]),
        "largest_put_gex": _min_or_none(frame["put_gex"]),
        "total_positive_gex": float(positive["net_gex"].sum()),
        "total_negative_gex": float(negative["net_gex"].sum()),
        "net_gex": net_gex,
        "total_abs_gex": total_abs_gex,
        "weighted_average_strike": weighted_strike,
        "weighted_average_call": call_weighted,
        "weighted_average_put": put_weighted,
        "median_gex_strike": median_gex_strike,
        "gex_entropy": entropy,
        "gex_concentration_index": concentration,
        "strike_skew": skew,
        "strike_kurtosis": kurtosis,
        "positive_gex_above": float(above.loc[above["net_gex"] > 0, "net_gex"].sum()),
        "positive_gex_below": float(below.loc[below["net_gex"] > 0, "net_gex"].sum()),
        "negative_gex_above": float(above.loc[above["net_gex"] < 0, "net_gex"].sum()),
        "negative_gex_below": float(below.loc[below["net_gex"] < 0, "net_gex"].sum()),
        "call_gex_above": float(above["call_gex"].sum()),
        "put_gex_below": float(below["put_gex"].sum()),
        "ratio_above_vs_below": _safe_div(float(above["total_abs_gex"].sum()), float(below["total_abs_gex"].sum())),
        "vacuum_above": _side_vacuum(frame, spot=spot, side="above"),
        "vacuum_below": _side_vacuum(frame, spot=spot, side="below"),
        "nearest_level_type": nearest_type,
        "nearest_level_distance": nearest_distance,
        "vacuum_score": vacuum,
        "compression_score": compression,
        "pinning_score": _safe_div(abs(float(spot or 0.0) - float(weighted_strike or spot or 0.0)), avg_spacing),
        "dealer_bias": float(above["net_gex"].clip(lower=0).sum() - below["net_gex"].clip(lower=0).sum()),
        "magnet_strength": magnet_strength,
        "magnet_distance": _pct_to(magnet, spot),
        "magnet_relative_strength": magnet_relative,
        "nearest_competing_magnet": competing,
        "gamma_density_5pct": gamma_density_5,
        "gamma_density_10pct": gamma_density_10,
        "gamma_density": gamma_density_5,
        "wall_dominance": _safe_div(call_wall_strength, put_wall_strength),
        "dealer_imbalance": dealer_imbalance,
        "call_gex_total": total_call_gex,
        "put_gex_total": total_put_gex,
        "avg_strike_spacing": avg_spacing,
    }


def _with_dynamics_and_ranks(rows: list[dict[str, Any]], prior_summary: pd.DataFrame) -> list[dict[str, Any]]:
    if not rows:
        return rows
    frame = pd.DataFrame(rows)
    for src, out in [
        ("gamma_flip", "gamma_flip_change_1d"),
        ("call_wall", "call_wall_change"),
        ("put_wall", "put_wall_change"),
        ("magnet", "magnet_change"),
        ("vega_wall", "vega_wall_change"),
        ("ceiling", "ceiling_change"),
        ("floor", "floor_change"),
    ]:
        frame[out] = None
        if not prior_summary.empty and src in prior_summary.columns:
            latest = _latest_prior_by_symbol_scope(prior_summary, src)
            frame[out] = frame.apply(
                lambda r: _diff(r.get(src), latest.get((r.get("symbol"), r.get("scope")))),
                axis=1,
            )
    for src, out in [
        ("gamma_flip", "gamma_flip_velocity_3d"),
        ("magnet", "magnet_velocity_3d"),
        ("call_wall", "callwall_velocity_3d"),
    ]:
        frame[out] = None
        if not prior_summary.empty and src in prior_summary.columns:
            third = _third_prior_by_symbol_scope(prior_summary, src)
            frame[out] = frame.apply(
                lambda r: _safe_div(_diff(r.get(src), third.get((r.get("symbol"), r.get("scope")))), 3.0),
                axis=1,
            )
    for col, rank_col in [
        ("dealer_imbalance", "dealer_imbalance_rank"),
        ("vacuum_score", "vacuum_score_rank"),
        ("pct_to_gamma_flip", "distance_to_gamma_flip_percentile"),
        ("gamma_density", "gamma_density_percentile"),
        ("gex_concentration_index", "gex_concentration_percentile"),
        ("pct_to_magnet", "distance_to_magnet_percentile"),
    ]:
        if col in frame.columns:
            frame[rank_col] = frame.groupby("scope")[col].rank(pct=True, na_option="keep")
    return frame.to_dict(orient="records")


def _load_prior_summary(*, output_root: Path, before_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not output_root.exists():
        return pd.DataFrame()
    for path in sorted(output_root.glob("*/dealer_level_summary.parquet")):
        if path.parent.name >= before_date.replace("-", ""):
            continue
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    # Drop empty/all-NA frames before concat: pandas deprecated including them
    # in dtype resolution, and an early trading day can have thin/empty prior
    # snapshots for some scopes.
    non_empty = [f for f in frames if not f.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def _latest_prior_by_symbol_scope(frame: pd.DataFrame, col: str) -> dict[tuple[str, str], float | None]:
    if frame.empty or col not in frame:
        return {}
    work = frame.copy()
    work["captured_at"] = pd.to_datetime(work.get("captured_at"), errors="coerce", utc=True)
    work = work.sort_values("captured_at")
    out: dict[tuple[str, str], float | None] = {}
    for row in work.to_dict(orient="records"):
        out[(row.get("symbol"), row.get("scope"))] = _num(row.get(col))
    return out


def _third_prior_by_symbol_scope(frame: pd.DataFrame, col: str) -> dict[tuple[str, str], float | None]:
    if frame.empty or col not in frame:
        return {}
    work = frame.copy()
    work["captured_at"] = pd.to_datetime(work.get("captured_at"), errors="coerce", utc=True)
    work = work.sort_values("captured_at")
    out: dict[tuple[str, str], float | None] = {}
    for key, group in work.groupby(["symbol", "scope"], dropna=False):
        vals = group[col].dropna().tolist()
        out[key] = _num(vals[-3] if len(vals) >= 3 else (vals[0] if vals else None))
    return out


def _num(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(val) or math.isinf(val) else val


def _nz(value: Any) -> float:
    return float(_num(value) or 0.0)


def _pct_to(level: Any, spot: Any) -> float | None:
    level_num = _num(level)
    spot_num = _num(spot)
    if level_num is None or spot_num in {None, 0.0}:
        return None
    return (level_num - float(spot_num)) / float(spot_num)


def _pct_distance(a: Any, b: Any, spot: Any) -> float | None:
    a_num = _num(a)
    b_num = _num(b)
    spot_num = _num(spot)
    if a_num is None or b_num is None or spot_num in {None, 0.0}:
        return None
    return abs(a_num - b_num) / float(spot_num)


def _above(spot: Any, level: Any) -> bool | None:
    spot_num = _num(spot)
    level_num = _num(level)
    return None if spot_num is None or level_num is None else bool(float(spot_num) > float(level_num))


def _below(spot: Any, level: Any) -> bool | None:
    spot_num = _num(spot)
    level_num = _num(level)
    return None if spot_num is None or level_num is None else bool(float(spot_num) < float(level_num))


def _inside(spot: Any, lower: Any, upper: Any) -> bool | None:
    spot_num = _num(spot)
    lower_num = _num(lower)
    upper_num = _num(upper)
    if spot_num is None or lower_num is None or upper_num is None:
        return None
    lo, hi = sorted((float(lower_num), float(upper_num)))
    return bool(lo <= float(spot_num) <= hi)


def _diff(a: Any, b: Any) -> float | None:
    a_num = _num(a)
    b_num = _num(b)
    return None if a_num is None or b_num is None else float(a_num - b_num)


def _safe_div(a: Any, b: Any) -> float | None:
    a_num = _num(a)
    b_num = _num(b)
    if a_num is None or b_num in {None, 0.0}:
        return None
    return float(a_num) / float(b_num)


def _max_or_none(series: pd.Series) -> float | None:
    return None if series.empty else float(series.max())


def _min_or_none(series: pd.Series) -> float | None:
    return None if series.empty else float(series.min())


def _weighted_average(values: pd.Series, weights: pd.Series) -> float | None:
    clean = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "w": pd.to_numeric(weights, errors="coerce")}).dropna()
    clean = clean[clean["w"] > 0]
    total = float(clean["w"].sum())
    if clean.empty or total <= 0.0:
        return None
    return float((clean["v"] * clean["w"]).sum() / total)


def _weighted_median(values: pd.Series, weights: pd.Series) -> float | None:
    clean = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "w": pd.to_numeric(weights, errors="coerce")}).dropna()
    clean = clean[clean["w"] > 0].sort_values("v")
    total = float(clean["w"].sum())
    if clean.empty or total <= 0.0:
        return None
    cume = clean["w"].cumsum()
    return float(clean.loc[cume >= total / 2.0, "v"].iloc[0])


def _entropy(weights: pd.Series) -> float | None:
    clean = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    clean = clean[clean > 0]
    total = float(clean.sum())
    if total <= 0.0:
        return None
    p = clean / total
    entropy = float(-(p * p.map(math.log)).sum())
    return entropy / math.log(len(p)) if len(p) > 1 else 0.0


def _concentration_index(weights: pd.Series) -> float | None:
    clean = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    clean = clean[clean > 0]
    total = float(clean.sum())
    if total <= 0.0:
        return None
    p = clean / total
    return float((p * p).sum())


def _weighted_moments(values: pd.Series, weights: pd.Series) -> tuple[float | None, float | None]:
    clean = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "w": pd.to_numeric(weights, errors="coerce")}).dropna()
    clean = clean[clean["w"] > 0]
    total = float(clean["w"].sum())
    if len(clean) < 2 or total <= 0.0:
        return None, None
    mean = float((clean["v"] * clean["w"]).sum() / total)
    var = float((((clean["v"] - mean) ** 2) * clean["w"]).sum() / total)
    if var <= 0.0:
        return 0.0, 0.0
    std = math.sqrt(var)
    skew = float((((clean["v"] - mean) / std) ** 3 * clean["w"]).sum() / total)
    kurt = float((((clean["v"] - mean) / std) ** 4 * clean["w"]).sum() / total)
    return skew, kurt


def _average_strike_spacing(strikes: pd.Series) -> float | None:
    unique = sorted(set(float(x) for x in pd.to_numeric(strikes, errors="coerce").dropna()))
    if len(unique) < 2:
        return None
    gaps = [b - a for a, b in zip(unique, unique[1:]) if b > a]
    return float(pd.Series(gaps).median()) if gaps else None


def _significant_strikes(frame: pd.DataFrame) -> list[float]:
    weights = pd.to_numeric(frame["total_abs_gex"], errors="coerce").fillna(0.0)
    if weights.empty or float(weights.max()) <= 0.0:
        return []
    threshold = max(float(weights.quantile(0.75)), float(weights.max()) * 0.15)
    return sorted(float(x) for x in frame.loc[weights >= threshold, "strike"].dropna().tolist())


def _vacuum_score(frame: pd.DataFrame, *, spot: float | None) -> float | None:
    strikes = _significant_strikes(frame)
    if len(strikes) < 2:
        return None
    largest_gap = max(b - a for a, b in zip(strikes, strikes[1:]))
    return _safe_div(largest_gap, spot)


def _side_vacuum(frame: pd.DataFrame, *, spot: float | None, side: str) -> float | None:
    spot_num = _num(spot)
    strikes = _significant_strikes(frame)
    if spot_num is None:
        return None
    strikes = [x for x in strikes if (x > spot_num if side == "above" else x < spot_num)]
    if len(strikes) < 2:
        return None
    return _safe_div(max(b - a for a, b in zip(strikes, strikes[1:])), spot_num)


def _compression_score(frame: pd.DataFrame, *, spot: float | None) -> float | None:
    strikes = _significant_strikes(frame)
    if len(strikes) < 2:
        return None
    gaps = [b - a for a, b in zip(strikes, strikes[1:]) if b > a]
    median_gap = float(pd.Series(gaps).median()) if gaps else None
    return _safe_div(spot, median_gap)


def _magnet_features(frame: pd.DataFrame, *, magnet: float | None) -> tuple[float | None, float | None, float | None]:
    magnet_num = _num(magnet)
    total_abs = float(frame["total_abs_gex"].abs().sum()) if "total_abs_gex" in frame else 0.0
    if magnet_num is None or frame.empty:
        return None, None, None
    idx = (frame["strike"] - magnet_num).abs().idxmin()
    strength = float(abs(frame.loc[idx, "net_gex"]))
    relative = strength / total_abs if total_abs else None
    ordered = frame.assign(_abs_net=frame["net_gex"].abs()).sort_values("_abs_net", ascending=False)
    competitors = ordered[ordered["strike"] != float(frame.loc[idx, "strike"])]
    competitor = float(competitors["strike"].iloc[0]) if not competitors.empty else None
    return strength, relative, competitor


def _gamma_density(frame: pd.DataFrame, *, spot: float | None, pct: float) -> float | None:
    spot_num = _num(spot)
    if spot_num is None:
        return None
    band = abs(float(spot_num) * float(pct))
    near = frame[(frame["strike"] >= float(spot_num) - band) & (frame["strike"] <= float(spot_num) + band)]
    return float(near["total_abs_gex"].abs().sum())


def _level_strength(frame: pd.DataFrame, level: Any, col: str) -> float | None:
    level_num = _num(level)
    if level_num is None or frame.empty or col not in frame:
        return None
    idx = (frame["strike"] - level_num).abs().idxmin()
    return float(frame.loc[idx, col])


def _nearest_level(levels: dict[str, Any], *, spot: float | None) -> tuple[str | None, float | None]:
    spot_num = _num(spot)
    if spot_num is None:
        return None, None
    candidates = {
        "Gamma Flip": levels.get("gamma_flip"),
        "Call Wall": levels.get("call_wall"),
        "Put Wall": levels.get("put_wall"),
        "Magnet": levels.get("nearest_magnet"),
        "Vega Wall": levels.get("vega_wall"),
        "Ceiling": levels.get("call_wall"),
        "Floor": levels.get("put_wall"),
    }
    clean = [(name, _pct_to(value, spot_num)) for name, value in candidates.items() if _num(value) is not None]
    if not clean:
        return None, None
    name, dist = min(clean, key=lambda x: abs(float(x[1] or 0.0)))
    return name, dist


def _write_daily_frame(path: Path, rows: list[dict[str, Any]], dedupe_cols: list[str]) -> None:
    if rows:
        frame = pd.DataFrame(rows)
    elif path.exists():
        return
    else:
        frame = pd.DataFrame(columns=dedupe_cols)
    if path.exists():
        prior = pd.read_parquet(path)
        frame = pd.concat([prior, frame], ignore_index=True)
    keep_cols = [col for col in dedupe_cols if col in frame.columns]
    if keep_cols:
        frame = frame.drop_duplicates(keep_cols, keep="last")
    frame.to_parquet(path, index=False)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture dealer-positioning option snapshots.")
    parser.add_argument("--universe", default=str(UNIVERSE))
    parser.add_argument("--output-root", default=str(OUT_ROOT))
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--scopes", nargs="*", default=list(SCOPES))
    parser.add_argument("--all-universe", action="store_true", help="Use all rows instead of only is_eligible names.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strike-window-pct", type=float, default=0.50)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--min-total-oi", type=float, default=DEFAULT_MIN_TOTAL_OI)
    parser.add_argument("--min-option-volume", type=float, default=DEFAULT_MIN_OPTION_VOLUME)
    parser.add_argument("--min-adv", type=float, default=DEFAULT_MIN_ADV)
    parser.add_argument("--min-market-cap", type=float, default=DEFAULT_MIN_MARKET_CAP)
    parser.add_argument("--no-liquidity-prefilter", action="store_true")
    parser.add_argument("--count-only", action="store_true", help="Print universe/liquidity prefilter counts without API calls.")
    parser.add_argument("--snapshot-date", default=None, help="Storage/evaluation date, YYYY-MM-DD. Defaults to today ET.")
    parser.add_argument("--ref-date", default=None, help="Expiration-reference date, YYYY-MM-DD. Defaults to snapshot date.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel symbol/scope fetch workers (default 1).")
    parser.add_argument(
        "--no-publish-states",
        action="store_true",
        help="Capture only; do not publish DEALER states to the governed path.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    if args.count_only:
        counts = count_candidate_symbols(
            universe_path=Path(args.universe),
            eligible_only=not args.all_universe,
            liquidity_prefilter=not args.no_liquidity_prefilter,
            min_adv=float(args.min_adv),
            min_market_cap=float(args.min_market_cap),
        )
        print(json.dumps(counts, sort_keys=True))
        return 0
    symbols = load_symbols(
        universe_path=Path(args.universe),
        tickers=args.tickers,
        eligible_only=not args.all_universe,
        liquidity_prefilter=not args.no_liquidity_prefilter and not args.tickers,
        min_adv=float(args.min_adv),
        min_market_cap=float(args.min_market_cap),
        limit=args.limit,
    )
    if not symbols:
        raise SystemExit("no symbols selected")
    result = capture_snapshots(
        publish_states=not args.no_publish_states,
        symbols=symbols,
        scopes=tuple(str(scope) for scope in args.scopes),
        output_root=Path(args.output_root),
        strike_window_pct=float(args.strike_window_pct),
        sleep_seconds=float(args.sleep_seconds),
        min_total_oi=float(args.min_total_oi),
        min_option_volume=float(args.min_option_volume),
        min_adv=float(args.min_adv),
        min_market_cap=float(args.min_market_cap),
        universe_path=Path(args.universe),
        snapshot_date=args.snapshot_date,
        ref_date=args.ref_date,
        workers=int(args.workers),
    )
    print(
        "dealer snapshot capture complete: "
        f"candidates={result.candidate_symbols} skipped={result.skipped_symbols} scopes={result.scopes} "
        f"summaries={result.summary_rows} strikes={result.strike_rows} errors={result.errors} output={result.output_dir}"
    )
    return 0 if result.summary_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
