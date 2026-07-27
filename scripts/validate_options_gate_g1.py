#!/usr/bin/env python
"""Gate G1 validation for the options-instrument-routing experiment.

Plan: docs/superpowers/plans/2026-07-25-options-instrument-routing-experiment.md
(Phase 1, "Validation (Gate G1)"). This script is the single reproducible entry
point for the three sub-tasks:

  G1a -- cross-source IV check: compare Alpaca-implied IV (from real option
         bars, inverted via research.options_lab.pricing.implied_vol) against
         Schwab's reported call_iv/put_iv in the 16 days of
         Data/dealer_positioning/historical_snapshots/*/dealer_strike_ladder.parquet.

  G1b -- real-fill reprice: reprice all 575 real closed option trades in
         Data/analysis/multi_ticker_swing_live/paired_option_trades.csv from
         real Alpaca option bars at the real entry/exit timestamps, and
         compare to the actual fills.

  G1c -- fills.py calibration: derive a spread model from the G1b entry/exit
         residuals (Schwab snapshots carry no bid/ask column -- verified
         below -- so the plan's documented fallback applies) and report the
         diagnostics that back research/options_lab/fills.py's calibrated
         constants.

Run: .venv/bin/python scripts/validate_options_gate_g1.py [--task g1a,g1b,g1c]
Outputs: research/options_experiment/data/g1a_iv_check.csv,
         research/options_experiment/data/g1b_reprice.csv,
         printed summary tables (also captured into 01_gate_g1.md by hand).

Network calls are real (Alpaca reads), cached on disk by chain_cache.py, and
sized deliberately: G1a samples ~12 liquid names x 8 reliable-timestamp days
(not a bulk universe pull); G1b touches all 575 real trades because the task
requires an honest coverage number, but batches bar fetches per (root,
expiry) group (288 groups, not 575*2 individual calls).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from research.options_lab import chain_cache, pricing  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("validate_g1")

SNAPSHOT_DIR = REPO_ROOT / "Data" / "dealer_positioning" / "historical_snapshots"
PAIRED_TRADES_PATH = (
    REPO_ROOT / "Data" / "analysis" / "multi_ticker_swing_live" / "paired_option_trades.csv"
)
OUT_DIR = REPO_ROOT / "research" / "options_experiment" / "data"

# --------------------------------------------------------------------------
# G1a config
# --------------------------------------------------------------------------

# Days where captured_at/timestamp cluster tightly around 15:45 ET (verified
# by inspection -- see 01_gate_g1.md "Timestamp alignment" section). The
# other 8 available days (20260702, 20260707, 20260708, 20260709, 20260712,
# 20260713, 20260722) either have a captured_at far from 15:45 ET (looks like
# a backfill/reprocessing run at an arbitrary later wall-clock time) or a
# corrupt/schema-empty file, and are excluded from the primary comparison.
G1A_RELIABLE_DAYS = [
    "2026-07-14", "2026-07-15", "2026-07-16", "2026-07-17",
    "2026-07-20", "2026-07-21", "2026-07-23", "2026-07-24",
]

# A fixed, liquid, diverse sample (not cherry-picked post-hoc -- chosen by
# ranking total daily_week call_oi+put_oi across the reliable days, before
# any IV comparison was run). Kept small deliberately per the task's
# "sample intelligently, don't bulk-download" instruction.
G1A_CANDIDATE_SYMBOLS = [
    "NVDA", "MSTR", "NFLX", "TSLA", "IREN", "PLTR",
    "NBIS", "SOFI", "SMCI", "HOOD", "NOK", "INTC",
]

VEGA_DOLLAR_THRESHOLD = 0.05  # per plan's Phase 1b finding: vega < ~$0.05/vol-pt is ill-conditioned
MONEYNESS_BINS = [0.0, 0.02, 0.05, 0.10, 0.20, np.inf]
MONEYNESS_LABELS = ["0-2%", "2-5%", "5-10%", "10-20%", "20%+"]
DTE_BINS = [-0.01, 2, 7, 21, 45, 90, np.inf]
DTE_LABELS = ["0-2", "3-7", "8-21", "22-45", "46-90", "90+"]


def _load_ladder(day: str) -> pd.DataFrame | None:
    day_key = day.replace("-", "")
    path = SNAPSHOT_DIR / day_key / "dealer_strike_ladder.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("ladder %s: unreadable (%s)", day, exc)
        return None
    if df.empty or "expirations" not in df.columns:
        logger.warning("ladder %s: empty/schema-broken (cols=%s)", day, list(df.columns))
        return None
    return df


def check_bid_ask_columns() -> bool:
    """Verify (never assume) whether the Schwab snapshots carry real bid/ask.
    Returns True if any bid/ask-like column is present in any usable snapshot.
    """
    found_cols: set[str] = set()
    for day in G1A_RELIABLE_DAYS:
        df = _load_ladder(day)
        if df is None:
            continue
        found_cols |= {c for c in df.columns if "bid" in c.lower() or "ask" in c.lower()}
    return bool(found_cols)


def run_g1a() -> pd.DataFrame:
    rows: list[dict] = []
    for day in G1A_RELIABLE_DAYS:
        ladder = _load_ladder(day)
        if ladder is None:
            continue
        for symbol in G1A_CANDIDATE_SYMBOLS:
            dw = ladder[(ladder["symbol"] == symbol) & (ladder["scope"] == "daily_week")].copy()
            if dw.empty:
                continue
            dw["n_exp"] = dw["expirations"].apply(lambda s: len(json.loads(s)))
            dw = dw[dw["n_exp"] == 1]
            if dw.empty:
                continue  # basis requires an unambiguous single-expiry ladder (see docstring)
            expiry = json.loads(dw["expirations"].iloc[0])[0]
            spot = float(dw["spot"].iloc[0])
            if not (np.isfinite(spot) and spot > 0):
                continue

            try:
                contracts = chain_cache.discover_contracts(symbol, expiry, expiry)
            except Exception as exc:
                logger.warning("discover_contracts failed for %s/%s: %s", symbol, expiry, exc)
                continue
            if contracts.empty:
                continue

            long_rows = []
            for _, r in dw.iterrows():
                for right, iv_col in (("C", "call_iv"), ("P", "put_iv")):
                    long_rows.append({"strike": r["strike"], "right": right, "schwab_iv": r[iv_col]})
            schwab_long = pd.DataFrame(long_rows)
            merged = schwab_long.merge(contracts, on=["strike", "right"], how="inner")
            if merged.empty:
                continue

            osi_syms = merged["osi_symbol"].tolist()
            try:
                bars = chain_cache.fetch_bars(
                    osi_syms, "30Min", f"{day}T13:30:00Z", f"{day}T20:15:00Z"
                )
            except Exception as exc:
                logger.warning("fetch_bars failed for %s/%s: %s", symbol, expiry, exc)
                continue
            if bars.empty:
                continue
            bars = bars.copy()
            bars["t"] = pd.to_datetime(bars["t"], utc=True)
            target_bar_start = pd.Timestamp(f"{day}T19:30:00Z")  # 15:30 ET bar (contains 15:45 ET)
            asof_target = pd.Timestamp(f"{day}T19:45:00Z")  # 15:45 ET, the Schwab capture target
            expiry_close = pd.Timestamp(f"{expiry}T20:00:00Z")  # 16:00 ET, standard equity-option expiry
            T = (expiry_close - asof_target).total_seconds() / (365.25 * 86400.0)
            if T <= 0:
                continue
            try:
                r_rate = pricing.risk_free_rate(day, tenor_years=T)
            except Exception as exc:
                logger.warning("risk_free_rate failed for %s: %s", day, exc)
                continue

            for _, row in merged.iterrows():
                sub = bars[bars["osi_symbol"] == row["osi_symbol"]]
                if sub.empty:
                    continue
                gaps = (sub["t"] - target_bar_start).abs()
                b = sub.loc[gaps.idxmin()]
                gap_min = float(gaps.min().total_seconds() / 60.0)
                price = b["vw"] if pd.notna(b["vw"]) and b["vw"] > 0 else b["c"]
                if pd.isna(price) or price <= 0:
                    continue
                alpaca_iv = pricing.implied_vol(
                    price=float(price), S=spot, K=float(row["strike"]), T=T,
                    r=r_rate, q=0.0, right=row["right"], american=False,
                )
                vega_dollar = None
                if alpaca_iv is not None:
                    g = pricing.bsm_greeks(
                        S=spot, K=float(row["strike"]), T=T, r=r_rate, q=0.0,
                        sigma=alpaca_iv, right=row["right"],
                    )
                    vega_dollar = float(g.vega) / 100.0
                rows.append({
                    "day": day, "symbol": symbol, "expiry": expiry,
                    "strike": float(row["strike"]), "right": row["right"],
                    "spot": spot, "schwab_iv_pct": float(row["schwab_iv"]),
                    "alpaca_price": float(price), "bar_gap_minutes": gap_min,
                    "alpaca_iv": alpaca_iv,
                    "abs_err_vol_pts": (
                        abs(alpaca_iv * 100.0 - float(row["schwab_iv"])) if alpaca_iv is not None else np.nan
                    ),
                    "vega_dollar_per_volpt": vega_dollar,
                    "moneyness": abs(np.log(float(row["strike"]) / spot)),
                    "dte_calendar": (pd.Timestamp(expiry) - pd.Timestamp(day)).days,
                    "open_interest": row.get("open_interest"),
                })
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "g1a_iv_check.csv", index=False)
    return out


def summarize_g1a(df: pd.DataFrame) -> None:
    print(f"\n=== G1a: cross-source IV check ===")
    print(f"total (symbol,day,strike,right) rows attempted: {len(df)}")
    n_solved = df["alpaca_iv"].notna().sum()
    print(f"alpaca_iv solved (not None): {n_solved} ({n_solved / max(len(df), 1):.1%})")
    print(f"median bar_gap_minutes: {df['bar_gap_minutes'].median():.1f}")

    # exact call_iv == put_iv duplication check (data-quality caveat)
    dup_frac_all = []
    for day in G1A_RELIABLE_DAYS:
        ladder = _load_ladder(day)
        if ladder is None:
            continue
        diff = (ladder["call_iv"] - ladder["put_iv"]).abs()
        dup_frac_all.append((diff < 1e-6).mean())
    if dup_frac_all:
        print(f"Schwab call_iv==put_iv (exact/near-exact) fraction across reliable days: "
              f"{np.mean(dup_frac_all):.1%}")

    priced = df.dropna(subset=["alpaca_iv", "abs_err_vol_pts"]).copy()
    priced["moneyness_bucket"] = pd.cut(priced["moneyness"], MONEYNESS_BINS, labels=MONEYNESS_LABELS)
    priced["dte_bucket"] = pd.cut(priced["dte_calendar"], DTE_BINS, labels=DTE_LABELS)
    priced["vega_tier"] = np.where(
        priced["vega_dollar_per_volpt"] >= VEGA_DOLLAR_THRESHOLD, "high_vega", "low_vega"
    )

    for tier in ("high_vega", "low_vega"):
        sub = priced[priced["vega_tier"] == tier]
        if sub.empty:
            continue
        print(f"\n--- vega tier: {tier} (n={len(sub)}) ---")
        print(f"median abs err (vol pts): {sub['abs_err_vol_pts'].median():.2f}, "
              f"IQR: [{sub['abs_err_vol_pts'].quantile(.25):.2f}, {sub['abs_err_vol_pts'].quantile(.75):.2f}]")
        table = sub.groupby(["moneyness_bucket", "dte_bucket"], observed=True)["abs_err_vol_pts"].agg(
            ["median", lambda s: s.quantile(.25), lambda s: s.quantile(.75), "count"]
        )
        table.columns = ["median", "q25", "q75", "n"]
        print(table.to_string())


# --------------------------------------------------------------------------
# G1b: real-fill reprice
# --------------------------------------------------------------------------

def run_g1b() -> pd.DataFrame:
    trades = pd.read_csv(PAIRED_TRADES_PATH)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)

    # Group by (root, expiry) so bar fetches batch (shared start/end -> the
    # symbols in a group share one gap window -> chain_cache batches them
    # into as few Alpaca calls as possible instead of one call per trade).
    parsed = {}
    for sym in trades["symbol"].unique():
        try:
            parsed[sym] = chain_cache.parse_osi_symbol(sym)
        except ValueError:
            parsed[sym] = None

    groups: dict[tuple[str, str], dict] = {}
    for _, row in trades.iterrows():
        p = parsed.get(row["symbol"])
        if p is None:
            continue
        key = (p.root, p.expiry)
        g = groups.setdefault(key, {"symbols": set(), "min_t": row["entry_time"], "max_t": row["exit_time"]})
        g["symbols"].add(row["symbol"])
        g["min_t"] = min(g["min_t"], row["entry_time"])
        g["max_t"] = max(g["max_t"], row["exit_time"])

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for (root, expiry), g in groups.items():
        start = (g["min_t"] - pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end = (g["max_t"] + pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            bars = chain_cache.fetch_bars(sorted(g["symbols"]), "1Min", start, end)
        except Exception as exc:
            logger.warning("fetch_bars failed for group (%s,%s): %s", root, expiry, exc)
            continue
        if bars.empty:
            continue
        bars = bars.copy()
        bars["t"] = pd.to_datetime(bars["t"], utc=True)
        for sym in g["symbols"]:
            sub = bars[bars["osi_symbol"] == sym]
            if not sub.empty:
                bars_by_symbol[sym] = sub.sort_values("t")

    def model_price_at(sym: str, ts: pd.Timestamp, max_lookback_min: float = 60.0):
        sub = bars_by_symbol.get(sym)
        if sub is None or sub.empty:
            return None, None
        at_or_before = sub[sub["t"] <= ts]
        if at_or_before.empty:
            return None, None
        last = at_or_before.iloc[-1]
        gap_min = (ts - last["t"]).total_seconds() / 60.0
        if gap_min > max_lookback_min:
            return None, gap_min
        price = last["c"]
        return (float(price) if pd.notna(price) else None), gap_min

    out_rows = []
    for _, row in trades.iterrows():
        sym = row["symbol"]
        modeled_entry, entry_gap = model_price_at(sym, row["entry_time"])
        modeled_exit, exit_gap = model_price_at(sym, row["exit_time"])
        repriceable = modeled_entry is not None and modeled_exit is not None
        entry_bias_pct = (
            (modeled_entry - row["entry_price_option"]) / row["entry_price_option"]
            if modeled_entry is not None and row["entry_price_option"] else np.nan
        )
        exit_bias_pct = (
            (modeled_exit - row["exit_price_option"]) / row["exit_price_option"]
            if modeled_exit is not None and row["exit_price_option"] else np.nan
        )
        sim_pnl = (
            (modeled_exit - modeled_entry) * row["qty"] * 100.0 if repriceable else np.nan
        )
        out_rows.append({
            "symbol": sym, "ticker": row["ticker"], "option_type": row["option_type"],
            "dte_at_entry": row["dte_at_entry"],
            "entry_time": row["entry_time"], "exit_time": row["exit_time"],
            "actual_entry_price": row["entry_price_option"], "actual_exit_price": row["exit_price_option"],
            "modeled_entry_price": modeled_entry, "modeled_exit_price": modeled_exit,
            "entry_gap_minutes": entry_gap, "exit_gap_minutes": exit_gap,
            "entry_bias_pct": entry_bias_pct, "exit_bias_pct": exit_bias_pct,
            "actual_pnl": row["pnl_dollars"], "simulated_pnl": sim_pnl,
            "qty": row["qty"], "repriceable": repriceable,
        })
    out = pd.DataFrame(out_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "g1b_reprice.csv", index=False)
    return out


def summarize_g1b(df: pd.DataFrame) -> None:
    print(f"\n=== G1b: real-fill reprice ===")
    n = len(df)
    n_ok = int(df["repriceable"].sum())
    print(f"total trades: {n}, repriceable (both entry & exit): {n_ok} ({n_ok / n:.1%})")
    print(f"NOT repriceable: {n - n_ok} ({(n - n_ok) / n:.1%})")

    ok = df[df["repriceable"]].copy()
    for label, col in (("entry_bias_pct", "entry_bias_pct"), ("exit_bias_pct", "exit_bias_pct")):
        s = ok[col].dropna()
        print(f"\n{label}: n={len(s)}, signed median={s.median():.4f}, "
              f"IQR=[{s.quantile(.25):.4f}, {s.quantile(.75):.4f}], mean={s.mean():.4f}")

    print(f"\nmedian entry_gap_minutes: {ok['entry_gap_minutes'].median():.2f}, "
          f"median exit_gap_minutes: {ok['exit_gap_minutes'].median():.2f}")

    corr = ok[["simulated_pnl", "actual_pnl"]].dropna().corr().iloc[0, 1]
    mae = (ok["simulated_pnl"] - ok["actual_pnl"]).abs().median()
    print(f"\nsimulated vs actual PnL: corr={corr:.3f}, median abs error=${mae:.2f}")
    print(f"total actual PnL: ${ok['actual_pnl'].sum():,.2f}, total simulated PnL: ${ok['simulated_pnl'].sum():,.2f}")

    print("\n--- by side ---")
    for side in ("C", "P"):
        sub = ok[ok["option_type"] == side]
        if sub.empty:
            continue
        print(f"{side}: n={len(sub)}, entry_bias median={sub['entry_bias_pct'].median():.4f}, "
              f"exit_bias median={sub['exit_bias_pct'].median():.4f}, "
              f"actual_pnl_sum=${sub['actual_pnl'].sum():,.2f}, sim_pnl_sum=${sub['simulated_pnl'].sum():,.2f}")

    print("\n--- by DTE bucket ---")
    ok["dte_bucket"] = pd.cut(ok["dte_at_entry"], DTE_BINS, labels=DTE_LABELS)
    table = ok.groupby("dte_bucket", observed=True).agg(
        n=("entry_bias_pct", "size"),
        entry_bias_median=("entry_bias_pct", "median"),
        exit_bias_median=("exit_bias_pct", "median"),
        actual_pnl_sum=("actual_pnl", "sum"),
        sim_pnl_sum=("simulated_pnl", "sum"),
    )
    print(table.to_string())


# --------------------------------------------------------------------------
# G1c: fills.py calibration inputs (derived from G1b residuals, per the
# plan's documented fallback since dealer_strike_ladder has no bid/ask)
# --------------------------------------------------------------------------

def run_g1c(g1b_df: pd.DataFrame) -> pd.DataFrame:
    ok = g1b_df[g1b_df["repriceable"]].copy()
    ok = ok[(ok["modeled_entry_price"] > 0)]
    # Round-trip cost proxy: buy above modeled price, sell below modeled
    # price -> (actual_entry - modeled_entry) - (actual_exit - modeled_exit)
    # approximates one full bid/ask spread paid over the round trip (see
    # research/options_lab/fills.py docstring for the derivation and caveats).
    ok["round_trip_cost_dollars"] = (
        (ok["actual_entry_price"] - ok["modeled_entry_price"])
        - (ok["actual_exit_price"] - ok["modeled_exit_price"])
    )
    ok["round_trip_cost_pct"] = ok["round_trip_cost_dollars"] / ok["modeled_entry_price"]

    trades = pd.read_csv(PAIRED_TRADES_PATH)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    ok = ok.merge(
        trades[["symbol", "entry_time", "strike", "entry_price_underlying"]].drop_duplicates(
            subset=["symbol", "entry_time"]
        ),
        on=["symbol", "entry_time"], how="left",
    )
    ok["moneyness"] = (ok["strike"] / ok["entry_price_underlying"]).apply(
        lambda x: abs(np.log(x)) if pd.notna(x) and x > 0 else np.nan
    )

    # underlying ADV: trailing 20-day mean volume strictly before entry_date (no lookahead)
    adv_cache: dict[str, pd.DataFrame] = {}
    def underlying_adv(ticker: str, entry_time: pd.Timestamp) -> float | None:
        if ticker not in adv_cache:
            path = REPO_ROOT / "Data" / "shared" / "bars" / "1d" / f"{ticker}.parquet"
            if not path.exists():
                adv_cache[ticker] = None
            else:
                adv_cache[ticker] = pd.read_parquet(path, columns=["timestamp", "volume"])
        bars = adv_cache[ticker]
        if bars is None:
            return None
        prior = bars[bars["timestamp"] < entry_time]
        if len(prior) < 5:
            return None
        return float(prior["volume"].tail(20).mean())

    ok["underlying_adv"] = ok.apply(lambda r: underlying_adv(r["ticker"], r["entry_time"]), axis=1)

    # OI/volume per contract, from the same cached bars/contracts used above
    def contract_oi(symbol: str) -> float | None:
        try:
            p = chain_cache.parse_osi_symbol(symbol)
        except ValueError:
            return None
        try:
            c = chain_cache.discover_contracts(p.root, p.expiry, p.expiry)
        except Exception:
            return None
        m = c.loc[c["osi_symbol"] == symbol]
        if m.empty or pd.isna(m.iloc[0]["open_interest"]):
            return None
        return float(m.iloc[0]["open_interest"])

    ok["open_interest"] = ok["symbol"].apply(contract_oi)

    # Contract volume: sum of real bar volume ('v') over the same window
    # already fetched/cached for this trade in run_g1b (entry-30min ..
    # exit+30min) -- a disk-cache hit, no new network calls, since that
    # exact window was already recorded in chain_cache's meta sidecar.
    def contract_volume(row) -> float | None:
        start = (row["entry_time"] - pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end = (row["exit_time"] + pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            bars = chain_cache.fetch_bars([row["symbol"]], "1Min", start, end)
        except Exception:
            return None
        if bars.empty or "v" not in bars.columns:
            return None
        sub = bars[bars["osi_symbol"] == row["symbol"]]
        if sub.empty or sub["v"].isna().all():
            return None
        return float(sub["v"].fillna(0).sum())

    ok["exit_time"] = pd.to_datetime(ok["exit_time"], utc=True)
    ok["contract_volume"] = ok.apply(contract_volume, axis=1)

    fit_df = ok.dropna(
        subset=["round_trip_cost_pct", "moneyness", "dte_at_entry", "open_interest",
                "contract_volume", "underlying_adv"]
    ).copy()
    # Winsorize the noisy proxy target before fitting (bar-close-based, not
    # true bid/ask -- extreme outliers are almost certainly stale/thin bars,
    # not real spread).
    lo, hi = fit_df["round_trip_cost_pct"].quantile([0.02, 0.98])
    fit_df = fit_df[(fit_df["round_trip_cost_pct"] >= lo) & (fit_df["round_trip_cost_pct"] <= hi)]
    fit_df = fit_df[fit_df["round_trip_cost_pct"] > 0]  # log-target needs positive cost

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fit_df.to_csv(OUT_DIR / "g1c_calibration_inputs.csv", index=False)
    return fit_df


def summarize_g1c(fit_df: pd.DataFrame) -> dict:
    print(f"\n=== G1c: fills.py spread calibration (fallback: G1b fill residuals) ===")
    print(f"usable trades for calibration fit: {len(fit_df)}")
    if fit_df.empty:
        print("insufficient data to calibrate -- fills.py will ship with an uncalibrated floor only")
        return {}

    print(f"round_trip_cost_pct: median={fit_df['round_trip_cost_pct'].median():.4f}, "
          f"IQR=[{fit_df['round_trip_cost_pct'].quantile(.25):.4f}, "
          f"{fit_df['round_trip_cost_pct'].quantile(.75):.4f}]")

    y = np.log(fit_df["round_trip_cost_pct"].to_numpy())
    X = np.column_stack([
        np.ones(len(fit_df)),
        fit_df["moneyness"].to_numpy(),
        np.log1p(fit_df["dte_at_entry"].to_numpy()),
        np.log1p(fit_df["open_interest"].to_numpy()),
        np.log1p(fit_df["contract_volume"].to_numpy()),
        np.log1p(fit_df["underlying_adv"].to_numpy()),
    ])
    coefs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coefs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    resid_std = float(np.std(y - y_hat, ddof=len(coefs)))

    names = ["intercept", "moneyness", "log1p(dte)", "log1p(oi)", "log1p(volume)", "log1p(underlying_adv)"]
    print("\nOLS fit: log(round_trip_cost_pct) ~ moneyness + log1p(dte) + log1p(oi) + log1p(volume) + log1p(adv)")
    for name, c in zip(names, coefs):
        print(f"  {name:24s} {c: .6f}")
    print(f"  R^2 = {r2:.4f}, residual std (log space) = {resid_std:.4f}, n = {len(fit_df)}")

    return {
        "coefficients": dict(zip(names, [float(c) for c in coefs])),
        "r2": r2,
        "residual_std_log": resid_std,
        "n": int(len(fit_df)),
        "median_round_trip_cost_pct": float(fit_df["round_trip_cost_pct"].median()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="g1a,g1b,g1c", help="comma-separated: g1a,g1b,g1c")
    args = ap.parse_args()
    tasks = set(args.task.split(","))

    has_bid_ask = check_bid_ask_columns()
    print(f"dealer_strike_ladder.parquet bid/ask columns present: {has_bid_ask}")

    g1b_df = None
    if "g1b" in tasks or "g1c" in tasks:
        g1b_df = run_g1b()
        summarize_g1b(g1b_df)

    if "g1a" in tasks:
        g1a_df = run_g1a()
        summarize_g1a(g1a_df)

    if "g1c" in tasks:
        assert g1b_df is not None
        fit_df = run_g1c(g1b_df)
        calib = summarize_g1c(fit_df)
        with open(OUT_DIR / "g1c_calibration.json", "w") as f:
            json.dump(calib, f, indent=2)


if __name__ == "__main__":
    main()
