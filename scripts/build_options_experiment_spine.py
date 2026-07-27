"""Build the Phase-0 options-routing-experiment signal spine.

Normalizes every in-scope module's historical trades/signals into one
schema (see SPINE_COLUMNS) and writes:

  - research/options_experiment/data/signal_spine.parquet
  - research/options_experiment/00_inventory.md

Plan: docs/superpowers/plans/2026-07-25-options-instrument-routing-experiment.md
Scope: this script is Phase 0 ("Experiment spine") ONLY. It does not touch
option chains, pricing, liquidity gates, or strategy/routing logic -- it
normalizes historical signal/trade provenance so later phases have one
honest, documented input to replay.

Design rules followed throughout (see AGENTS.md):
  - signal_ts (decision time) is kept distinct from entry_ts (fill time)
    wherever the source actually records both; where a source only has one
    timestamp, signal_ts is set equal to it and that fact is recorded here.
  - Never fabricate a column. Missing fields are left null and documented
    in the NOTES list below, which is dumped verbatim into the inventory.
  - All timestamps are coerced to UTC tz-aware.

Run:
    .venv/bin/python scripts/build_options_experiment_spine.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "research" / "options_experiment"
DATA_OUT = OUT_DIR / "data" / "signal_spine.parquet"
INVENTORY_OUT = OUT_DIR / "00_inventory.md"

BARS_4H_DIR = REPO_ROOT / "Data" / "shared" / "bars" / "4h"
BARS_1D_DIR = REPO_ROOT / "Data" / "shared" / "bars" / "1d"
DEALER_SNAPSHOT_DIR = REPO_ROOT / "Data" / "dealer_positioning" / "historical_snapshots"

OPTION_HISTORY_START = pd.Timestamp("2024-02-01", tz="UTC")

SPINE_COLUMNS = [
    "module", "ticker", "signal_ts", "entry_ts", "exit_ts", "direction",
    "entry_px_underlying", "exit_px_underlying", "exit_reason", "bars_held",
    "atr_at_entry", "tp_price", "sl_price", "score", "cadence",
    "source_file", "provenance",
]

VALID_PROVENANCE = {"backtest_frozen_test", "backtest_insample", "live_real"}
VALID_CADENCE = {"4h", "30m", "daily", "intraday"}

# Free-text gap/decision notes collected while building each source. Every
# entry documents something a reader needs to know before trusting a row
# count or a column. Dumped verbatim (as a numbered list) into the inventory.
NOTES: list[str] = []


def _note(text: str) -> None:
    NOTES.append(" ".join(text.split()))  # normalize internal whitespace/newlines


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes and column order for one source's normalized frame."""
    df = df.reindex(columns=SPINE_COLUMNS).copy()
    for c in ["signal_ts", "entry_ts", "exit_ts"]:
        # format="ISO8601": several sources mix "YYYY-MM-DD HH:MM:SS+00:00" (space
        # separator) and "YYYY-MM-DDTHH:MM:SS+00:00" (T separator) timestamp strings
        # within the same column (e.g. meta_ranker's closed_trades.jsonl entry_bar vs.
        # a recovered entry_bar backfilled from live_signal_audit.jsonl). Without an
        # explicit format, pandas infers a format from the first value and silently
        # coerces every differently-formatted string in the column to NaT instead of
        # raising -- caught by inspecting a real row during validation.
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce", format="ISO8601")
    df["direction"] = pd.array(df["direction"], dtype="Int64")
    for c in ["entry_px_underlying", "exit_px_underlying", "bars_held", "atr_at_entry",
              "tp_price", "sl_price", "score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["module", "ticker", "exit_reason", "cadence", "source_file", "provenance"]:
        df[c] = df[c].astype("object")
    return df


# ---------------------------------------------------------------------------
# Source 1: momentum_expansion + multi_ticker_swing_htf (4H, frozen test)
# ---------------------------------------------------------------------------

def load_momentum_htf_4h() -> pd.DataFrame:
    path = REPO_ROOT / "backtests/position_sizing_4h/equal_notional_trades.parquet"
    raw = pd.read_parquet(path)
    module = np.where(raw["strategy"] == "momentum", "momentum_expansion", "multi_ticker_swing_htf")
    df = pd.DataFrame({
        "module": module,
        "ticker": raw["ticker"],
        "signal_ts": raw["entry_ts"],
        "entry_ts": raw["entry_ts"],
        "exit_ts": raw["exit_ts"],
        "direction": raw["direction"],
        "entry_px_underlying": raw["entry_price"],
        "exit_px_underlying": raw["exit_price"],
        "exit_reason": raw["exit_reason"],
        "bars_held": raw["bars_held"],
        "atr_at_entry": raw["atr_at_entry"],
        "tp_price": raw["tp_price"],
        "sl_price": raw["sl_price"],
        "score": raw["score"],
        "cadence": "4h",
        "source_file": _rel(path),
        "provenance": "backtest_frozen_test",
    })
    n_mom = int((raw["strategy"] == "momentum").sum())
    n_htf = int((raw["strategy"] == "htf").sum())
    _note(
        f"""Source: {_rel(path)} ({len(raw)} rows: {n_mom} momentum, {n_htf} htf).
        Verified via backtests/position_sizing_4h/method.json that both legs are literally the
        *_frozen_test_trades.parquet outputs of family_compare_clean (momentum xgb_classifier
        seed45, htf lgbm_classifier seed46) -- despite living under position_sizing_4h/, this file
        IS the frozen-test population for both modules, not in-sample. All rows are ingested
        regardless of the source's own `accepted` boolean column (that flag reflects whether one
        specific equal-notional position-sizing test's capital cap admitted the trade, not signal
        validity -- Phase 3 should be able to route every candidate signal, not only the ones that
        fit inside one sizing rule's book). signal_ts has no separate source column and is set equal
        to entry_ts (no decision timestamp distinct from the fill is recorded upstream in this
        ledger)."""
    )
    return _finalize(df)


def check_ev_experiments_final_subset() -> None:
    """htf_final_/momentum_final_frozen_test_trades.parquet: verify subset, document, don't ingest."""
    eq = pd.read_parquet(REPO_ROOT / "backtests/position_sizing_4h/equal_notional_trades.parquet")
    htf_final_path = REPO_ROOT / "backtests/ev_experiments_4h/htf_final_frozen_test_trades.parquet"
    mom_final_path = REPO_ROOT / "backtests/ev_experiments_4h/momentum_final_frozen_test_trades.parquet"
    htf_final = pd.read_parquet(htf_final_path)
    mom_final = pd.read_parquet(mom_final_path)

    eq_htf_keys = set(zip(eq.loc[eq.strategy == "htf", "ticker"], eq.loc[eq.strategy == "htf", "entry_ts"]))
    eq_mom_keys = set(zip(eq.loc[eq.strategy == "momentum", "ticker"], eq.loc[eq.strategy == "momentum", "entry_ts"]))
    htf_overlap = len(set(zip(htf_final.ticker, htf_final.entry_ts)) & eq_htf_keys)
    mom_overlap = len(set(zip(mom_final.ticker, mom_final.entry_ts)) & eq_mom_keys)

    if htf_overlap != len(htf_final) or mom_overlap != len(mom_final):
        raise AssertionError(
            "ev_experiments_4h 'final' frozen-test files are no longer a pure (ticker, entry_ts) "
            "subset of source 1 (equal_notional_trades.parquet) -- the dedupe assumption behind "
            "skipping them as a separate spine source no longer holds and must be re-checked "
            f"(htf {htf_overlap}/{len(htf_final)}, momentum {mom_overlap}/{len(mom_final)})."
        )

    _note(
        f"""{_rel(htf_final_path)} ({len(htf_final)} rows) and {_rel(mom_final_path)}
        ({len(mom_final)} rows) were checked against source 1 by exact (ticker, entry_ts) key:
        {htf_overlap}/{len(htf_final)} htf rows and {mom_overlap}/{len(mom_final)} momentum rows are
        EXACT subsets of the equal_notional_trades.parquet ledger already ingested above. These
        'final' files are the ev_experiments top-K / rank-EV-selected DEPLOYED-POLICY picks, not a
        distinct signal population, so they are NOT ingested as additional spine rows -- doing so
        would double-count the same underlying signals. If a later phase wants the deployed-subset
        specifically (rather than the full candidate population), filter the momentum_expansion /
        multi_ticker_swing_htf spine rows by (ticker, entry_ts) membership in these two files."""
    )


# ---------------------------------------------------------------------------
# Source 3: meta_ranker (4H, live real)
# ---------------------------------------------------------------------------

def _parse_occ_symbol(sym: str):
    """Parse an OCC option symbol -> (underlying, right, strike). None if not OCC-shaped."""
    m = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", sym)
    if not m:
        return None
    root, _expiry, right, strike = m.groups()
    return root, right, int(strike) / 1000.0


def load_meta_ranker() -> pd.DataFrame:
    closed_path = REPO_ROOT / "Data/inference/meta_ranker/closed_trades.jsonl"
    audit_path = REPO_ROOT / "Data/inference/meta_ranker/live_signal_audit.jsonl"

    closed = [json.loads(l) for l in closed_path.read_text().splitlines() if l.strip()]
    audit = [json.loads(l) for l in audit_path.read_text().splitlines() if l.strip()]

    # order_plan entry events give a (symbol -> [{bar, underlying, side, score}, ...]) history,
    # which lets us recover a missing entry_bar and join direction/score for the closed trades.
    # Bar strings are parsed to real Timestamps immediately: closed_trades.jsonl's "bar"/
    # "entry_bar" use a space separator ("2026-07-07 14:00:00+00:00") while
    # live_signal_audit.jsonl's "bar" uses a "T" separator ("2026-07-07T14:00:00+00:00") --
    # comparing/joining on the raw strings silently drops every match between the two files.
    entries_by_symbol: dict[str, list[dict]] = {}
    total_entry_events = 0
    for ev in audit:
        if ev.get("event") != "order_plan":
            continue
        sig_audits = ev.get("signal_audits", {})
        for p in ev.get("plan", []):
            if p.get("reason") != "entry":
                continue
            total_entry_events += 1
            sym = p["symbol"]
            occ = _parse_occ_symbol(sym)
            underlying = occ[0] if occ else sym
            sa = sig_audits.get(underlying, {})
            entries_by_symbol.setdefault(sym, []).append({
                "bar": pd.Timestamp(ev["bar"]),
                "side": sa.get("side"),
                "score": sa.get("score"),
            })

    rows = []
    n_missing_entry_bar_recovered = 0
    for r in closed:
        sym = r["order_symbol"]
        occ = _parse_occ_symbol(sym)
        underlying = occ[0] if occ else r["ticker"]
        is_option = r.get("route") == "option"
        exit_bar = pd.Timestamp(r["bar"])

        entry_bar = pd.Timestamp(r["entry_bar"]) if r.get("entry_bar") else None
        cand = sorted(entries_by_symbol.get(sym, []), key=lambda x: x["bar"])
        if entry_bar is None:
            before = [c for c in cand if c["bar"] < exit_bar]
            if before:
                entry_bar = before[-1]["bar"]
                n_missing_entry_bar_recovered += 1

        match = next((c for c in cand if c["bar"] == entry_bar), None)
        side = match["side"] if match else None
        score = match["score"] if match else None
        direction = 1 if side == "long" else (-1 if side == "short" else None)

        rows.append({
            "module": "meta_ranker",
            "ticker": underlying,
            "signal_ts": entry_bar,
            "entry_ts": entry_bar,
            "exit_ts": exit_bar,
            "direction": direction,
            "entry_px_underlying": None if is_option else r.get("entry_avg_price"),
            "exit_px_underlying": None if is_option else r.get("exit_fill_price"),
            "exit_reason": r.get("exit_reason"),
            "bars_held": r.get("runs_held"),
            "atr_at_entry": None,
            "tp_price": None,
            "sl_price": None,
            "score": score,
            "cadence": "4h",
            "source_file": _rel(closed_path),
            "provenance": "live_real",
        })

    n_open_approx = total_entry_events - len(closed)
    _note(
        f"""meta_ranker: {len(closed)} rows ingested from {_rel(closed_path)}. Recovered
        {n_missing_entry_bar_recovered} missing entry_bar value(s) by matching the same
        order_symbol's most recent order_plan 'entry' event strictly before the exit bar in
        {_rel(audit_path)} (a real join against real data, not a fabricated value). direction and
        score are also joined from that matched order_plan's own embedded
        signal_audits[underlying] (side long/short -> +1/-1; score = ranker score at entry).
        {_rel(audit_path)} records {total_entry_events} total order_plan 'entry' events across its
        history vs only {len(closed)} closed trades -- approximately {n_open_approx} entries are
        still-open positions with no exit_ts yet and are NOT included in the spine (a trade needs an
        exit to compute a realized move). This materially undercounts meta_ranker's true signal
        volume and should be revisited once more positions close.
        entry_px_underlying/exit_px_underlying are null for option-routed rows: the source records
        the OPTION premium fill (entry_avg_price/exit_fill_price), not the underlying price, and no
        underlying-price bar join was attempted for Phase 0 to avoid conflating premium and
        underlying in one column. atr_at_entry/tp_price/sl_price are not recorded by this module's
        audit log at all and are null for every row."""
    )
    return _finalize(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Source 4: dealer_ranker (4H per plan spec; source data is session-granular)
# ---------------------------------------------------------------------------

def load_dealer_ranker() -> pd.DataFrame:
    path = REPO_ROOT / "Data/analysis/dealer_ranker_july_exploratory/trade_outcomes.csv"
    raw = pd.read_csv(path)
    total = len(raw)
    canon = raw[
        (raw.policy == "long_all") & (raw.rank_group == "top_10") & (raw.horizon_sessions == 1)
    ].copy()

    df = pd.DataFrame({
        "module": "dealer_ranker",
        "ticker": canon["ticker"],
        "signal_ts": canon["captured_at"],
        "entry_ts": canon["entry_time"],
        "exit_ts": canon["exit_time"],
        "direction": np.where(canon["side"] == "long", 1, -1),
        "entry_px_underlying": canon["entry_price"],
        "exit_px_underlying": canon["exit_price"],
        "exit_reason": "time",
        "bars_held": canon["horizon_sessions"],
        "atr_at_entry": None,
        "tp_price": None,
        "sl_price": None,
        "score": None,
        "cadence": "4h",
        "source_file": _rel(path),
        "provenance": "backtest_insample",
    })

    n_ties = int(
        (pd.to_datetime(canon["entry_time"], utc=True) == pd.to_datetime(canon["exit_time"], utc=True)).sum()
    )
    _note(
        f"""dealer_ranker: {total} raw rows in {_rel(path)} cover EVERY ranked ticker (rank_group in
        top_10 / ranks_11_50 / bottom_10 / ranks_51_to_bottom_decile) at 3 hypothetical exit horizons
        (1/2/3 sessions) x 2 policies (long_all, dealer_directional) -- it is a sensitivity study, not
        a trade ledger. Deduped to the module's actual traded population: policy='long_all'
        (REPORT.md: 'matches the current ranker runner's default call/long behavior';
        dealer_directional is described there as 'a separate, unoptimized diagnostic'),
        rank_group='top_10' (the module ranks and would trade its top 10 names),
        horizon_sessions=1 (shortest / least-speculative exit) -> {len(canon)} canonical rows across
        {canon['snapshot_date'].nunique()} snapshot dates and {canon['ticker'].nunique()} tickers.
        dealer_swing_rank (1-10 ordinal, lower=stronger) exists in the source but is NOT mapped into
        'score': it is an ordinal rank, not a magnitude, and mixing it with other modules'
        higher-is-better probability scores would conflate units -- use dealer_swing_rank directly
        from the raw CSV if needed. exit_reason is set to the literal string 'time' for every row
        (not sourced from a column) because REPORT.md documents these as fixed-horizon exits by
        construction -- a faithful label of the source's known design, not an invented value.
        KNOWN DATA QUALITY ISSUE: for horizon_sessions=1, entry_time and exit_time are numerically
        IDENTICAL in {n_ties}/{len(canon)} rows even though entry_price != exit_price -- the source
        stamps both with the same session-date placeholder (04:00 UTC) rather than true intraday
        open/close clock times. exit_ts is therefore a SESSION LABEL, not a real fill timestamp, for
        this module -- do not read holding period as literally zero from timestamps alone; use
        bars_held (=horizon_sessions) instead.
        cadence is recorded as '4h' per this experiment's module-mapping spec, but the source data
        granularity is actually per-session (daily), not 4h bars -- bars_held holds the source's
        native horizon_sessions unit (trading sessions), not a 4h-bar count.
        provenance is 'backtest_insample' rather than 'backtest_frozen_test': REPORT.md frames this
        explicitly as an exploratory/hypothesis study with no declared train/val/test split or
        held-out protocol (the ranking rule is deterministic off dealer positioning, applied
        retrospectively to real historical dealer snapshots, not fit) -- 'frozen_test' would overstate
        the rigor of the protocol. Sample is intentionally tiny (only ~15 days of dealer-snapshot
        history exist at all, per project memory) and should not be forced to a routing
        recommendation without an explicit power-analysis caveat in Phase 4."""
    )
    return _finalize(df)


# ---------------------------------------------------------------------------
# Source 5: multi_ticker_swing (30m): backtest OOF leg + live real leg
# ---------------------------------------------------------------------------

def load_swing_30m_oof() -> pd.DataFrame:
    frames = []
    for _side, path in [
        ("long", REPO_ROOT / "strategies/multi_ticker_swing/backtest/results_oof/long/trades.parquet"),
        ("short", REPO_ROOT / "strategies/multi_ticker_swing/backtest/results_oof/short/trades.parquet"),
    ]:
        raw = pd.read_parquet(path)
        df = pd.DataFrame({
            "module": "multi_ticker_swing",
            "ticker": raw["ticker"],
            "signal_ts": raw["entry_time"],
            "entry_ts": raw["entry_time"],
            "exit_ts": raw["exit_time"],
            "direction": raw["direction"],
            "entry_px_underlying": raw["entry_price"],
            "exit_px_underlying": raw["exit_price"],
            "exit_reason": raw["exit_reason"],
            "bars_held": None,
            "atr_at_entry": None,
            "tp_price": None,
            "sl_price": None,
            "score": None,
            "cadence": "30m",
            "source_file": _rel(path),
            "provenance": "backtest_frozen_test",
        })
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    _note(
        """multi_ticker_swing (30m backtest leg): strategies/multi_ticker_swing/backtest/results/
        trades.parquet is empty (0 rows), as flagged in the task brief. Hunted results*/ and found
        many exploratory sweep directories (sweep, sweep_1500/600, sweep_v2/v2_clean/v3/v3_shared/
        v4/v4_shared/v5 tier1/tier2, oof_600/1500, 3m_600/1500, compare_current_focus_v3,
        compare_new_focus_v3, results_comp/{clf_lgbm_s43, grid, grid_focused, ranker_xgb_s44,
        ranker_xgb_s44_short}, results_oof/{long, short, grid_long, grid_short, demo_v2}).
        Selected results_oof/long + results_oof/short as canonical: results_oof/README.md explicitly
        documents these as leak-free walk-forward OOF scores ('fold-5 OOF scores come from a model
        trained only on prior data ... unlike the final full-history models these can be honestly
        tested on this window'), test window 2025-08 -> 2026-06.
        results_comp/ranker_xgb_s44/trades.parquet -- the model project memory flags as the promoted
        swing competition winner -- was DELIBERATELY NOT ingested here: per that same results_oof
        README caveat it is a 'full-history' trained model evaluated on the same window, so its
        numbers may leak train-period information; confirmed near-duplicate to results_oof/long
        (9,665 vs 9,569 rows, identical first two trades, identical date range 2025-08-07 ->
        2026-06-04) so ingesting it would not add a distinct signal population, only a
        leakage-risked near-copy. All sweep_*/tier*/oof_600/oof_1500/compare_* directories are
        earlier exploratory parameter sweeps superseded by results_oof and were not ingested.
        bars_held/atr_at_entry/tp_price/sl_price/score are not present in results_oof/{long,short}/
        trades.parquet (columns are only ticker, direction, entry/exit time+price, pnl, exit_reason,
        size) and are null for every row of this leg; signal_ts has no separate column and is set
        equal to entry_ts."""
    )
    return _finalize(out)


def load_swing_30m_live() -> pd.DataFrame:
    path = REPO_ROOT / "Data/analysis/multi_ticker_swing_live/paired_option_trades.csv"
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "module": "multi_ticker_swing",
        "ticker": raw["ticker"],
        "signal_ts": raw["signal_ts"],
        "entry_ts": raw["entry_time"],
        "exit_ts": raw["exit_time"],
        "direction": raw["direction"],
        "entry_px_underlying": raw["entry_price_underlying"],
        "exit_px_underlying": raw["exit_price"],
        "exit_reason": raw["exit_reason"],
        "bars_held": raw["bars_held"],
        "atr_at_entry": raw["atr_at_entry"],
        "tp_price": None,
        "sl_price": raw["sl_price"],
        "score": raw["ev_score"],
        "cadence": "30m",
        "source_file": _rel(path),
        "provenance": "live_real",
    })
    n_missing_underlying = int(raw["entry_price_underlying"].isna().sum())
    n_tickers = int(raw["ticker"].nunique())
    _note(
        f"""multi_ticker_swing (30m live leg): {_rel(path)}, {len(raw)} real closed option round
        trips (FIFO-paired from raw Alpaca fills per Data/analysis/multi_ticker_swing_live/report.md),
        {n_tickers} unique tickers. entry_px_underlying/exit_px_underlying here are the source's
        entry_price_underlying / exit_price columns (verified UNDERLYING price), distinct from the
        source's own entry_price_option / exit_price_option columns (option premium, not used here).
        {n_missing_underlying}/{len(raw)} rows have no audit match (audit_type is
        'broker_position_missing' or null) and so carry null direction / entry_px_underlying /
        signal_ts / atr_at_entry -- these rows still have a real option pnl_dollars fill but no
        underlying-price anchor, and are kept (not dropped) with those fields null. tp_price is null
        for every row: the source has sl_price but no explicit tp_price column (ref_high/ref_low are
        structural reference levels the module also tracks, not a stated profit target, so were not
        repurposed as tp_price). score is populated from the source's own ev_score column (a real EV
        estimate recorded at signal time), not fabricated."""
    )
    return _finalize(df)


# ---------------------------------------------------------------------------
# Source 6: intraday_structure -- no ledger
# ---------------------------------------------------------------------------

def note_intraday_structure() -> None:
    transitions_path = REPO_ROOT / "Data/inference/intraday_structure/transitions.jsonl"
    n_lines = sum(1 for _ in transitions_path.open())
    _note(
        f"""intraday_structure: NO TRADE LEDGER FOUND. {_rel(transitions_path)} ({n_lines} lines) is
        a state-machine transition log (WATCHING/CLOSED/etc. setup lifecycle events such as
        'candidate_ttl_expired'), not a realized-trade ledger with entry/exit fills and PnL.
        Data/inference/intraday_structure/active_signals.json is a live snapshot of currently-open
        candidates, also not a closed-trade record. Consistent with LIVING_SUMMARY.md:
        intraday_structure is paper-only and new and has not yet produced a closed, priced trade
        ledger. Zero rows contributed to the spine for this module; revisit once a real
        fills/closed-trades ledger exists (e.g. mirroring meta_ranker's closed_trades.jsonl
        convention)."""
    )


# ---------------------------------------------------------------------------
# Coverage helpers (bars history, optionable universe)
# ---------------------------------------------------------------------------

def _bars_available_tickers(bars_dir: Path) -> set[str]:
    return {p.stem for p in bars_dir.glob("*.parquet")}


def _load_optionable_universe() -> set[str]:
    """~700-symbol reference set of liquidity-screened, optionable symbols.

    Built from the union of `symbol` across all 16 days of real Schwab
    dealer-positioning chain snapshots. shared_universe.csv was also
    inspected but carries no options-specific eligibility flag, so it is
    not used as the G0 reference (documented in the inventory).
    """
    syms: set[str] = set()
    for d in sorted(DEALER_SNAPSHOT_DIR.glob("*")):
        f = d / "dealer_level_summary.parquet"
        if f.exists():
            syms |= set(pd.read_parquet(f, columns=["symbol"])["symbol"].dropna().unique())
    return syms


# ---------------------------------------------------------------------------
# Inventory report
# ---------------------------------------------------------------------------

def _fmt_days(ts_min, ts_max) -> str:
    if pd.isna(ts_min) or pd.isna(ts_max):
        return "n/a"
    return f"{ts_min.date()} -> {ts_max.date()}"


def _quantiles(s: pd.Series) -> tuple[float, float, float]:
    s = s.dropna()
    if s.empty:
        return (float("nan"),) * 3
    return (s.quantile(0.25), s.quantile(0.5), s.quantile(0.75))


def _module_section(module: str, df: pd.DataFrame, optionable: set[str],
                     bars_4h: set[str], bars_1d: set[str]) -> str:
    lines = [f"## {module}", ""]
    n = len(df)
    lines.append(f"- n rows: {n}")
    if n == 0:
        lines.append("- (no rows -- see notes)")
        lines.append("")
        return "\n".join(lines)

    date_min = df["entry_ts"].min()
    date_max = df["entry_ts"].max()
    lines.append(f"- date range (entry_ts): {_fmt_days(date_min, date_max)}")
    lines.append(f"- unique tickers: {df['ticker'].nunique()}")

    prov = df["provenance"].value_counts(dropna=False)
    lines.append(f"- provenance breakdown: {prov.to_dict()}")

    cadence = df["cadence"].value_counts(dropna=False)
    lines.append(f"- cadence breakdown: {cadence.to_dict()}")

    # holding period
    cal_days = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds() / 86400.0
    p25, p50, p75 = _quantiles(cal_days)
    lines.append(f"- holding period, calendar days (p25/median/p75): {p25:.2f} / {p50:.2f} / {p75:.2f}"
                 if not pd.isna(p50) else "- holding period, calendar days: n/a")
    bp25, bp50, bp75 = _quantiles(df["bars_held"])
    if not pd.isna(bp50):
        lines.append(f"- holding period, bars_held native units (p25/median/p75): {bp25:.2f} / {bp50:.2f} / {bp75:.2f} "
                     f"(n={df['bars_held'].notna().sum()}/{n} rows have bars_held)")
    else:
        lines.append("- holding period, bars_held: not available for any row of this module (see notes)")

    # realized underlying move
    have_px = df["entry_px_underlying"].notna() & df["exit_px_underlying"].notna() & df["direction"].notna()
    ret_pct = (df.loc[have_px, "exit_px_underlying"] / df.loc[have_px, "entry_px_underlying"] - 1.0) \
        * df.loc[have_px, "direction"].astype(float)
    rp25, rp50, rp75 = _quantiles(ret_pct)
    lines.append(
        f"- realized underlying move, signed % (p25/median/p75), n={have_px.sum()}/{n}: "
        + (f"{rp25:.4f} / {rp50:.4f} / {rp75:.4f}" if not pd.isna(rp50) else "n/a")
    )
    have_atr = have_px & df["atr_at_entry"].notna() & (df["atr_at_entry"] != 0)
    atr_move = (
        (df.loc[have_atr, "exit_px_underlying"] - df.loc[have_atr, "entry_px_underlying"])
        * df.loc[have_atr, "direction"].astype(float) / df.loc[have_atr, "atr_at_entry"]
    )
    ap25, ap50, ap75 = _quantiles(atr_move)
    lines.append(
        f"- realized underlying move, ATR units (p25/median/p75), n={have_atr.sum()}/{n}: "
        + (f"{ap25:.4f} / {ap50:.4f} / {ap75:.4f}" if not pd.isna(ap50) else "n/a")
    )

    win_rate = float((ret_pct > 0).mean()) if have_px.any() else float("nan")
    lines.append(
        f"- win rate (underlying move > 0, n={have_px.sum()}): "
        + (f"{win_rate:.3f}" if not pd.isna(win_rate) else "n/a")
    )

    exit_mix = df["exit_reason"].value_counts(dropna=False)
    lines.append(f"- exit-reason mix: {exit_mix.to_dict()}")

    n_pre_2024_02 = int((df["entry_ts"] < OPTION_HISTORY_START).sum())
    lines.append(f"- trades with entry_ts before 2024-02-01 (unroutable, pre-Alpaca-option-history): {n_pre_2024_02}/{n}")

    tickers = df["ticker"].dropna().unique()
    no_4h = sorted(set(tickers) - bars_4h)
    no_1d = sorted(set(tickers) - bars_1d)
    rows_no_4h = int(df["ticker"].isin(no_4h).sum())
    rows_no_1d = int(df["ticker"].isin(no_1d).sum())
    lines.append(
        f"- tickers lacking Data/shared/bars/4h history: {len(no_4h)}/{len(tickers)} unique "
        f"({rows_no_4h}/{n} rows)"
    )
    lines.append(
        f"- tickers lacking Data/shared/bars/1d history: {len(no_1d)}/{len(tickers)} unique "
        f"({rows_no_1d}/{n} rows)"
    )
    if no_4h:
        sample = ", ".join(no_4h[:15]) + (", ..." if len(no_4h) > 15 else "")
        lines.append(f"  - sample missing-4h tickers: {sample}")

    # Gate G0
    n_optionable = int(df["ticker"].isin(optionable).sum())
    g0_share = n_optionable / n if n else float("nan")
    lines.append(
        f"- **Gate G0** (ticker in the ~{len(optionable)}-symbol optionable reference set): "
        f"{n_optionable}/{n} = {g0_share:.1%} {'PASS (>=80%)' if g0_share >= 0.8 else 'FAIL (<80%) -> shares-only by data availability'}"
    )

    null_cols = [c for c in SPINE_COLUMNS if df[c].isna().all()]
    lines.append(f"- columns null for EVERY row of this module: {null_cols if null_cols else 'none'}")

    lines.append("")
    return "\n".join(lines)


def build_inventory(spine: pd.DataFrame, optionable: set[str], bars_4h: set[str], bars_1d: set[str]) -> str:
    lines = [
        "# Options Experiment Spine -- Inventory (Phase 0)",
        "",
        f"Generated by `scripts/build_options_experiment_spine.py`. "
        f"Total spine rows: {len(spine)}. "
        f"Modules: {sorted(spine['module'].unique().tolist())}.",
        "",
        "Plan: docs/superpowers/plans/2026-07-25-options-instrument-routing-experiment.md "
        "(Phase 0 -- Experiment spine section).",
        "",
        "## Gate G0 summary",
        "",
        "Reference optionable universe: union of `symbol` across all "
        f"{len(sorted(DEALER_SNAPSHOT_DIR.glob('*')))} days of real Schwab dealer-positioning chain "
        f"snapshots under Data/dealer_positioning/historical_snapshots/*/dealer_level_summary.parquet "
        f"({len(optionable)} unique symbols). Data/shared/universe/shared_universe.csv was inspected "
        "and does NOT carry an options-specific eligibility flag, so it was not used as the primary "
        "G0 reference (kept as a documented alternative if a future phase wants a broader, "
        "less liquidity-screened universe).",
        "",
    ]

    g0_rows = []
    for module, df in spine.groupby("module"):
        n = len(df)
        n_opt = int(df["ticker"].isin(optionable).sum())
        share = n_opt / n if n else float("nan")
        g0_rows.append((module, n, n_opt, share))
    lines.append("| module | n rows | optionable rows | share | gate |")
    lines.append("|---|---:|---:|---:|---|")
    for module, n, n_opt, share in g0_rows:
        gate = "PASS" if share >= 0.8 else "FAIL (shares-only by data availability)"
        lines.append(f"| {module} | {n} | {n_opt} | {share:.1%} | {gate} |")
    lines.append("")

    lines.append("## Per-module detail")
    lines.append("")
    for module in sorted(spine["module"].unique()):
        lines.append(_module_section(module, spine[spine["module"] == module], optionable, bars_4h, bars_1d))

    lines.append("intraday_structure")
    lines.append("")
    lines.append("- n rows: 0 -- no closed-trade ledger exists for this module yet (see notes).")
    lines.append("")

    lines.append("## Source-by-source notes, decisions, and known gaps")
    lines.append("")
    for i, n in enumerate(NOTES, start=1):
        lines.append(f"{i}. {n}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_spine() -> pd.DataFrame:
    check_ev_experiments_final_subset()
    note_intraday_structure()

    frames = [
        load_momentum_htf_4h(),
        load_meta_ranker(),
        load_dealer_ranker(),
        load_swing_30m_oof(),
        load_swing_30m_live(),
    ]
    spine = pd.concat(frames, ignore_index=True)
    spine = spine.reindex(columns=SPINE_COLUMNS)
    spine = spine.sort_values(["module", "entry_ts"], na_position="last").reset_index(drop=True)
    return spine


def validate_spine(spine: pd.DataFrame) -> None:
    """Fail-fast structural checks (also exercised by the pytest suite)."""
    missing_cols = set(SPINE_COLUMNS) - set(spine.columns)
    if missing_cols:
        raise AssertionError(f"spine is missing required columns: {missing_cols}")

    for c in ["signal_ts", "entry_ts", "exit_ts"]:
        non_null = spine[c].dropna()
        if len(non_null) and not isinstance(non_null.dtype, pd.DatetimeTZDtype):
            raise AssertionError(f"column {c} is not tz-aware")

    bad_dir = spine["direction"].dropna()
    if not bad_dir.isin([1, -1]).all():
        raise AssertionError("direction column contains values outside {-1, +1}")

    bad_prov = ~spine["provenance"].isin(VALID_PROVENANCE)
    if bad_prov.any():
        raise AssertionError(f"provenance contains values outside {VALID_PROVENANCE}: "
                              f"{spine.loc[bad_prov, 'provenance'].unique()}")

    bad_cadence = ~spine["cadence"].isin(VALID_CADENCE)
    if bad_cadence.any():
        raise AssertionError(f"cadence contains values outside {VALID_CADENCE}: "
                              f"{spine.loc[bad_cadence, 'cadence'].unique()}")

    both = spine["entry_ts"].notna() & spine["exit_ts"].notna()
    # dealer_ranker's horizon_sessions=1 rows legitimately stamp entry_ts == exit_ts
    # (documented session-label limitation) -- require exit >= entry, not strictly >.
    if (spine.loc[both, "exit_ts"] < spine.loc[both, "entry_ts"]).any():
        raise AssertionError("found rows where exit_ts < entry_ts")


def main() -> None:
    spine = build_spine()
    validate_spine(spine)

    optionable = _load_optionable_universe()
    bars_4h = _bars_available_tickers(BARS_4H_DIR)
    bars_1d = _bars_available_tickers(BARS_1D_DIR)

    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    spine.to_parquet(DATA_OUT, index=False)

    inventory = build_inventory(spine, optionable, bars_4h, bars_1d)
    INVENTORY_OUT.write_text(inventory)

    print(f"wrote {len(spine)} rows -> {_rel(DATA_OUT)}")
    print(f"wrote inventory -> {_rel(INVENTORY_OUT)}")
    print(spine.groupby(["module", "provenance"]).size())


if __name__ == "__main__":
    main()
