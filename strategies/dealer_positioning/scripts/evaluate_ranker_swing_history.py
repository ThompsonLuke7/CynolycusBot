"""Exploratory, timestamp-safe evaluation of dealer-ranker swing selection.

This study deliberately tests the ranking only after its recorded ``captured_at``
time. Each candidate enters at the next available regular-session open and
exits at the close after a fixed number of sessions. It evaluates underlying
daily OHLCV returns, not option-contract P&L, and is not an Intraday Structure replay.

The short July history is suitable for feasibility evidence and case studies,
not a claim of a durable trading edge.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


REPO = Path(__file__).resolve().parents[3]
DEFAULT_HISTORY = REPO / "Data/dealer_positioning/rankings/dealer_swing_rankings_history.parquet"
DEFAULT_BARS = REPO / "Data/shared/bars/1d"
DEFAULT_OUTPUT = REPO / "Data/analysis/dealer_ranker_july_exploratory"
ET = ZoneInfo("America/New_York")


def _load_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["timestamp", "open", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "close"])
    frame = frame[(frame["open"] > 0) & (frame["close"] > 0)].sort_values("timestamp")
    frame["session_date"] = frame["timestamp"].dt.tz_convert(ET).dt.date
    session = frame.groupby("session_date", as_index=False).agg(
        entry_time=("timestamp", "first"),
        entry_price=("open", "first"),
        exit_time=("timestamp", "last"),
        exit_price=("close", "last"),
    )
    return session.sort_values("session_date").reset_index(drop=True)


def next_session_trade(bars: pd.DataFrame, captured_at: object, horizon_sessions: int) -> dict | None:
    """Enter the next distinct ET session after capture; avoid same-day leakage."""
    if bars.empty or horizon_sessions < 1:
        return None
    capture = pd.Timestamp(captured_at)
    if capture.tzinfo is None:
        capture = capture.tz_localize("UTC")
    capture_date = capture.tz_convert(ET).date()
    start = int(bars["session_date"].searchsorted(capture_date, side="right"))
    exit_index = start + horizon_sessions - 1
    if exit_index >= len(bars):
        return None
    entry = bars.iloc[start]
    exit_ = bars.iloc[exit_index]
    return {
        "entry_session": entry["session_date"].isoformat(),
        "entry_time": entry["entry_time"].isoformat(),
        "entry_price": float(entry["entry_price"]),
        "exit_session": exit_["session_date"].isoformat(),
        "exit_time": exit_["exit_time"].isoformat(),
        "exit_price": float(exit_["exit_price"]),
    }


def _rank_group(rank: int, max_rank: int, top_k: int) -> str:
    if rank <= top_k:
        return f"top_{top_k}"
    if rank <= 50:
        return "ranks_11_50"
    if rank > max_rank - top_k:
        return f"bottom_{top_k}"
    return "ranks_51_to_bottom_decile"


def _policy_rows(row: pd.Series) -> list[tuple[str, float]]:
    policies = [("long_all", 1.0)]
    direction = str(row.get("dealer_direction", "")).lower()
    if direction == "bullish":
        policies.append(("dealer_directional", 1.0))
    elif direction == "bearish":
        policies.append(("dealer_directional", -1.0))
    return policies


def run_study(
    rankings: pd.DataFrame,
    *,
    bars_root: Path,
    horizons: tuple[int, ...] = (1, 2, 3),
    top_k: int = 10,
    round_trip_cost_bps: float = 12.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"snapshot_date", "captured_at", "symbol", "dealer_swing_rank"}
    missing = required - set(rankings.columns)
    if missing:
        raise KeyError(f"ranking history missing columns: {sorted(missing)}")
    frame = rankings.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True, errors="coerce")
    frame["dealer_swing_rank"] = pd.to_numeric(frame["dealer_swing_rank"], errors="coerce")
    frame = frame.dropna(subset=["captured_at", "dealer_swing_rank"])

    @lru_cache(maxsize=None)
    def bars_for(symbol: str) -> pd.DataFrame | None:
        path = bars_root / f"{symbol}.parquet"
        return _load_bars(path) if path.exists() else None

    spy = bars_for("SPY")
    if spy is None or spy.empty:
        raise FileNotFoundError(f"SPY daily bars are required under {bars_root}")
    rows: list[dict] = []
    for snapshot_date, group in frame.groupby("snapshot_date", sort=True):
        max_rank = int(group["dealer_swing_rank"].max())
        for record in group.itertuples(index=False):
            symbol = str(record.symbol).upper()
            bars = bars_for(symbol)
            if bars is None or bars.empty:
                continue
            source = record._asdict()
            for horizon in horizons:
                trade = next_session_trade(bars, source["captured_at"], horizon)
                benchmark = next_session_trade(spy, source["captured_at"], horizon)
                if trade is None or benchmark is None:
                    continue
                if trade["entry_session"] != benchmark["entry_session"] or trade["exit_session"] != benchmark["exit_session"]:
                    continue
                raw_return = trade["exit_price"] / trade["entry_price"] - 1.0
                spy_return = benchmark["exit_price"] / benchmark["entry_price"] - 1.0
                for policy, side in _policy_rows(pd.Series(source)):
                    gross_return = side * raw_return
                    net_return = gross_return - round_trip_cost_bps / 10_000.0
                    rows.append({
                        "snapshot_date": str(snapshot_date),
                        "captured_at": source["captured_at"].isoformat(),
                        "ticker": symbol,
                        "dealer_swing_rank": int(source["dealer_swing_rank"]),
                        "rank_group": _rank_group(int(source["dealer_swing_rank"]), max_rank, top_k),
                        "policy": policy,
                        "side": "long" if side > 0 else "short",
                        "horizon_sessions": horizon,
                        "raw_underlying_return": raw_return,
                        "gross_return": gross_return,
                        "net_return": net_return,
                        "spy_return": spy_return,
                        "excess_return": gross_return - spy_return,
                        **trade,
                    })
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, pd.DataFrame(), pd.DataFrame()
    portfolios = _portfolio_rows(trades, top_k)
    summary = _summary_rows(portfolios)
    return trades, portfolios, summary


def _portfolio_rows(trades: pd.DataFrame, top_k: int) -> pd.DataFrame:
    sets = {
        "all_ranked": pd.Series(True, index=trades.index),
        f"top_{top_k}": trades["dealer_swing_rank"] <= top_k,
        "ranks_11_50": (trades["dealer_swing_rank"] > top_k) & (trades["dealer_swing_rank"] <= 50),
        f"bottom_{top_k}": trades["rank_group"].eq(f"bottom_{top_k}"),
    }
    rows: list[dict] = []
    for policy in sorted(trades["policy"].unique()):
        for horizon in sorted(trades["horizon_sessions"].unique()):
            base = trades[(trades["policy"] == policy) & (trades["horizon_sessions"] == horizon)]
            for bucket, mask in sets.items():
                subset = base.loc[mask.reindex(base.index, fill_value=False)]
                for snapshot_date, group in subset.groupby("snapshot_date", sort=True):
                    rows.append({
                        "snapshot_date": snapshot_date,
                        "policy": policy,
                        "horizon_sessions": horizon,
                        "bucket": bucket,
                        "names": int(len(group)),
                        "mean_net_return": float(group["net_return"].mean()),
                        "median_net_return": float(group["net_return"].median()),
                        "win_rate": float((group["net_return"] > 0).mean()),
                        "mean_excess_return": float(group["excess_return"].mean()),
                    })
    return pd.DataFrame(rows)


def _summary_rows(portfolios: pd.DataFrame) -> pd.DataFrame:
    if portfolios.empty:
        return pd.DataFrame()
    grouped = portfolios.groupby(["policy", "horizon_sessions", "bucket"], as_index=False).agg(
        snapshots=("snapshot_date", "nunique"),
        average_names=("names", "mean"),
        mean_net_return=("mean_net_return", "mean"),
        median_net_return=("mean_net_return", "median"),
        positive_snapshot_rate=("mean_net_return", lambda x: float((x > 0).mean())),
        mean_excess_return=("mean_excess_return", "mean"),
    )
    return grouped.sort_values(["policy", "horizon_sessions", "bucket"]).reset_index(drop=True)


def write_report(
    output_dir: Path,
    *,
    trades: pd.DataFrame,
    portfolios: pd.DataFrame,
    summary: pd.DataFrame,
    cost_bps: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_dir / "trade_outcomes.csv", index=False)
    portfolios.to_csv(output_dir / "snapshot_portfolios.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    lines = [
        "# July Dealer Ranker Exploratory Study",
        "",
        "This is a fixed-hypothesis, underlying-price study of a short dealer-ranking history.",
        "Each ranking is actionable only after its recorded `captured_at`; entry is the next distinct regular-session open and exits are fixed 1/2/3-session closes.",
        f"A {cost_bps:.1f} bps round-trip underlying cost is included. It is not option P&L and not an Intraday Structure 1-minute replay.",
        "",
        f"Rows evaluated: {len(trades):,}.  Snapshot portfolios: {len(portfolios):,}.",
        "",
        "## Portfolio summary",
        "",
        "```text\n" + summary.to_string(index=False, float_format=lambda value: f"{value:.4f}") + "\n```" if not summary.empty else "No complete outcomes were available.",
        "",
        "## Interpretation limits",
        "",
        "- The number of independent ranking dates is small; this can identify a lead, not establish an edge.",
        "- The ranker is assessed after publication, so this does not claim it predicted intraday moves before its snapshot was captured.",
        "- `long_all` matches the current ranker runner's default call/long behavior. `dealer_directional` is a separate, unoptimized diagnostic that excludes neutral names.",
        "- Broad one-minute bars and pre-pivot dealer snapshots are still required to validate the Intraday Structure Engine and MU-style V-pivot entries.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory dealer-ranker swing evaluation using daily OHLCV bars.")
    parser.add_argument("--rankings", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--bars-root", type=Path, default=DEFAULT_BARS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--round-trip-cost-bps", type=float, default=12.0)
    args = parser.parse_args()
    rankings = pd.read_parquet(args.rankings)
    trades, portfolios, summary = run_study(
        rankings,
        bars_root=args.bars_root,
        horizons=tuple(sorted(set(args.horizons))),
        top_k=max(1, int(args.top_k)),
        round_trip_cost_bps=max(0.0, float(args.round_trip_cost_bps)),
    )
    write_report(args.output, trades=trades, portfolios=portfolios, summary=summary, cost_bps=float(args.round_trip_cost_bps))
    print(summary.to_string(index=False) if not summary.empty else "No complete outcomes available.")


if __name__ == "__main__":
    main()
