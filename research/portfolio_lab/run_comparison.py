"""WS-D real-data comparison: fixed-$5,000 vs. equal-weight-topN vs.
covariance/vol-targeted sizing, on the SAME signal stream, date range, and
execution assumptions.

Signal stream: reuses the momentum_expansion family_backtest's existing
cached test-window scores for its documented overall-best family/seed
(lgbm_classifier seed 43, `strategies/momentum_expansion/backtest/results/
family_compare/comparison_summary.json` -> `clf_vs_ranker.winner ==
"classifier"`; `best_by_family` -> lgbm_classifier ranks first by
ret_over_dd). Does NOT invent a signal: this is the same scored parquet
`run_family_compare.py` already wrote to
`.../family_compare/scores/lgbm_classifier_s43.parquet` on the held-out test
window (2025-05-20 .. 2026-05-14).

Order policy (top_k / TP-SL-hold): uses the sweep's best top_k=10 row
(tp=5.0 ATR, sl=5.0 ATR, hold=75 bars) from `sweep_results.csv` -- the same
values as the strategy's own documented overall-best policy, just at
top_k=10 (the task's "top-10" comparison point) instead of the sweep-winning
top_k=20.

--- Reference capital ---

``--capital`` defaults to the REAL paper-account equity, not a round $100k.
Source: ``Data/inference/account_snapshots/broker_equity_20260724_paper.jsonl``,
last record, ``captured_at_utc: 2026-07-25T00:05:00Z``: ``equity=926111.39``,
122 open positions, gross market value ``$618,332.93`` (0.67x equity -- the
live book is NOT leveraged; an earlier draft of this comparison used a round
$100k reference capital, which is what originally produced the incorrect
"policy (a) implies ~4x leverage" claim -- see LIVING_SUMMARY.md correction).
Policy (a) (fixed $5,000/entry) is capital-INDEPENDENT: its dollar position
sizes, dollar P&L, and Sharpe ratio do not change with ``--capital`` at all
(Sharpe is scale-invariant under a constant capital denominator). Its
*percentage* metrics (return %, max-DD %) DO mechanically shrink as
``--capital`` grows, since the same fixed-dollar drawdown gets divided by a
bigger number -- this is a denominator artifact, not a change in risk, and
is called out explicitly in the printed report.

--- Slot count ---

``--target-concurrent`` (default 10) is the slot cap used for policies
(b)/(c) in the "slot-capped" table. Capping (b)/(c) at 10 concurrent names
while policy (a) runs UNCONSTRAINED (it averaged ~81 concurrent positions
across five modules' worth of live book) means the slot-capped table's trade
counts differ by an order of magnitude (1,217 vs ~134) -- that is an
ADMISSION/turnover effect, not a sizing effect, and conflating the two would
be exactly the "wins by trading less" trap the task's hard constraints warn
against. This script therefore ALSO runs a second, "slot-matched" table
where (b)/(c) use ``target_concurrent`` rounded to policy (a)'s own realized
average concurrency, so the sizing comparison (equal-weight vs. covariance/
vol-target) is evaluated on a comparable number of concurrently-held names,
isolating the sizing question from the slot-count question.

Usage:
  .venv/bin/python -m research.portfolio_lab.run_comparison
  .venv/bin/python -m research.portfolio_lab.run_comparison --capital 926111.39 --target-concurrent 10
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from strategies.momentum_expansion.backtest import family_backtest as fb

from research.portfolio_lab import portfolio_backtest as pb
from research.portfolio_lab import sizing as sz
from research.portfolio_lab.daily_context import DailyContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("portfolio_lab.run_comparison")

ROOT = Path(__file__).resolve().parents[2]
SCORES_PATH = ROOT / "strategies/momentum_expansion/backtest/results/family_compare/scores/lgbm_classifier_s43.parquet"
THEME_MAP_PATH = ROOT / "themes/dynamic_theme/outputs/ticker_theme_assignments.csv"
BROKER_SNAPSHOT_PATH = ROOT / "Data/inference/account_snapshots/broker_equity_20260724_paper.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

TOP_K = 10
TP_MULT = 5.0
SL_MULT = 5.0
MAX_HOLD = 75
# Real paper-account equity, last record in BROKER_SNAPSHOT_PATH
# (captured_at_utc=2026-07-25T00:05:00Z). See module docstring "Reference
# capital" for full provenance and the leverage-claim correction.
REAL_ACCOUNT_EQUITY = 926_111.39
DEFAULT_TARGET_CONCURRENT = 10
FIXED_NOTIONAL = 5_000.0            # matches core.live_4h_exec.ExecPolicy.target_notional default
COST_BPS_ROUND_TRIP = 10.0          # 5bps each way -- research assumption, applied identically to all policies


def _load_theme_map() -> dict:
    if not THEME_MAP_PATH.exists():
        logger.warning("theme map not found at %s -- theme caps will be inert", THEME_MAP_PATH)
        return {}
    df = pd.read_csv(THEME_MAP_PATH, usecols=["ticker", "theme_1"])
    return dict(zip(df["ticker"].astype(str), df["theme_1"].astype(str)))


def _cov_sizing_cfg(capital: float, target_concurrent: int) -> sz.SizingConfig:
    return sz.SizingConfig(
        capital=capital, target_concurrent=target_concurrent,
        per_position_cap_pct=0.12, per_sector_cap_pct=0.30, per_theme_cap_pct=0.25,
        target_portfolio_vol_annual=0.15, max_vol_scale=3.0,
        adv_participation_cap_pct=0.05, corr_penalty_strength=1.0,
    )


def _run_one(name, policy, candidates, daily_ctx, sector_map, theme_map, capital, cost_bps):
    logger.info("running policy: %s (target_concurrent=%s)", name, policy.target_concurrent)
    trades, snapshots = pb.run_policy(
        candidates, policy=policy, daily_ctx=daily_ctx,
        sector_map=sector_map, theme_map=theme_map, cost_bps_round_trip=cost_bps,
    )
    metrics = pb.compute_metrics(trades, snapshots, capital=capital, n_candidates_seen=len(candidates))
    logger.info("[%s] trades=%s sharpe_daily=%s max_dd_pct=%s turnover_x=%s avg_concurrent=%s "
                "avg_pos_pct_adv=%s", name, metrics.get("trades"), metrics.get("sharpe_daily"),
                metrics.get("max_dd_pct"), metrics.get("turnover_annualized_x"),
                metrics.get("avg_concurrent_positions"), metrics.get("avg_position_pct_of_adv"))
    return metrics, trades


def run(
    out_dir: Path = RESULTS_DIR, *, capital: float = REAL_ACCOUNT_EQUITY,
    target_concurrent: int = DEFAULT_TARGET_CONCURRENT, cost_bps: float = COST_BPS_ROUND_TRIP,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading scored test-window signal stream: %s", SCORES_PATH)
    scored = pd.read_parquet(SCORES_PATH, columns=["timestamp", "ticker", "score"])
    signals = fb.select_signals(scored, TOP_K, allow_short=False)
    logger.info("top_k=%d signal rows=%d unique tickers=%d unique timestamps=%d",
                TOP_K, len(signals), signals["ticker"].nunique(), signals["timestamp"].nunique())

    bar_cache = fb.BarCache()
    candidates = pb._resolve_candidates(signals, bar_cache, tp_mult=TP_MULT, sl_mult=SL_MULT, max_hold=MAX_HOLD)
    logger.info("resolved candidates=%d (of %d signal rows; some drop for missing/short bar history)",
                len(candidates), len(signals))

    tickers = sorted(candidates["ticker"].unique().tolist())
    daily_ctx = DailyContext(tickers=tickers)
    theme_map = _load_theme_map()
    theme_hit = sum(1 for t in tickers if t in theme_map)
    logger.info("theme map coverage: %d/%d tickers resolved", theme_hit, len(tickers))
    # Sector map intentionally None throughout this run: plan defect D1 documents 100%
    # "Unknown" sector metadata; the empirical sector-ETF resolver is WS-A's deliverable
    # and out of scope for WS-D. The per-sector cap mechanism is real and unit-tested
    # (tests/test_portfolio_backtest.py::test_sector_cap_blocks_over_concentrated_admission)
    # but is INERT in this real-data run.
    sector_map: dict = {}

    # --- (a) fixed $5,000/entry: capital-independent, run once, reused in both tables ---
    policy_a = pb.PolicyConfig(name="a_fixed_5000", fixed_notional=FIXED_NOTIONAL, target_concurrent=None)
    metrics_a, trades_a = _run_one("a_fixed_5000", policy_a, candidates, daily_ctx, sector_map, theme_map,
                                   capital, cost_bps)
    baseline_avg_concurrent = metrics_a["avg_concurrent_positions"]
    matched_slots = max(1, round(baseline_avg_concurrent))
    logger.info("baseline (a) realized avg_concurrent_positions=%.2f -> slot-matched target_concurrent=%d",
                baseline_avg_concurrent, matched_slots)

    tables = {}
    all_trades = {"a_fixed_5000": trades_a}
    for label, slots in [
        ("slot_capped", target_concurrent),
        ("slot_matched_to_baseline_concurrency", matched_slots),
    ]:
        policy_b = pb.PolicyConfig(
            name=f"b_equal_weight_top{slots}", equal_weight=True, target_concurrent=slots,
            sizing_cfg=sz.SizingConfig(capital=capital, target_concurrent=slots),
        )
        policy_c = pb.PolicyConfig(
            name=f"c_cov_vol_target_slots{slots}", sizing_cfg=_cov_sizing_cfg(capital, slots),
            target_concurrent=slots,
        )
        metrics_b, trades_b = _run_one(policy_b.name, policy_b, candidates, daily_ctx, sector_map, theme_map,
                                       capital, cost_bps)
        metrics_c, trades_c = _run_one(policy_c.name, policy_c, candidates, daily_ctx, sector_map, theme_map,
                                       capital, cost_bps)
        tables[label] = {
            "target_concurrent": slots,
            "metrics": {"a_fixed_5000": metrics_a, policy_b.name: metrics_b, policy_c.name: metrics_c},
        }
        all_trades[policy_b.name] = trades_b
        all_trades[policy_c.name] = trades_c

    summary = {
        "signal_source": str(SCORES_PATH.relative_to(ROOT)),
        "family_seed": "lgbm_classifier_s43",
        "top_k": TOP_K, "tp_atr_mult": TP_MULT, "sl_atr_mult": SL_MULT, "max_hold": MAX_HOLD,
        "capital": capital,
        "capital_source": (
            f"{BROKER_SNAPSHOT_PATH.relative_to(ROOT)} last record, "
            "captured_at_utc=2026-07-25T00:05:00Z, equity=926111.39, 122 open positions, "
            "gross_market_value=618332.93 (0.67x equity, not leveraged)"
            if abs(capital - REAL_ACCOUNT_EQUITY) < 1e-6 else "user-supplied --capital"
        ),
        "cost_bps_round_trip": cost_bps,
        "candidates_seen": len(candidates),
        "date_range": [str(candidates["signal_ts"].min()), str(candidates["signal_ts"].max())],
        "sector_map_active": False,
        "theme_map_coverage": f"{theme_hit}/{len(tickers)}",
        "policy_a_note": (
            "fixed_notional sizing is capital-independent: dollar P&L and Sharpe are unchanged "
            "by --capital; only pct-of-capital metrics (return %, max_dd_pct) shrink as capital "
            "grows -- a denominator artifact, not a change in risk."
        ),
        "slot_matching_note": (
            f"baseline (a) realized avg {baseline_avg_concurrent:.2f} concurrent positions "
            f"(unconstrained). 'slot_capped' caps (b)/(c) at {target_concurrent} concurrent names "
            f"(an admission/turnover effect vs. (a), NOT isolating sizing). "
            f"'slot_matched_to_baseline_concurrency' caps (b)/(c) at {matched_slots} concurrent "
            "names to match (a)'s realized concurrency, isolating the sizing-only comparison "
            "between (b) and (c)."
        ),
        "tables": tables,
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    for label, table in tables.items():
        pd.DataFrame(table["metrics"]).T.to_csv(out_dir / f"comparison_metrics_{label}.csv")
    for name, trades in all_trades.items():
        pd.DataFrame([vars(t) for t in trades]).to_parquet(out_dir / f"trades_{name}.parquet", index=False)
    logger.info("wrote results to %s", out_dir)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--capital", type=float, default=REAL_ACCOUNT_EQUITY,
                   help="Reference capital for pct-of-capital metrics and (b)/(c) sizing. "
                        f"Defaults to the real paper-account equity ({REAL_ACCOUNT_EQUITY:,.2f}), "
                        "see module docstring for provenance.")
    p.add_argument("--target-concurrent", type=int, default=DEFAULT_TARGET_CONCURRENT,
                   help="Slot cap for the 'slot_capped' table. A second 'slot_matched' table "
                        "is always also produced, using (a)'s realized average concurrency.")
    p.add_argument("--cost-bps", type=float, default=COST_BPS_ROUND_TRIP)
    args = p.parse_args()
    run(args.out_dir, capital=args.capital, target_concurrent=args.target_concurrent, cost_bps=args.cost_bps)
