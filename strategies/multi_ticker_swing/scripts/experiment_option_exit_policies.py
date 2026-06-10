from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.multi_ticker_swing.scripts.analyze_live_trade_fills import load_audit


BASE = Path("Data/analysis/multi_ticker_swing_live")
MULT = 100.0


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(BASE / "paired_option_trades.csv")
    opens, _closes, bars, _signals = load_audit(Path("UI/swing_audit"))
    if bars.empty:
        return trades, pd.DataFrame()
    bars = bars.dropna(subset=["option_symbol"]).copy()
    bars["audit_ts"] = pd.to_datetime(bars["audit_ts"], utc=True, errors="coerce")
    bars = _num(
        bars,
        [
            "option_last_price",
            "option_best_price",
            "pnl_pct_underlying_mark",
            "underlying_close",
            "underlying_high",
            "underlying_low",
        ],
    )
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    trades = _num(
        trades,
        [
            "entry_price_option",
            "exit_price_option",
            "pnl_dollars",
            "qty",
            "direction",
            "exit_pnl_pct",
            "p_dir",
            "ev_score",
            "dte_at_entry",
        ],
    )
    return trades, bars


def _actual_result(trade: pd.Series) -> tuple[float, str, pd.Timestamp | None]:
    pnl = float(trade["pnl_dollars"])
    reason = str(trade.get("exit_reason") or "actual_exit")
    return pnl, reason, trade.get("exit_time")


def _simulate_trade(
    trade: pd.Series,
    path: pd.DataFrame,
    *,
    stop_loss_pct: float | None,
    no_progress_minutes: int | None,
    no_progress_mfe_pct: float | None,
    trail_arm_pct: float | None,
    trail_giveback_pct: float | None,
    take_profit_pct: float | None,
) -> tuple[float, str, pd.Timestamp | None]:
    entry = float(trade["entry_price_option"])
    qty = float(trade.get("qty") or 1.0)
    if not np.isfinite(entry) or entry <= 0:
        return _actual_result(trade)
    best = entry
    trail_armed = False
    checked_np = False
    for row in path.itertuples(index=False):
        price = float(row.option_last_price)
        if not np.isfinite(price) or price <= 0:
            continue
        ts = row.audit_ts
        best = max(best, price)
        pct = price / entry - 1.0
        mfe = best / entry - 1.0
        held_min = (ts - trade["entry_time"]).total_seconds() / 60.0
        if stop_loss_pct is not None and pct <= -float(stop_loss_pct):
            return (price - entry) * qty * MULT, f"stop_{stop_loss_pct:.2f}", ts
        if take_profit_pct is not None and pct >= float(take_profit_pct):
            return (price - entry) * qty * MULT, f"tp_{take_profit_pct:.2f}", ts
        if (
            no_progress_minutes is not None
            and no_progress_mfe_pct is not None
            and not checked_np
            and held_min >= int(no_progress_minutes)
        ):
            checked_np = True
            if mfe < float(no_progress_mfe_pct):
                return (price - entry) * qty * MULT, (
                    f"no_progress_{no_progress_minutes}m_{no_progress_mfe_pct:.2f}"
                ), ts
        if trail_arm_pct is not None and trail_giveback_pct is not None:
            if mfe >= float(trail_arm_pct):
                trail_armed = True
            if trail_armed:
                floor_profit = (best - entry) * (1.0 - float(trail_giveback_pct))
                if price - entry <= floor_profit:
                    return (price - entry) * qty * MULT, (
                        f"trail_{trail_arm_pct:.2f}_{trail_giveback_pct:.2f}"
                    ), ts
    return _actual_result(trade)


def run_grid(trades: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    policies = []
    paths: dict[str, pd.DataFrame] = {}
    for symbol, group in bars.dropna(subset=["option_last_price"]).sort_values("audit_ts").groupby("option_symbol"):
        paths[str(symbol)] = group[["audit_ts", "option_last_price"]].reset_index(drop=True)
    trade_records = []
    for trade in trades.itertuples(index=False):
        s = pd.Series(trade._asdict())
        path = paths.get(str(s["symbol"]), pd.DataFrame(columns=["audit_ts", "option_last_price"]))
        if not path.empty:
            path = path[
                (path["audit_ts"] >= s["entry_time"])
                & (path["audit_ts"] <= s["exit_time"])
            ]
        trade_records.append((s, path))
    stop_losses = [None, 0.35, 0.45, 0.55, 0.70]
    no_progress = [(None, None), (45, 0.05), (60, 0.10), (90, 0.15), (120, 0.25)]
    trails = [(None, None), (0.50, 0.25), (0.75, 0.25), (1.00, 0.25)]
    take_profits = [None, 1.50, 2.00, 3.00]
    for sl, np_cfg, tr_cfg, tp in itertools.product(stop_losses, no_progress, trails, take_profits):
        np_min, np_mfe = np_cfg
        arm, gb = tr_cfg
        name = (
            f"sl={sl}|np={np_min}/{np_mfe}|trail={arm}/{gb}|tp={tp}"
        )
        rows = []
        for s, path in trade_records:
            pnl, reason, ts = _simulate_trade(
                s,
                path,
                stop_loss_pct=sl,
                no_progress_minutes=np_min,
                no_progress_mfe_pct=np_mfe,
                trail_arm_pct=arm,
                trail_giveback_pct=gb,
                take_profit_pct=tp,
            )
            rows.append(
                {
                    "symbol": s["symbol"],
                    "ticker": s["ticker"],
                    "option_type": s["option_type"],
                    "pnl": pnl,
                    "reason": reason,
                    "exit_ts": ts,
                }
            )
        res = pd.DataFrame(rows)
        for side, side_df in [("all", res), ("calls", res[res["option_type"] == "C"]), ("puts", res[res["option_type"] == "P"])]:
            policies.append(
                {
                    "policy": name,
                    "side": side,
                    "n": len(side_df),
                    "net_pnl": side_df["pnl"].sum(),
                    "avg_pnl": side_df["pnl"].mean(),
                    "win_rate": (side_df["pnl"] > 0).mean(),
                    "stop_hits": side_df["reason"].astype(str).str.startswith("stop_").sum(),
                    "np_hits": side_df["reason"].astype(str).str.startswith("no_progress_").sum(),
                    "trail_hits": side_df["reason"].astype(str).str.startswith("trail_").sum(),
                    "tp_hits": side_df["reason"].astype(str).str.startswith("tp_").sum(),
                    "actual_exits": side_df["reason"].isin(["actual_exit", "nan"]).sum(),
                }
            )
    return pd.DataFrame(policies).sort_values(["side", "net_pnl"], ascending=[True, False])


def summarize_by_side(results: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "option_exit_policy_grid.csv", index=False)
    lines = ["# Option Exit Policy Replay", ""]
    for side in ["all", "calls", "puts"]:
        top = results[results["side"] == side].sort_values("net_pnl", ascending=False).head(15)
        lines.append(f"## {side.title()}")
        lines.append(top.to_string(index=False))
        lines.append("")
    (out_dir / "option_exit_policy_grid.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    trades, bars = _load_inputs()
    results = run_grid(trades, bars)
    out = BASE / "experiments"
    summarize_by_side(results, out)
    baseline = {
        "all": trades["pnl_dollars"].sum(),
        "calls": trades.loc[trades["option_type"] == "C", "pnl_dollars"].sum(),
        "puts": trades.loc[trades["option_type"] == "P", "pnl_dollars"].sum(),
    }
    print("baseline", baseline)
    for side in ["all", "calls", "puts"]:
        top = results[results["side"] == side].sort_values("net_pnl", ascending=False).head(5)
        print(f"\n{side}")
        print(top.to_string(index=False))
    print(f"\nout={out}")


if __name__ == "__main__":
    main()
