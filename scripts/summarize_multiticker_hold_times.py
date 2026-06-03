from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_multiticker_entry_timing_experiment import _events_to_frames

DEFAULT_AUDITS = [
    Path("UI/swing_audit/swing_session_20260528T120501Z.jsonl"),
    Path("UI/swing_audit/swing_session_20260529T120845Z.jsonl"),
    Path("UI/swing_audit/paper/swing_session_20260601T120304Z.jsonl"),
]


def _load_closed(audits: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for audit in audits:
        if not audit.exists():
            print(f"missing_audit={audit}")
            continue
        closed, _, _ = _events_to_frames(audit)
        if not closed.empty:
            frames.append(closed)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audits", nargs="*", type=Path, default=DEFAULT_AUDITS)
    args = parser.parse_args()

    df = _load_closed(args.audits)
    if df.empty:
        print("closed_trades=0")
        return

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["closed_ts"] = pd.to_datetime(df["closed_ts"], utc=True, errors="coerce")
    df["hold_minutes"] = (df["closed_ts"] - df["entry_time"]).dt.total_seconds() / 60.0
    df["option_pnl_dollars"] = pd.to_numeric(df["option_pnl_dollars"], errors="coerce")
    df["option_ret_pct"] = pd.to_numeric(df["option_ret_pct"], errors="coerce")
    df["underlying_signed_ret_pct"] = pd.to_numeric(df["underlying_signed_ret_pct"], errors="coerce")
    df["direction"] = pd.to_numeric(df["direction"], errors="coerce")
    df = df.dropna(subset=["entry_time", "closed_ts", "hold_minutes", "option_pnl_dollars"])
    df = df[df["hold_minutes"] >= 0].copy()

    fresh = df[df["is_fresh"].astype(str).str.lower().eq("true")]
    calls = df[df["direction"].eq(1)]
    puts = df[df["direction"].eq(-1)]

    print("sources=" + ",".join(str(path) for path in args.audits))
    print(f"closed_trades={len(df)}")
    print(f"avg_hold_minutes={df['hold_minutes'].mean():.2f}")
    print(f"median_hold_minutes={df['hold_minutes'].median():.2f}")
    print(f"fresh_closed_trades={len(fresh)}")
    print(f"fresh_avg_hold_minutes={fresh['hold_minutes'].mean():.2f}")
    print(f"calls_avg_hold_minutes={calls['hold_minutes'].mean():.2f}")
    print(f"puts_avg_hold_minutes={puts['hold_minutes'].mean():.2f}")
    print()
    print("top_3_by_option_pnl:")
    cols = [
        "ticker",
        "direction",
        "entry_time",
        "closed_ts",
        "hold_minutes",
        "option_symbol",
        "option_entry_price",
        "option_exit_price",
        "option_pnl_dollars",
        "option_ret_pct",
        "underlying_signed_ret_pct",
        "exit_reason",
        "is_fresh",
    ]
    top = df.sort_values("option_pnl_dollars", ascending=False).head(3).copy()
    top["hold_minutes"] = top["hold_minutes"].round(2)
    top["option_ret_pct"] = top["option_ret_pct"].round(2)
    top["underlying_signed_ret_pct"] = top["underlying_signed_ret_pct"].round(2)
    print(top[cols].to_string(index=False))

    if not fresh.empty:
        print()
        print("top_3_fresh_by_option_pnl:")
        fresh_top = fresh.sort_values("option_pnl_dollars", ascending=False).head(3).copy()
        fresh_top["hold_minutes"] = fresh_top["hold_minutes"].round(2)
        fresh_top["option_ret_pct"] = fresh_top["option_ret_pct"].round(2)
        fresh_top["underlying_signed_ret_pct"] = fresh_top["underlying_signed_ret_pct"].round(2)
        print(fresh_top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
