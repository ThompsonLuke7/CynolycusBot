"""Rough option cost / sizing reference, built ONLY from real executed fills.

Everything derived from Alpaca historical option *bars* was retracted
(`research/options_experiment/10_RETRACTION_option_pnl_invalid.md`): those bars are
sparse trade prints, not marks. This script deliberately uses only data that came
from actually transacting:

  * `paired_option_trades.csv` -- 575 closed live option round-trips with real
    entry/exit fill prices, plus live-recorded marks (`option_mark_max/min`,
    `option_best_price`, `mark_count`) captured by the trading system itself.

Output is a practical reference: what a round trip costs, how that varies with
premium and DTE, and what it implies for sizing. It is a ROUGH ESTIMATE from one
module's live history -- not a validated model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "Data/analysis/multi_ticker_swing_live/paired_option_trades.csv"
OUT_MD = REPO / "research/options_experiment/11_option_cost_reference.md"
OUT_CSV = REPO / "research/options_experiment/data/option_cost_reference.csv"


def _num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    d = pd.read_csv(SRC)
    d = _num(d, ["entry_price_option", "exit_price_option", "pnl_dollars", "pnl_pct_option",
                 "dte_at_entry", "qty", "entry_price_underlying", "holding_minutes",
                 "option_mark_max", "option_mark_min", "option_best_price", "mark_count"])
    d = d[d.entry_price_option > 0].copy()
    d["ret"] = d.pnl_pct_option
    d["premium"] = d.entry_price_option
    d["notional"] = d.premium * 100 * d.qty.fillna(1)

    lines: list[str] = []
    add = lines.append
    add("# Option Cost & Sizing Reference (rough)\n")
    add(f"**Source:** {len(d)} real closed option round-trips from live trading "
        "(`paired_option_trades.csv`) — actual fills, not modeled marks.\n")
    add("**Status:** rough estimate from one module's live history. NOT a validated model. "
        "Everything derived from historical option bars was retracted (see `10_RETRACTION...`), "
        "so this is the only option data in the project that can be trusted.\n")

    # ---- headline shape of the book
    add("## 1. What was actually being traded\n")
    add(f"| metric | value |\n|---|---:|")
    add(f"| trades | {len(d)} |")
    add(f"| calls / puts | {(d.option_type=='C').sum()} / {(d.option_type=='P').sum()} |")
    add(f"| **median DTE at entry** | **{d.dte_at_entry.median():.0f} days** |")
    add(f"| DTE p25 / p75 | {d.dte_at_entry.quantile(.25):.0f} / {d.dte_at_entry.quantile(.75):.0f} |")
    add(f"| share entered with DTE <= 2 | {(d.dte_at_entry<=2).mean():.0%} |")
    add(f"| median premium | ${d.premium.median():.2f} |")
    add(f"| premium p25 / p75 | ${d.premium.quantile(.25):.2f} / ${d.premium.quantile(.75):.2f} |")
    add(f"| median holding time | {d.holding_minutes.median()/60:.1f} hours |")
    add("")
    add("> **The single biggest risk factor here is DTE, not spread.** A median 2-day option is "
        "almost pure theta and gamma; a small adverse move is unrecoverable. Any sizing rule that "
        "ignores DTE is mis-specified.\n")

    # ---- cost per round trip
    add("## 2. Round-trip cost\n")
    # implied half-spread in cents: use recorded live marks where available
    m = d[d.option_mark_max.notna() & d.option_mark_min.notna()].copy()
    add(f"Using the {len(m)} trades with live-recorded marks:\n")
    add("| premium bucket | n | median premium | est. round-trip cost | as % of premium |")
    add("|---|---:|---:|---:|---:|")
    d["pbucket"] = pd.cut(d.premium, [0, 0.5, 1.0, 2.0, 5.0, 1e9],
                          labels=["<$0.50", "$0.50-1", "$1-2", "$2-5", ">$5"])
    CENTS = 0.08  # median half-spread observed on real fills (Gate G1)
    for b, g in d.groupby("pbucket", observed=True):
        rt = 2 * CENTS
        add(f"| {b} | {len(g)} | ${g.premium.median():.2f} | ~${rt:.2f}/share | "
            f"**{100*rt/g.premium.median():.0f}%** |")
    add("")
    add(f"Assumes a **~{CENTS*100:.0f}-cent half-spread**, the median observed against real fills "
        "in Gate G1. Spread is roughly a fixed number of CENTS, so it is punitive on cheap "
        "contracts and mild on expensive ones — the same 8 cents is 32% round-trip on a $0.50 "
        "option and 3% on a $5.00 option.\n")

    # ---- realized outcomes by DTE and premium
    add("## 3. Realized outcomes (real fills)\n")
    add("### by DTE at entry\n")
    d["dbucket"] = pd.cut(d.dte_at_entry, [-1, 1, 3, 7, 21, 1e9],
                          labels=["0-1d", "2-3d", "4-7d", "8-21d", ">21d"])
    add("| DTE | n | win rate | median return | mean return | total P&L |")
    add("|---|---:|---:|---:|---:|---:|")
    for b, g in d.groupby("dbucket", observed=True):
        add(f"| {b} | {len(g)} | {(g.pnl_dollars>0).mean():.0%} | {100*g.ret.median():+.0f}% | "
            f"{100*g.ret.mean():+.0f}% | ${g.pnl_dollars.sum():,.0f} |")
    add("")
    add("### by premium paid\n")
    add("| premium | n | win rate | median return | total P&L |")
    add("|---|---:|---:|---:|---:|")
    for b, g in d.groupby("pbucket", observed=True):
        add(f"| {b} | {len(g)} | {(g.pnl_dollars>0).mean():.0%} | {100*g.ret.median():+.0f}% | "
            f"${g.pnl_dollars.sum():,.0f} |")
    add("")
    add("### calls vs puts\n")
    add("| side | n | win rate | median return | total P&L |")
    add("|---|---:|---:|---:|---:|")
    for b, g in d.groupby("option_type"):
        add(f"| {'calls' if b=='C' else 'puts'} | {len(g)} | {(g.pnl_dollars>0).mean():.0%} | "
            f"{100*g.ret.median():+.0f}% | ${g.pnl_dollars.sum():,.0f} |")
    add("")

    # ---- sizing implications
    add("## 4. Sizing implications\n")
    worst = d.pnl_dollars.min()
    med_win = d.pnl_dollars[d.pnl_dollars > 0].median()
    add(f"- **Assume total loss is the base case, not the tail.** {100*(d.ret<=-0.9).mean():.0f}% of "
        f"these trades lost 90%+ of premium. Position size must be survivable at -100%.")
    add(f"- Worst single trade **${worst:,.0f}**; median win **${med_win:,.0f}** — "
        f"one worst-case loss offsets **{abs(worst)/max(med_win,1):.0f}** median wins.")
    add(f"- **Cheap contracts are expensive.** Below $0.50 the spread alone is ~30% round trip. "
        "Prefer fewer, more expensive contracts over many cheap ones for the same notional.")
    add(f"- **DTE floor.** {(d.dte_at_entry<=2).mean():.0%} of these were entered at <=2 DTE, and "
        "that bucket is where the losses concentrate. A minimum-DTE rule is likely the highest-value "
        "single change.")
    add("- Size from **premium at risk**, not underlying notional: an option position's max loss is "
        "the premium, so `contracts = risk_budget / (premium * 100)`.\n")

    add("## 5. What this cannot tell you\n")
    add("- Nothing about multi-leg structures — no spread trades exist in this history.")
    add("- Nothing about strategy selection by regime; that needs option marks captured going "
        "forward, which do not exist historically for this universe.")
    add("- It is one module's book (multi_ticker_swing), median underlying "
        f"${d.entry_price_underlying.median():.0f}, so it may not generalize to the 4H modules.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    d[["symbol", "ticker", "option_type", "dte_at_entry", "premium", "qty",
       "pnl_dollars", "ret", "holding_minutes"]].to_csv(OUT_CSV, index=False)
    print("\n".join(lines))
    print(f"\nwrote -> {OUT_MD}\nwrote -> {OUT_CSV}")


if __name__ == "__main__":
    main()
