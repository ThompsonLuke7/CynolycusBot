"""Personal trade review: what separated the winning round trips from the losing ones.

    python -m core.personal_portfolio.trade_review [--reconcile]

Reads Data/personal/transactions.jsonl and the trade-history bar cache (fill them by opening the
History section of the personal page, or with trade_bars.refresh_daily/refresh_intraday).
``--reconcile`` also reads current Schwab positions (GET) to mark open trades and check them.
Writes Data/personal/reports/trade_review_{date}.md (gitignored: it is real-money P&L).

Evaluation protocol (fixed before looking at 2026 results by pattern):
  * Unit = a resolved round trip (closed, both ends in the record). P&L = cash incl. fees.
  * Splits are entry-time features only (trade_analysis); outcome-defined groups (hold length,
    held to expiry) describe exits, never entries.
  * Each split is shown for 2020-2025 and for 2026 separately: a pattern that does not keep its
    sign in the later period is not treated as real.
  * Win-rate differences get a two-proportion z and the minimum detectable difference at 80%
    power (2.8 standard errors); dollar averages get a bootstrap 95% range.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.personal_portfolio import data, trade_analysis as ta, trade_history as th

SPLIT_YEAR = "2026"
SPLITS = [
    ("SPY above its 50-day average", lambda d: d["spy_vs_sma50"] >= 0, lambda d: d["spy_vs_sma50"].notna()),
    ("Underlying above its 50-day average", lambda d: d["vs_sma50"] >= 0, lambda d: d["vs_sma50"].notna()),
    ("Both above their 50-day averages", lambda d: (d["vs_sma50"] >= 0) & (d["spy_vs_sma50"] >= 0),
     lambda d: d["vs_sma50"].notna() & d["spy_vs_sma50"].notna()),
    ("Option DTE at entry >= 8 days", lambda d: d["dte_at_entry"] >= 8, lambda d: d["kind"] == "option"),
    ("Option at or in the money (>= -5%)", lambda d: d["moneyness_entry"] >= -0.05, lambda d: d["moneyness_entry"].notna()),
    ("Not on a leveraged ETF", lambda d: d["underlying_class"] != "leveraged ETF", lambda d: d["underlying_class"].notna()),
    ("Underlying not down >2% into the entry", lambda d: d["move_into_entry"] >= -0.02, lambda d: d["move_into_entry"].notna()),
    ("Within 10% of the 20-day high", lambda d: d["dist_high_20"] >= -0.10, lambda d: d["dist_high_20"].notna()),
    ("Position size <= $1,000", lambda d: d["cost"].abs() <= 1000, lambda d: d["cost"].notna()),
    ("Calls (vs puts)", lambda d: d["put_call"] == "CALL", lambda d: d["kind"] == "option"),
    ("Held through an earnings report", lambda d: d["held_through_earnings"] == True,  # noqa: E712
     lambda d: d["held_through_earnings"].notna()),
]


TIMING_COLS = ("entry_style", "entry_pctile", "entry_wait_5d", "entry_prerun_5d", "exit_trigger", "exit_pctile",
               "post_exit_1d", "post_exit_2d", "post_exit_10d", "post_exit_best_10d", "drift_2d", "model_err_exit",
               "hold_to_expiry_pnl", "hold_plus_1d_pnl", "hold_plus_2d_pnl", "hold_plus_5d_pnl", "repo_policy_pnl",
               "take_profit_50_pnl", "roll_all_pnl", "iv_entry", "stop50_late_pnl", "dd50_dte", "dd50_expiry_value",
               "dd50_held_past", "dist_high_252", *(f"ev_{v}" for v in ("base", "delay1", "delay2", "retest2", "retest3",
                                                                       "retest5", "confirm2", "confirm4")))
COUNTERFACTUALS = [("hold_plus_1d_pnl", "Hold 1 more session", "model"), ("hold_plus_2d_pnl", "Hold 2 more sessions", "model"),
                   ("hold_plus_5d_pnl", "Hold 5 more sessions", "model"), ("hold_to_expiry_pnl", "Hold to expiry", "exact"),
                   ("take_profit_50_pnl", "Take profit at +50%", "model"),
                   ("repo_policy_pnl", "Repo live 4H exit policy", "model"),
                   ("roll_all_pnl", "Roll proceeds into a 30-day ATM option", "model entry, exact payoff")]


def _money(v) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{'-' if v < 0 else ''}${abs(v):,.0f}"


def _pct(v, signed=False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.0%}" if signed else f"{v:.0%}"


def frame(episodes: list[dict]) -> pd.DataFrame:
    rows = []
    for e in ta.resolved(episodes):
        rows.append({**{k: e.get(k) for k in ("key", "kind", "put_call", "pnl", "return_pct", "chart_symbol", "cost",
                                              "dte_at_entry", "hold_days", "exit_reason", "open_date", "n_entries")},
                     **{k: v for k, v in e.items() if k.startswith("b_")}, **e.get("ctx", {})})
    df = pd.DataFrame(rows)
    for col in ("spy_vs_sma50", "vs_sma50", "moneyness_entry", "move_into_entry", "dist_high_20", "dte_at_entry", "ret_20d",
                "under_move_dir", "mfe_dir", "mae_dir", "post_exit_5d_dir", "held_through_earnings", "underlying_class",
                *TIMING_COLS):
        if col not in df.columns:
            df[col] = np.nan
    df["win"] = df["pnl"] > 0
    df["year"] = df["open_date"].str[:4]
    return df


def _wr_test(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float | None, float | None]:
    n1, n2 = len(a), len(b)
    if n1 < 5 or n2 < 5:
        return None, None
    p = (a["win"].sum() + b["win"].sum()) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return ((a["win"].mean() - b["win"].mean()) / se if se else None), 2.8 * se


def split_table(df: pd.DataFrame) -> list[str]:
    out = ["| Entry condition | Period | With: n · win · median ret · avg $ | Without: n · win · median ret · avg $ | z (win rate) | detectable diff |",
           "|---|---|---|---|---|---|"]
    for name, cond, valid in SPLITS:
        for label, d in (("2020-25", df[df["year"] < SPLIT_YEAR]), (SPLIT_YEAR, df[df["year"] == SPLIT_YEAR]),
                         ("all", df)):
            d = d[valid(d).fillna(False).astype(bool)]
            m = cond(d).fillna(False).astype(bool)
            a, b = d[m], d[~m]
            z, mde = _wr_test(a, b)
            cell = lambda g: (f"{len(g)} · {_pct(g['win'].mean())} · {_pct(g['return_pct'].median(), True)} · "  # noqa: E731
                              f"{_money(g['pnl'].mean())}") if len(g) else "0"
            out.append(f"| {name if label == '2020-25' else ''} | {label} | {cell(a)} | {cell(b)} | "
                       f"{'—' if z is None else f'{z:+.1f}'} | {'—' if mde is None else f'{mde:.0%}'} |")
    return out


def breakdown_table(rows: list[dict], field: str, title: str) -> list[str]:
    out = [f"**{title}**", "", "| Group | n | Win rate | Net P&L | Avg P&L (95% range) | Median return | Profit factor |",
           "|---|---|---|---|---|---|---|"]
    for g in ta.breakdown(rows, field):
        ci = g["ci_avg_pnl"]
        rng = f"{_money(g['avg_pnl'])} ({_money(ci[0])} to {_money(ci[1])})" if ci else _money(g["avg_pnl"])
        pf = f"{g['profit_factor']:.2f}" if g.get("profit_factor") else "—"
        out.append(f"| {g['label']} | {g['n']} | {_pct(g['win_rate'])} | {_money(g['total_pnl'])} | {rng} | "
                   f"{_pct(g.get('median_return'), True)} | {pf} |")
    return out + [""]


def timing_section(df: pd.DataFrame) -> list[str]:
    opt = df[df["kind"] == "option"]
    out = ["## Timing: early or late, entry or exit or instrument", "",
           "Entry style and exit trigger use only what was known at that moment; every other timing column looks "
           "forward on purpose (it is a review, not a signal). Option what-ifs are Black-Scholes priced from the "
           "volatility implied by your own fills, except hold-to-expiry, which is exact.", ""]
    m = opt["model_err_exit"].dropna()
    if len(m):
        out += [f"- Model check: your {len(m)} exit fills priced from the entry fill's implied vol land within a median "
                f"{math.exp(m.abs().median()) - 1:.0%} of the real price (bias {m.median():+.2f} log). Median IV at entry "
                f"{opt['iv_entry'].median():.0%}.", ""]
    out += ["| Entry style (known at entry) | n | Win rate | Median return | Net P&L |", "|---|---|---|---|---|"]
    for k, g in df.groupby(df["entry_style"].fillna("n/a")):
        out.append(f"| {k} | {len(g)} | {_pct(g['win'].mean())} | {_pct(g['return_pct'].median(), True)} | {_money(g['pnl'].sum())} |")
    w, l_ = df[df["win"]], df[~df["win"]]
    out += ["", f"- Entries: {(df['entry_wait_5d'] <= -0.03).mean():.0%} of them saw the stock go another 3%+ against you "
                f"within 5 sessions (losers: median {l_['entry_wait_5d'].median():+.1%}, winners {w['entry_wait_5d'].median():+.1%}). "
                f"Before entering, the stock had typically already moved {df['entry_prerun_5d'].median():+.1%} off its 5-session "
                f"extreme. Median entry percentile in the +-5 session range: winners {w['entry_pctile'].median():.2f}, "
                f"losers {l_['entry_pctile'].median():.2f} (0 = the low).", "",
            "| Exit trigger (known at exit) | n | Next 2 sessions: mean (vs the name's normal drift) | +5% or more in 2 | -5% or worse in 2 | Best in next 10 (median) |",
            "|---|---|---|---|---|---|"]
    sold = df[df["exit_reason"] != "expired"]
    for k, g in sold.groupby(sold["exit_trigger"].fillna("n/a")):
        x = (g["post_exit_2d"] - g["drift_2d"]).dropna()
        if x.empty:
            continue                                   # trades whose bars failed the consistency check
        ci = 1.96 * x.std() / math.sqrt(len(x)) if len(x) > 2 else float("nan")
        out.append(f"| {k} | {len(g)} | {g['post_exit_2d'].mean():+.2%} ({x.mean():+.2%} ± {ci:.2%}) | "
                   f"{(g['post_exit_2d'] >= 0.05).mean():.0%} | {(g['post_exit_2d'] <= -0.05).mean():.0%} | "
                   f"{g['post_exit_best_10d'].median():+.1%} |")
    o = opt[opt["under_move_dir"].notna()]
    cats = [("right & paid", (o["under_move_dir"] > 0) & (o["pnl"] > 0), "it worked"),
            ("stock went your way, option still lost", (o["under_move_dir"] > 0) & (o["pnl"] <= 0), "instrument"),
            ("was right (3%+) during the trade, gave it back by exit",
             (o["under_move_dir"] <= 0) & (o["pnl"] <= 0) & (o["mfe_dir"] >= 0.03), "exit / management"),
            ("never worked", (o["under_move_dir"] <= 0) & (o["pnl"] <= 0) & (o["mfe_dir"].fillna(0) < 0.03), "entry / thesis"),
            ("stock went against you, option still paid", (o["under_move_dir"] <= 0) & (o["pnl"] > 0), "luck / IV")]
    out += ["", "| Where the option P&L went | n | Net P&L | Points at |", "|---|---|---|---|"]
    out += [f"| {lab} | {int(mask.sum())} ({mask.mean():.0%}) | {_money(o.loc[mask, 'pnl'].sum())} | {at} |" for lab, mask, at in cats]
    out += [f"", f"- Stock direction right by the exit on {(o['under_move_dir'] > 0).mean():.0%} of option trades; the option made "
                 f"money on {(o['pnl'] > 0).mean():.0%}. Same dollars in shares with the same timing: "
                 f"{_money((o['cost'].abs() * o['under_move_dir']).sum())} vs options {_money(o['pnl'].sum())}.", "",
            "| Exit what-if (options sold before expiry) | Basis | n | Actual | What-if | Better on | Median change | Change without its best 3 |",
            "|---|---|---|---|---|---|---|---|"]
    c = opt[opt["exit_reason"] != "expired"]
    for col, lab, basis in COUNTERFACTUALS:
        g = c[c[col].notna()]
        d = g[col] - g["pnl"]
        if not len(g):
            continue
        out.append(f"| {lab} | {basis} | {len(g)} | {_money(g['pnl'].sum())} | {_money(g[col].sum())} | {(d > 0).mean():.0%} | "
                   f"{_money(d.median())} | {_money(d.sum() - d.nlargest(3).sum())} |")
    return out + [""]


VARIANT_LABELS = [("ev_delay1", "Enter 1 session later (close)"), ("ev_delay2", "Enter 2 sessions later (close)"),
                  ("ev_retest2", "Limit 2% better within 5 sessions, else skip"),
                  ("ev_retest3", "Limit 3% better within 5 sessions, else skip"),
                  ("ev_retest5", "Limit 5% better within 5 sessions, else skip"),
                  ("ev_confirm2", "Wait for a close +2% your way within 5, else skip"),
                  ("ev_confirm4", "Wait for a close +4% your way within 5, else skip")]

# Registered 2026-11-18 after the in-sample study above. Trades OPENED on or after this date are out of sample.
REGISTERED = "2026-11-18"


def _auc(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None, int]:
    """Rank AUC of x for y (wins) and its standard error under no effect."""
    m = x.notna()
    x, y = x[m], y[m].astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 < 5 or n0 < 5:
        return None, None, int(m.sum())
    r = x.rank()
    a = (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(a), math.sqrt((n1 + n0 + 1) / (12 * n1 * n0)), int(m.sum())


def _late_stop(d: pd.DataFrame) -> tuple[float | None, int]:
    f = d[d["stop50_late_pnl"].notna() & (d["stop50_late_pnl"] != d["pnl"])]
    return ((f["stop50_late_pnl"] > f["pnl"]).mean() if len(f) else None), len(f)


def _comeback(d: pd.DataFrame) -> tuple[float | None, int]:
    g = d[d["dd50_dte"].notna() & (d["dd50_dte"] <= 14)]
    return ((g["dd50_expiry_value"] >= 1).mean() if len(g) else None), len(g)


def _short_dated_median(d: pd.DataFrame) -> tuple[float | None, int]:
    g = d[(d["kind"] == "option") & (d["dte_at_entry"] <= 7)]
    return (float(g["return_pct"].median()) if len(g) else None), len(g)


HYPOTHESES = [
    ("H1 Options at half value with <= 14 days left rarely get back to breakeven by expiry",
     "breakeven share <= 15%", lambda d: _comeback(d), lambda v: v <= 0.15),
    ("H2 Selling at that point beats holding", "beats holding on > 50%", lambda d: _late_stop(d), lambda v: v > 0.5),
    ("H3 Winners sit closer to their 52-week high at entry", "AUC > 0.5",
     lambda d: _auc(d["dist_high_252"], d["win"])[::2], lambda v: v > 0.5),
    ("H4 Lower option IV at entry wins more", "AUC < 0.5",
     lambda d: _auc(d.loc[d["kind"] == "option", "iv_entry"], d.loc[d["kind"] == "option", "win"])[::2], lambda v: v < 0.5),
    ("H5 Longer-dated options win more", "AUC > 0.5",
     lambda d: _auc(d.loc[d["kind"] == "option", "dte_at_entry"], d.loc[d["kind"] == "option", "win"])[::2],
     lambda v: v > 0.5),
    ("H6 Options with <= 7 days to expiry lose on the median trade", "median return < 0",
     lambda d: _short_dated_median(d), lambda v: v < 0),
    ("H7 Waiting for a 3% better price costs more than it saves", "net vs your entries < 0",
     lambda d: (((d["ev_retest3"].fillna(0) * d["cost"].abs()).sum() - (d["ev_base"] * d["cost"].abs()).sum())
                if d["ev_base"].notna().sum() else None, int(d["ev_base"].notna().sum())), lambda v: v < 0),
]


def wait_and_hold_section(df: pd.DataFrame) -> list[str]:
    out = ["## Should you wait to enter? Should you hold?", "",
           "Every variant keeps the trade's own holding period and exits at a daily close; option values are "
           "Black-Scholes from your entry fill's implied vol (estimates), stock prices are exact. Skipped = the price "
           "never offered that entry within 5 sessions.", ""]
    for label, g in (("Options", df[(df["kind"] == "option") & df["ev_base"].notna()]),
                     ("Stocks / ETFs", df[(df["kind"] != "option") & df["ev_base"].notna()])):
        if g.empty:
            continue
        cost = g["cost"].abs()
        base = (g["ev_base"] * cost).sum()
        out += [f"**{label}** — {len(g)} trades; your entries: win {(g['ev_base'] > 0).mean():.0%}, {_money(base)} on your sizes", "",
                "| Entry rule | Took (win) | P&L of taken | Same trades at your entry | Skipped: winners / losers (their P&L) | Net vs you |",
                "|---|---|---|---|---|---|"]
        for col, lab in VARIANT_LABELS:
            took, skip = g[g[col].notna()], g[g[col].isna()]
            tv = (took[col] * took["cost"].abs()).sum()
            out.append(f"| {lab} | {len(took)} ({(took[col] > 0).mean() if len(took) else 0:.0%}) | {_money(tv)} | "
                       f"{_money((took['ev_base'] * took['cost'].abs()).sum())} | {int((skip['ev_base'] > 0).sum())} / "
                       f"{int((skip['ev_base'] <= 0).sum())} ({_money((skip['ev_base'] * skip['cost'].abs()).sum())}) | "
                       f"{_money(tv - base)} |")
        out.append("")
    dd = df[df["dd50_dte"].notna()]
    if len(dd):
        out += ["**Once an option was worth half of what you paid** (first daily close at <= 50%, model; what it was worth "
                "at expiry is exact)", "", "| Days left at that point | n | Back to breakeven by expiry | Expired worthless | "
                "Average value at expiry (share of cost) |", "|---|---|---|---|---|"]
        for lab, g in (("<= 14", dd[dd["dd50_dte"] <= 14]), ("15-45", dd[(dd["dd50_dte"] > 14) & (dd["dd50_dte"] <= 45)]),
                       ("> 45", dd[dd["dd50_dte"] > 45]), ("all", dd)):
            if len(g):
                out.append(f"| {lab} | {len(g)} | {(g['dd50_expiry_value'] >= 1).mean():.0%} | "
                           f"{(g['dd50_expiry_value'] == 0).mean():.0%} | {g['dd50_expiry_value'].mean():.0%} |")
        share, n = _late_stop(df)
        f = df[df["stop50_late_pnl"].notna() & (df["stop50_late_pnl"] != df["pnl"])]
        out += ["", f"- You still held it past that point in {int(dd['dd50_held_past'].sum())} of {len(dd)} cases. "
                    f"Selling at the first close at <= 50% with <= 14 days left (model) fired on {n} trades you held: "
                    f"{_money(f['pnl'].sum())} -> {_money(f['stop50_late_pnl'].sum())}, better on "
                    f"{0 if share is None else share:.0%}; it cut {int((f['pnl'] > 0).sum())} eventual winners.", ""]
    return out


def forward_section(df: pd.DataFrame) -> list[str]:
    ins, fwd = df[df["open_date"] < REGISTERED], df[df["open_date"] >= REGISTERED]
    out = [f"## Registered hypotheses (registered {REGISTERED}; trades opened since then are out of sample)", "",
           "| Hypothesis | Passes if | In-sample (n) | Out of sample (n) | Verdict |", "|---|---|---|---|---|"]
    for name, rule, fn, ok in HYPOTHESES:
        cells, verdict = [], "waiting for trades"
        for part in (ins, fwd):
            try:
                v, n = fn(part)
            except (KeyError, ValueError, ZeroDivisionError):
                v, n = None, 0
            cells.append("—" if v is None else (f"{v:.2f} ({n})" if abs(v) < 5 else f"{_money(v)} ({n})"))
            if part is fwd and v is not None and n >= 30:
                verdict = "holds" if ok(v) else "fails"
            elif part is fwd and v is not None:
                verdict = f"too few ({n} < 30)"
        out.append(f"| {name} | {rule} | {cells[0]} | {cells[1]} | {verdict} |")
    a, se, n = _auc(ins["dist_high_252"], ins["win"])
    out += ["", f"- Scale: an AUC of 0.57 is about 1 standard error from 0.5 at 100 trades and about 2 at 400; "
                f"the in-sample AUCs above (n ~ {n}) carry a no-effect standard error of {0 if se is None else se:.3f}. "
                "Read the out-of-sample column after a few hundred new trades, not a few dozen.", ""]
    return out


def build_report(episodes: list[dict], checks: dict, issues: list[dict], rec: dict | None, now: datetime) -> str:
    res = ta.resolved(episodes)
    df = frame(episodes)
    s = ta.summarize(res)
    opt = df[df["kind"] == "option"]
    both = opt[opt["under_move_dir"].notna() & opt["return_pct"].notna()]
    rho = both["return_pct"].corr(both["under_move_dir"], method="spearman") if len(both) > 10 else None
    top5 = df.nlargest(5, "pnl")
    n_open = sum(e["status"] == "open" for e in episodes)
    n_unres = sum(e["status"] != "closed" and e["incomplete"] or e["status"] == "unresolved" for e in episodes)
    lines = [f"# Personal trade review — {now.astimezone(data.ET):%Y-%m-%d}", "",
             f"Schwab account, every trade from {episodes[0]['open_date'] if episodes else '—'} to "
             f"{max((e['close_date'] or e['open_date']) for e in episodes) if episodes else '—'}. "
             f"{s['n']} resolved round trips, {n_open} open, {n_unres} unresolved. In-sample description of one "
             "trader's history: nothing here is a backtest or a trading signal.", "",
             "## Data checks", "",
             f"- Opening/closing inferred from the running quantity agrees with Schwab's tag on "
             f"{checks['effect_checked'] - checks['effect_mismatch']} of {checks['effect_checked']} fills.",
             f"- Option trade return vs the underlying's directional move while held: Spearman "
             f"{'—' if rho is None else f'{rho:.2f}'} over {len(both)} option trades (fills and bars line up; "
             "theta and IV explain the rest).",
             f"- Records not attributed to any trade: " + ("; ".join(f"{(i.get('time') or '')[:10]} "
                                                                     f"{_money(i.get('net_amount'))} ({i['issue']})"
                                                                     for i in issues) or "none") + "."]
    if rec is not None:
        lines.append(f"- Current Schwab positions: {len(rec['matched'])} match the rebuilt history exactly; "
                     f"quantity mismatches {rec['qty_mismatch'] or 'none'}; held without history "
                     f"{rec['held_without_history'] or 'none'}; unresolved {[x['key'] for x in rec['open_not_held']] or 'none'}.")
    lines += ["", "## Overall", "",
              f"- Win rate {_pct(s['win_rate'])}; net realized {_money(s['total_pnl'])}; expectancy "
              f"{_money(s['avg_pnl'])}/trade (95% range {_money(ta.bootstrap_mean(df['pnl'].tolist())[0])} to "
              f"{_money(ta.bootstrap_mean(df['pnl'].tolist())[1])}); profit factor {s['profit_factor']:.2f}; "
              f"median return {_pct(s['median_return'], True)}; median hold {s['median_hold_days']:.1f} days.",
              f"- Average win {_money(s['avg_win'])} vs average loss {_money(s['avg_loss'])}.",
              f"- Concentration: the 5 best trades made {_money(top5['pnl'].sum())} "
              f"({', '.join(top5['key'])}). Without them the history is {_money(df['pnl'].sum() - top5['pnl'].sum())}.",
              ""]
    lines += breakdown_table(res, "b_year", "By year opened")
    lines += ["## Entry conditions: with vs without, and does it hold in " + SPLIT_YEAR + "?", "",
              "Only information known at the entry fill. 'detectable diff' = the win-rate gap this sample "
              "could detect 80% of the time; a smaller real effect would look like noise here.", ""]
    lines += split_table(df) + [""]
    w, l_ = opt[opt["win"]], opt[~opt["win"]]
    worked = l_[l_["mfe_dir"] >= 0.03]
    lines += ["## Exits (outcome measures, not entry signals)", "",
              f"- Median hold: winners {df[df['win']]['hold_days'].median():.1f} days vs losers "
              f"{df[~df['win']]['hold_days'].median():.1f} days.",
              f"- Losing option trades whose underlying was at least 3% in your favour at some point while held: "
              f"{len(worked)} of {l_['mfe_dir'].notna().sum()} ({len(worked) / max(1, l_['mfe_dir'].notna().sum()):.0%}); "
              f"together they lost {_money(worked['pnl'].sum())}.",
              f"- Winning option trades: median best underlying move while held {w['mfe_dir'].median():+.1%}, "
              f"median move captured at exit {w['under_move_dir'].median():+.1%}; the move continued >= 3% in the "
              f"5 sessions after exit in {(w['post_exit_5d_dir'] >= 0.03).sum()} of {w['post_exit_5d_dir'].notna().sum()}.",
              f"- Held to expiry: {int((df['exit_reason'] == 'expired').sum())} trades, "
              f"{int(df[df['exit_reason'] == 'expired']['win'].sum())} won, "
              f"{_money(df[df['exit_reason'] == 'expired']['pnl'].sum())}.",
              "- Without option marks the counterfactual exit (what selling at +X% on the underlying would have "
              "paid) cannot be priced from this data; see next steps.", ""]
    lines += timing_section(df)
    lines += wait_and_hold_section(df)
    lines += forward_section(df)
    q = df["return_pct"].quantile(0.75)
    best, rest = df[df["return_pct"] >= q], df[df["return_pct"] < q]
    prof = [("SPY above 50-day", lambda d: (d["spy_vs_sma50"] >= 0).mean()),
            ("Underlying above 50-day", lambda d: (d["vs_sma50"] >= 0).mean()),
            ("Median DTE at entry", lambda d: d.loc[d["kind"] == "option", "dte_at_entry"].median()),
            ("Median moneyness at entry", lambda d: d["moneyness_entry"].median()),
            ("Median 20-day return before entry", lambda d: d["ret_20d"].median()),
            ("Median distance from 20-day high", lambda d: d["dist_high_20"].median()),
            ("Median move into entry", lambda d: d["move_into_entry"].median()),
            ("Leveraged-ETF share", lambda d: (d["underlying_class"] == "leveraged ETF").mean()),
            ("Held through earnings", lambda d: (d["held_through_earnings"] == True).mean()),  # noqa: E712
            ("Median size", lambda d: d["cost"].abs().median()),
            ("Median hold (days)", lambda d: d["hold_days"].median())]
    lines += [f"## Profile: your best 25% of trades (return >= {q:+.0%}) vs the rest", "",
              f"| | Best 25% (n={len(best)}) | Rest (n={len(rest)}) |", "|---|---|---|"]
    for name, fn in prof:
        a, b = fn(best), fn(rest)
        fmt = (lambda v: f"{v:.0f}") if "DTE" in name or "days" in name else \
              (lambda v: _money(v)) if "size" in name else (lambda v: f"{v:+.1%}") if "Median" in name else _pct
        lines.append(f"| {name} | {fmt(a)} | {fmt(b)} |")
    lines += [""]
    for field, title in (("b_type", "By type"), ("b_dte", "By DTE at entry"), ("b_moneyness", "By moneyness at entry"),
                         ("b_trend", "By trend at entry"), ("b_spy", "By SPY regime at entry"),
                         ("b_class", "By underlying class"), ("b_earnings", "By earnings timing"),
                         ("b_size", "By position size"), ("b_entry_time", "By entry time (ET)"),
                         ("b_entry_style", "By entry style"), ("b_exit_trigger", "By exit trigger"),
                         ("b_hold", "By holding time (outcome-defined)"), ("b_exit", "By exit (outcome-defined)")):
        lines += breakdown_table(res, field, title)
    lines += ["## Caveats", "",
              f"- {s['n']} trades, heavy-tailed P&L: dollar averages are dominated by a handful of trades; win rate "
              "and median return are the steadier measures.",
              f"- {len(SPLITS)} entry splits were examined; at that count one |z| > 2 is expected by chance. A split "
              "counts only if it keeps its sign in both periods.",
              "- Features are measured on the underlying; option IV at entry is not available historically.",
              ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reconcile", action="store_true", help="read current Schwab positions (GET) to mark open trades")
    parser.add_argument("--base", type=Path, default=None)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    fills, issues = th.normalize_fills(th.load_raw(args.base))
    if not fills:
        raise SystemExit("no stored transactions: open the History section of the personal page first")
    episodes, checks = th.build_episodes(fills, th.load_symbol_map(args.base))
    rec = None
    if args.reconcile:
        from core.personal_portfolio.holdings import fetch_holdings
        rec = th.attach_marks(episodes, fetch_holdings())
    ta.enrich(episodes, args.base, earnings=data.load_earnings(args.base))
    out = data.personal_dir(args.base) / "reports" / f"trade_review_{now.astimezone(data.ET):%Y-%m-%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(episodes, checks, issues, rec, now))
    print(out)


if __name__ == "__main__":
    main()
