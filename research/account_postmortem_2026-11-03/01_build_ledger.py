"""Build the account's interval-by-symbol P&L ledger, 2026-07-15 .. 2026-11-02.

For consecutive account snapshots t0 < t1 and every symbol s:

    contribution(s) = MV_t1(s) - MV_t0(s) + net fill cash of s in (t0, t1]

Fees, dividends and merger cash go to an `__other__` row. The sum of an
interval's rows must equal equity_t1 - equity_t0 (deposits excluded); the
residual is reported, not hidden.

Each contribution is tied to a trade *episode*: a symbol's run from the buy
that opens it from flat (per the broker fill history) to flat again. Each
episode is attributed to the module whose closed-trade ledger or live state
records it; anything unproven stays `unknown`.

Known mark artefact corrected here (and only here): ENHA's 2026-10-29
snapshots priced the post-split share at the pre-split quantity (3,436 x $12.50
instead of 343.6). Its MV on those snapshots is restated at qty/10.

Inputs are read-only. Outputs go to Data/research/account_postmortem_20261103/.

    PYTHONPATH=. .venv/bin/python research/account_postmortem_2026-11-03/01_build_ledger.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("Data/research/account_postmortem_20261103")
ET = "America/New_York"


def option_multiplier(symbol: str) -> int:
    return 100 if len(symbol) > 10 and symbol[-9] in "CP" else 1


def load_snapshots() -> tuple[pd.DataFrame, pd.DataFrame]:
    acct, pos = [], []
    for f in sorted(glob.glob("Data/inference/account_snapshots/broker_equity_*_paper.jsonl")):
        for line in open(f):
            r = json.loads(line)
            ts = pd.Timestamp(r["captured_at_utc"]).tz_convert(ET)
            a = r["account"]
            acct.append({"ts": ts, "equity": float(a["equity"]), "cash": float(a.get("cash") or 0)})
            for p in r.get("positions") or []:
                pos.append({"ts": ts, "symbol": p["symbol"], "qty": float(p["qty"]),
                            "mv": float(p["market_value"]), "price": float(p["current_price"]),
                            "avg_entry": float(p["avg_entry_price"]), "asset_class": p.get("asset_class")})
    acct = pd.DataFrame(acct).drop_duplicates("ts").sort_values("ts")
    pos = pd.DataFrame(pos).drop_duplicates(["ts", "symbol"])
    # One mark per session: the last snapshot captured on each calendar day.
    acct["day"] = acct.ts.dt.date
    daily = acct.groupby("day").tail(1).reset_index(drop=True)
    pos = pos[pos.ts.isin(daily.ts)].copy()
    enha = (pos.symbol == "ENHA") & (pos.qty == 3436) & (pos.price > 5)
    pos.loc[enha, "mv"] = pos.loc[enha, "qty"] / 10 * pos.loc[enha, "price"]
    daily.loc[daily.ts.isin(pos.loc[enha, "ts"]), "equity"] -= (
        pos.loc[enha].groupby("ts").apply(lambda g: (g.qty * g.price).sum() - g.mv.sum())
        .reindex(daily.loc[daily.ts.isin(pos.loc[enha, "ts"]), "ts"]).to_numpy()
    )
    return daily, pos


def exercise_fills() -> pd.DataFrame:
    """Auto-exercises as pseudo-fills of the underlying at the strike.

    The paper account's OPEXC activities carry net_amount 0, but the broker
    delivers 100 shares per contract and debits strike x 100 in cash (the four
    08-14 exercises are exactly the $266,750 the ledger was missing). The
    option leg needs no row: its position simply disappears from the next
    snapshot. Timed at 23:59 ET on the expiry date, after the day's last mark.
    """

    rows = json.loads((OUT / "broker_nontrade_20260701_20261103.json").read_text())
    out = []
    for r in rows:
        if r.get("activity_type") != "OPEXC":
            continue
        occ = r["symbol"]
        root, right, strike = occ[:-15], occ[-9], int(occ[-8:]) / 1000
        out.append({"id": f"opexc:{r['id']}", "symbol": root, "side": "buy" if right == "C" else "sell",
                    "qty": abs(float(r["qty"])) * 100, "price": strike, "order_id": f"opexc:{occ}",
                    "transaction_time": f"{r['date']}T23:59:00-04:00", "exercised_from": occ})
    return pd.DataFrame(out)


def load_fills() -> pd.DataFrame:
    rows = json.loads((OUT / "broker_fills_20260701_20261103.json").read_text())
    f = pd.concat([pd.DataFrame(rows), exercise_fills()], ignore_index=True)
    f["ts"] = pd.to_datetime(f.transaction_time, utc=True, format="mixed").dt.tz_convert(ET)
    f["qty"] = f.qty.astype(float)
    f["price"] = f.price.astype(float)
    f["mult"] = f.symbol.map(option_multiplier)
    f["signed_qty"] = np.where(f.side == "buy", f.qty, -f.qty)
    f["cash"] = -f.signed_qty * f.price * f.mult
    return f.sort_values("ts").reset_index(drop=True)


def load_other_cash() -> pd.DataFrame:
    rows = json.loads((OUT / "broker_nontrade_20260701_20261103.json").read_text())
    o = pd.DataFrame(rows)
    o["net"] = pd.to_numeric(o.net_amount, errors="coerce").fillna(0.0)
    o = o[(o.activity_type != "JNLC") & (o.net != 0)].copy()
    # Non-trade activities carry a date, not a time; treat them as end of day.
    o["ts"] = pd.to_datetime(o.date).dt.tz_localize(ET) + pd.Timedelta(hours=17)
    return o[["ts", "activity_type", "symbol", "net"]]


def assign_episodes(f: pd.DataFrame) -> pd.DataFrame:
    """Episode id per fill: a new one starts on a buy from a flat (fill-based) position."""

    f = f.copy()
    f["episode"] = None
    for sym, g in f.groupby("symbol", sort=False):
        held, ep = 0.0, None
        for idx, row in g.iterrows():
            if row.side == "buy" and held <= 1e-9:
                ep = f"{sym}@{row.ts.isoformat()}"
            f.at[idx, "episode"] = ep
            held = max(0.0, held + row.signed_qty)
    return f


UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
BLOCK_MARKERS = (
    ("multi_ticker_swing_htf", ("HTF Swing", "=== htf ===", "multi_ticker_swing_htf/live/runner.py")),
    ("meta_ranker", ("Meta Ranker", "=== meta ===", "meta_ranker/live_runner.py")),
    ("momentum_expansion", ("Momentum Expansion", "=== momentum ===", "momentum_expansion/live/runner.py")),
    ("dealer_ranker", ("Dealer Ranker", "live_ranked_options.py")),
)


def order_owner_index() -> dict[str, tuple[str, str]]:
    """Entry order id -> (module, evidence) from every server log.

    Direct evidence: `<module> pending-open: submitted buy ... id=<uuid>` and
    the 30m swing's `buy limit order submitted ... 'id': '<uuid>'`. Block
    evidence: an `OK buy ... id=<uuid>` line inside the stdout block of the
    runner last launched (by its launch line or banner). Runner subprocesses
    are staggered (:20 Meta, :25 HTF, :31 Momentum, :15 Dealer), but pre-open
    flushes can overlap, so block evidence is labelled separately.
    """

    cache = OUT / "order_owner_index.json"
    if cache.exists():
        return {k: tuple(v) for k, v in json.loads(cache.read_text()).items()}
    owners: dict[str, tuple[str, str]] = {}
    for path in sorted(glob.glob("logs/live_server/server_*.log")):
        block = None
        with open(path, errors="replace") as fh:
            for line in fh:
                for module, markers in BLOCK_MARKERS:
                    if any(m in line for m in markers) and ("launching" in line or line.startswith("===")):
                        block = module
                if "id" not in line:
                    continue
                m = UUID.search(line)
                if not m:
                    continue
                oid = m.group(0)
                if "pending-open: submitted buy" in line:
                    mod = line.split("[core.live_4h_exec]")[-1].split("pending-open")[0].strip()
                    owners[oid] = (mod, "pending_open_line")
                elif "multi_ticker_swing.live.runner" in line and "buy limit order submitted" in line:
                    owners[oid] = ("multi_ticker_swing", "swing_submit_line")
                elif line.lstrip().startswith("OK buy") and block and oid not in owners:
                    owners[oid] = (block, "runner_block")
    cache.write_text(json.dumps(owners))
    return owners


def module_map(f: pd.DataFrame) -> dict[str, str]:
    """Episode -> module: entry-order evidence first, then exits and live claims."""

    ep_of_order = f.dropna(subset=["episode"]).groupby("order_id").episode.first().to_dict()
    owners = order_owner_index()
    entry = f[(f.side == "buy")].dropna(subset=["episode"]).groupby("episode").order_id.first()
    by_entry = {ep: owners[oid][0] for ep, oid in entry.items() if oid in owners}
    votes: dict[str, set] = {}
    for path in glob.glob("Data/inference/*/closed_trades.jsonl"):
        module = path.split("/")[2]
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            ep = ep_of_order.get(r.get("order_id"))
            if ep:
                votes.setdefault(ep, set()).add(r.get("module") or module)
    from core.orphan_positions import MANAGED_STATE_PATHS

    first_buy = f[f.side == "buy"].groupby("episode").ts.min()
    for module, path in MANAGED_STATE_PATHS.items():
        for key, st in json.load(open(path))["managed"].items():
            sym = st.get("occ") or st.get("symbol") or key
            eps = [e for e in first_buy.index if e.startswith(f"{sym}@")]
            if eps:
                votes.setdefault(max(eps, key=lambda e: first_buy[e]), set()).add(module)
    swing = json.load(open("Data/inference/multi_ticker_swing/open_positions.json"))
    for p in swing.get("positions") or []:
        eps = [e for e in first_buy.index if e.startswith(f"{p.get('option_symbol')}@")]
        if eps:
            votes.setdefault(max(eps, key=lambda e: first_buy[e]), set()).add("multi_ticker_swing")
    # Symbol-and-time fallback for exits recorded without an order id (the 30m
    # swing, Intraday, SPY): a closed row of the symbol between entry and three
    # days after the last fill (expiries settle after it).
    span = f.dropna(subset=["episode"]).groupby("episode").agg(sym=("symbol", "first"), t0=("ts", "min"), t1=("ts", "max"))
    closed = []
    for path in glob.glob("Data/inference/*/closed_trades.jsonl"):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                closed.append((r.get("module") or path.split("/")[2], r.get("order_symbol") or r.get("ticker"), r.get("ts")))
    closed = pd.DataFrame(closed, columns=["module", "sym", "ts"])
    closed["ts"] = pd.to_datetime(closed.ts, utc=True, format="mixed", errors="coerce").dt.tz_convert(ET)
    by_sym = {s: g for s, g in closed.groupby("sym")}
    for ep, row in span.iterrows():
        if ep in votes or ep in by_entry or row.sym not in by_sym:
            continue
        g = by_sym[row.sym]
        hit = g[(g.ts >= row.t0 - pd.Timedelta(hours=1)) & (g.ts <= row.t1 + pd.Timedelta(days=3))]
        if len(hit):
            votes.setdefault(ep, set()).update(hit.module)
    out = {ep: (next(iter(m)) if len(m) == 1 else "+".join(sorted(m))) for ep, m in votes.items()}
    # The entry order outranks exit-side votes: ownership follows the entry.
    out.update(by_entry)
    # HTF and Momentum both launch at :25, so a runner-block attribution can
    # name the wrong one. A module's own audited order plan listing the buy
    # (the latest plan at or before the fill, within two days for deferred
    # opens) settles it when exactly one module planned it.
    plans = []
    for path in glob.glob("Data/inference/*/live_signal_audit.jsonl"):
        module = path.split("/")[2]
        for line in open(path):
            if '"order_plan"' not in line:
                continue
            r = json.loads(line)
            for p in (r.get("plan") or []) + (r.get("planned") or []):
                if p.get("side") == "buy":
                    plans.append((module, p.get("symbol"), r.get("bar")))
    plans = pd.DataFrame(plans, columns=["module", "sym", "bar"])
    plans["bar"] = pd.to_datetime(plans.bar, utc=True, format="mixed", errors="coerce").dt.tz_convert(ET)
    plan_by_sym = {s: g for s, g in plans.groupby("sym")}
    first_fill = f[f.side == "buy"].dropna(subset=["episode"]).groupby("episode").agg(sym=("symbol", "first"), ts=("ts", "min"), oid=("order_id", "first"))
    for ep, row in first_fill.iterrows():
        if owners.get(row.oid, ("", ""))[1] in {"pending_open_line", "swing_submit_line"} or row.sym not in plan_by_sym:
            continue
        g = plan_by_sym[row.sym]
        g = g[(g.bar <= row.ts) & (g.bar >= row.ts - pd.Timedelta(days=2))]
        if len(g):
            latest = g[g.bar == g.bar.max()]
            if latest.module.nunique() == 1:
                out[ep] = latest.module.iloc[0]
    # Shares delivered by an exercise belong to whoever held the option.
    for _, row in f[f.order_id.astype(str).str.startswith("opexc:")].iterrows():
        opt_eps = f[(f.symbol == row.exercised_from) & f.episode.notna()].episode
        if row.episode and len(opt_eps) and opt_eps.iloc[-1] in out and row.episode not in out:
            out[row.episode] = out[opt_eps.iloc[-1]]
    return out


def main() -> None:
    daily, pos = load_snapshots()
    fills = assign_episodes(load_fills())
    other = load_other_cash()
    modules = module_map(fills)

    mv = pos.pivot_table(index="ts", columns="symbol", values="mv", aggfunc="sum").reindex(daily.ts).fillna(0.0)
    rows, acct_rows = [], []
    for k in range(1, len(daily)):
        t0, t1 = daily.ts.iloc[k - 1], daily.ts.iloc[k]
        fw = fills[(fills.ts > t0) & (fills.ts <= t1)]
        cash = fw.groupby("symbol").cash.sum()
        syms = sorted(set(mv.columns[(mv.loc[t0] != 0) | (mv.loc[t1] != 0)]) | set(cash.index))
        for s in syms:
            c = float(mv.at[t1, s] if s in mv else 0) - float(mv.at[t0, s] if s in mv else 0) + float(cash.get(s, 0.0))
            # The episode live at the interval end (or, if it closed inside, its last).
            sf = fills[(fills.symbol == s) & (fills.ts <= t1)]
            ep = sf.episode.dropna().iloc[-1] if len(sf.episode.dropna()) else f"{s}@pre"
            rows.append({"t0": t0, "t1": t1, "symbol": s, "episode": ep, "contrib": c,
                         "mv_t0": float(mv.at[t0, s] if s in mv else 0), "mv_t1": float(mv.at[t1, s] if s in mv else 0),
                         "fill_cash": float(cash.get(s, 0.0))})
        oth = float(other[(other.ts > t0) & (other.ts <= t1)].net.sum())
        rows.append({"t0": t0, "t1": t1, "symbol": "__other__", "episode": "__other__", "contrib": oth,
                     "mv_t0": 0.0, "mv_t1": 0.0, "fill_cash": 0.0})
        d_eq = daily.equity.iloc[k] - daily.equity.iloc[k - 1]
        acct_rows.append({"t0": t0, "t1": t1, "equity_t0": daily.equity.iloc[k - 1], "equity_t1": daily.equity.iloc[k],
                          "d_equity": d_eq, "gross_t0": float(mv.loc[t0].clip(lower=0).sum())})
    contrib = pd.DataFrame(rows)
    contrib["module"] = contrib.episode.map(modules).fillna("unknown")
    contrib.loc[contrib.symbol == "__other__", "module"] = "__other__"
    contrib["route"] = np.where(contrib.symbol.map(option_multiplier) == 100, "option", "equity")
    acct = pd.DataFrame(acct_rows)
    sums = contrib.groupby("t1").contrib.sum()
    acct["ledger_sum"] = acct.t1.map(sums)
    acct["residual"] = acct.d_equity - acct.ledger_sum

    ep = fills.dropna(subset=["episode"]).groupby("episode").agg(
        symbol=("symbol", "first"), entry_ts=("ts", "min"), last_fill_ts=("ts", "max"),
        buy_cost=("cash", lambda c: -c[c < 0].sum()), sell_proceeds=("cash", lambda c: c[c > 0].sum()))
    ep["module"] = ep.index.map(modules).fillna("unknown")
    ep["route"] = np.where(ep.symbol.map(option_multiplier) == 100, "option", "equity")
    ep["pnl"] = contrib.groupby("episode").contrib.sum().reindex(ep.index).fillna(0.0)

    contrib.to_parquet(OUT / "ledger_contrib.parquet", index=False)
    acct.to_parquet(OUT / "ledger_account.parquet", index=False)
    ep.reset_index().to_parquet(OUT / "ledger_episodes.parquet", index=False)

    print(f"intervals {len(acct)}  rows {len(contrib)}  episodes {len(ep)}")
    print(f"equity {daily.equity.iloc[0]:,.0f} ({daily.ts.iloc[0]:%m-%d}) -> {daily.equity.iloc[-1]:,.0f} ({daily.ts.iloc[-1]:%m-%d})")
    print(f"sum d_equity {acct.d_equity.sum():,.0f}  ledger {acct.ledger_sum.sum():,.0f}  "
          f"|residual| total {acct.residual.abs().sum():,.0f}  max {acct.residual.abs().max():,.0f}")
    print("unattributed episodes:", int((ep.module == "unknown").sum()), "pnl", round(ep.loc[ep.module == "unknown", "pnl"].sum()))


if __name__ == "__main__":
    main()
