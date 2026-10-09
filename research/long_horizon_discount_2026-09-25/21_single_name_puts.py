"""Puts on the single names instead of the index: buy on the first red day after an extended run? (2026-10-07)

The user's follow-up to Part 9 (index puts are insurance, not an edge), with an NVDA chart: the stocks themselves
look overextended before they fall and the fall "usually lasts a few days", so buy puts on the NAME on its first
red day or on bad news. Four checks, cheapest first:

A. EVENT STUDY, shares only (no option model). What does a liquid-300 name do over the next 3/5/10 sessions after
   the trigger, against (i) every day a name was extended and (ii) every name-day? Signal on the close of day t,
   entry at the next open. "Extended" has two definitions, both reported:
     pct  close >= 15% above its 50-day average
     z    log(close / 50-day average) >= 1.5 x (63-day daily vol x sqrt(25)), i.e. extended for ITS OWN volatility
   Triggers (the name was extended on day t-1):
     red1  day t closes down and day t-1 was a 20-day closing high (the first red day off the high)
     red3  the same, and day t closes down 3% or more
     fade  first close below the 10-day average after 10 or more sessions above it (momentum running out)
     gap   day t opens 3% or more below the prior close and closes down (the price proxy for bad news; the
           catalyst feed does not reach back to 2019)
B. WHAT A PUT COSTS: real quotes. The nightly IV-surface capture holds closing bid/ask for ~910 names on 13
   sessions (2026-09-11..10-07). For the 2-week at-the-money put: implied vol against the name's own trailing
   realized vol, the bid/ask, and whether the put is dearer on the trigger day itself.
C. PUT ECONOMICS on the events of A: Black-Scholes premium at the implied vol B measures (a line fitted to
   implied against realized vol, because the ratio is NOT constant: it falls from 1.0 on quiet names to 0.7 on
   the most volatile) against the payoff the shares actually delivered. "Payback" = payoff per $1 of premium;
   1.00 is a fair price before the bid/ask. "Break-even" = the multiple of the quoted vol at which it pays.
D. PORTFOLIO: the 70/30 blend of Part 9, plus a put on each held name when its trigger fires (one put per share
   held, at the money, 10 sessions, one open put per name). Paired against the fixed 70/30.
E. THE CHART: every first red day off a 20-day closing high in NVDA over the last 12 months, and what followed.

PUTS IN C AND D ARE MODELLED, NOT QUOTED: single-name option history is trade prints (the retracted study), so
only B uses real prices and it covers 4 weeks. The +-20% rows and the break-even multiple bracket that.

    .venv/bin/python research/long_horizon_discount_2026-09-25/21_single_name_puts.py
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IV_DIR = REPO / "Data" / "dealer_positioning" / "iv_surface"
RAW_1D = REPO / "Data" / "shared" / "bars" / "1d"


def _load(name: str, file: str):
    spec = spec_from_file_location(name, HERE / file)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


an = _load("an", "18_sleeve_anatomy.py")
rb, pb = an.rb, an.pb
N_TOP, STEP, SHARE = 20, 4, 0.30
EXT_PCT, EXT_Z = 0.15, 1.5
HORIZONS = (3, 5, 10)
PUT_DAYS = 10                    # sessions the put lives (a 2-week expiry)
TRIGGERS = ["red1", "red3", "fade", "gap"]
SEED = 11


def bs_put(s, k, t, sigma):
    """Black-Scholes put, zero rate. Arrays broadcast; t in years."""
    s, k, sigma = np.asarray(s, float), np.asarray(k, float), np.asarray(sigma, float)
    t = np.maximum(np.asarray(t, float), 1e-9)
    sd = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + 0.5 * sd ** 2) / sd
    return k * ndtr(-(d1 - sd)) - s * ndtr(-d1)


def implied_vol(price, s, k, t) -> np.ndarray:
    """Bisection inverse of bs_put. NaN where the price is at or below intrinsic value."""
    price, s, k, t = (np.asarray(x, float) for x in (price, s, k, t))
    lo, hi = np.full(price.shape, 0.01), np.full(price.shape, 6.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        above = bs_put(s, k, t, mid) > price
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    out = (lo + hi) / 2
    out[(price <= np.maximum(k - s, 0)) | (out > 5.9) | (out < 0.011)] = np.nan
    return out


def build_signals(close: pd.DataFrame, open_: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Trigger flags known at each day's close, the vol the put is priced on, and the two 'extended' flags.

    `close`/`open_` are raw (NaN where the name did not trade); nothing is forward-filled here.
    """
    lr = np.log(close).diff()
    r = close.pct_change(fill_method=None)
    sma50, sma10 = close.rolling(50, min_periods=50).mean(), close.rolling(10, min_periods=10).mean()
    sd63 = lr.rolling(63, min_periods=40).std()
    rv = np.maximum(lr.rolling(21, min_periods=15).std(), sd63) * np.sqrt(252)
    ext = {"pct": close / sma50 - 1 >= EXT_PCT, "z": np.log(close / sma50) / (sd63 * np.sqrt(25)) >= EXT_Z}
    hi20_prev = (close >= close.rolling(20, min_periods=20).max()).shift(1, fill_value=False)
    above10 = (close > sma10).astype(float)
    ran10 = above10.rolling(10).sum().shift(1) == 10
    sig = {}
    for tag, e in ext.items():
        e_prev = e.shift(1, fill_value=False)
        sig[f"red1_{tag}"] = hi20_prev & e_prev & (r < 0)
        sig[f"red3_{tag}"] = hi20_prev & e_prev & (r <= -0.03)
        sig[f"fade_{tag}"] = ran10 & (close < sma10) & e.rolling(10).max().shift(1).eq(1)
        sig[f"gap_{tag}"] = e_prev & (open_ / close.shift(1) - 1 <= -0.03) & (r < 0)
    return sig, rv, ext


# ---------------------------------------------------------------- B. real quotes
def quote_panel() -> pd.DataFrame:
    """One row per name per captured session: the ~2-week at-the-money put's closing bid/ask and its implied vol."""
    rows = []
    for d in sorted(IV_DIR.glob("2026*")):
        try:
            c = pd.read_parquet(d / "iv_contracts.parquet", columns=["symbol", "snapshot_date", "option_type", "dte", "strike",
                                                                     "bid", "ask", "spot", "open_interest", "non_standard",
                                                                     "volatility_pct"])
        except Exception:                      # three early captures wrote an empty file
            continue
        c = c[(c["option_type"] == "P") & c["dte"].between(7, 21) & ~c["non_standard"] & (c["bid"] > 0) & (c["ask"] > c["bid"])]
        c = c[(c["strike"] / c["spot"] - 1).abs() <= 0.05]
        if c.empty:
            continue
        oi = c.groupby(["symbol", "dte"])["open_interest"].sum().reset_index()
        oi["gap14"] = (oi["dte"] - 14).abs()
        best = oi.sort_values(["symbol", "open_interest", "gap14"], ascending=[True, False, True]).drop_duplicates("symbol")
        c = c.merge(best[["symbol", "dte"]], on=["symbol", "dte"])
        c["off"] = (c["strike"] / c["spot"] - 1).abs()
        rows.append(c.sort_values("off").drop_duplicates("symbol"))
    q = pd.concat(rows, ignore_index=True)
    q = q[q["off"] <= 0.03].copy()
    q["date"] = pd.to_datetime(q["snapshot_date"])
    q["mid"] = (q["bid"] + q["ask"]) / 2
    t = q["dte"].values / 365
    q["iv_mid"] = implied_vol(q["mid"].values, q["spot"].values, q["strike"].values, t)
    q["iv_ask"] = implied_vol(q["ask"].values, q["spot"].values, q["strike"].values, t)
    q["half_spread"] = (q["ask"] - q["mid"]) / q["mid"]
    return q.dropna(subset=["iv_mid", "iv_ask"])


def raw_matrices(symbols: list[str], tail: int = 140) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Recent raw daily open/close for the quoted names (the unadjusted live cache; names with a split-sized jump dropped)."""
    opens, closes, dropped = {}, {}, 0
    for s in symbols:
        f = RAW_1D / f"{s}.parquet"
        if not f.exists():
            continue
        b = pd.read_parquet(f, columns=["timestamp", "open", "close"]).tail(tail)
        idx = pd.to_datetime(b["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
        cl = pd.Series(b["close"].values, index=idx)
        if (np.log(cl).diff().abs() > 0.6).any():
            dropped += 1
            continue
        opens[s], closes[s] = pd.Series(b["open"].values, index=idx), cl
    return pd.DataFrame(opens).sort_index(), pd.DataFrame(closes).sort_index(), dropped


def quote_section(liquid: set[str], sleeve: set[str]) -> tuple[list[str], tuple[float, float], float, float, pd.DataFrame]:
    """Returns (report lines, (a, b) of implied = a + b x realized, half spread to charge, how much dearer after a
    gap, each name's own median implied/line and half spread)."""
    q = quote_panel()
    o_raw, c_raw, dropped = raw_matrices(sorted(q["symbol"].unique()))
    sig, rv, ext = build_signals(c_raw, o_raw)
    long = lambda df, name: df.stack().rename(name).rename_axis(["date", "symbol"]).reset_index()  # noqa: E731
    q = q.merge(long(rv, "rv"), on=["date", "symbol"]).dropna(subset=["rv"])
    day_r = c_raw.pct_change(fill_method=None)
    gap_day = (o_raw / c_raw.shift(1) - 1 <= -0.03) & (day_r < 0)        # no 'extended' condition: any name
    for name, df in {"ext_z": ext["z"], "ext_pct": ext["pct"], "gap_day": gap_day, "down3": day_r <= -0.03, **sig}.items():
        q = q.merge(long(df.astype(bool), name), on=["date", "symbol"], how="left")
        q[name] = q[name].fillna(False).astype(bool)
    q["trigger"] = q[[f"{t}_{e}" for t in TRIGGERS for e in ("pct", "z")]].any(axis=1)
    q["k_mid"], q["k_ask"] = q["iv_mid"] / q["rv"], q["iv_ask"] / q["rv"]
    ql = q[q["symbol"].isin(liquid)]
    lines = [f"\nB. Real closing quotes: the ~2-week at-the-money put ({q['date'].min().date()}..{q['date'].max().date()}, "
             f"{q['date'].nunique()} sessions, {q['symbol'].nunique()} names quoted; {dropped} dropped for a split-sized jump)",
             f"   {'group':44s} {'name-days':>9s} {'names':>6s} {'IV':>6s} {'realized':>8s} {'IV/realized mid':>15s} {'at the ask':>10s} "
             f"{'half spread':>11s} {'premium %spot':>13s}"]

    def row(label, s):
        if len(s) == 0:
            return f"   {label:44s} {'0':>9s}"
        return (f"   {label:44s} {len(s):9,d} {s['symbol'].nunique():6d} {s['iv_mid'].median():6.0%} {s['rv'].median():8.0%} "
                f"{s['k_mid'].median():15.2f} {s['k_ask'].median():10.2f} {s['half_spread'].median():11.1%} "
                f"{(s['mid'] / s['spot']).median():13.2%}")

    lines.append(row("all quoted names", q))
    lines.append(row("liquid-300", ql))
    for lo, hi in ((0, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 9)):
        lines.append(row(f"  liquid-300, realized vol {lo:.0%}-{hi:.0%}" if hi < 9 else f"  liquid-300, realized vol over {lo:.0%}",
                         ql[(ql["rv"] >= lo) & (ql["rv"] < hi)]))
    lines.append(row("liquid-300, extended yesterday (either def.)", ql[ql["ext_z"] | ql["ext_pct"]]))
    lines.append(row("liquid-300, a trigger fired today", ql[ql["trigger"]]))
    lines.append(row("the 20 sleeve names (latest decision)", q[q["symbol"].isin(sleeve)]))
    # is the put dearer on the trigger day than the same name's put the session before?
    q = q.sort_values(["symbol", "date"])
    q["iv_prev"] = q.groupby("symbol")["iv_mid"].shift(1)
    q["gap_days"] = q.groupby("symbol")["date"].diff().dt.days
    chg = q[(q["gap_days"] <= 4) & q["iv_prev"].notna() & q["symbol"].isin(liquid)].assign(d=lambda x: x["iv_mid"] / x["iv_prev"] - 1)
    tr, ot = chg[chg["trigger"]]["d"], chg[~chg["trigger"]]["d"]
    lines.append(f"   implied vol against the same name's previous capture: trigger days {tr.median():+.1%} (n={len(tr)}), "
                 f"other days {ot.median():+.1%} (n={len(ot):,})")
    fit = ql[(ql["rv"] <= 1.5) & (ql["iv_mid"] <= 2.0)]
    b, a = np.polyfit(fit["rv"], fit["iv_mid"], 1)
    resid = fit["iv_mid"] - (a + b * fit["rv"])
    target = ql[ql["ext_z"] | ql["ext_pct"]]
    miss = (a + b * target["rv"]) / target["iv_mid"] - 1
    lines.append(f"   fitted on the liquid-300 name-days: implied = {a:.3f} + {b:.3f} x realized (R2 {1 - resid.var() / fit['iv_mid'].var():.2f}); "
                 f"on the extended names it is off by {miss.median():+.1%} (median), {miss.abs().median():.1%} (median absolute)")
    feed = q[q["volatility_pct"] > 0]
    lines.append(f"   check on the inversion: this script's implied vol / the feed's own figure, median {(feed['iv_mid'] / (feed['volatility_pct'] / 100)).median():.3f} "
                 f"(n={len(feed):,})")
    # is the put dearer than the line on the day the trigger fires? (all quoted names, for sample size)
    q["vs_line"] = q["iv_mid"] / (a + b * q["rv"]) - 1
    chg_all = q[(q["gap_days"] <= 4) & q["iv_prev"].notna()].assign(d=lambda x: x["iv_mid"] / x["iv_prev"] - 1)
    lines.append("   quoted implied vol against the fitted line, and against the same name's previous capture, on the day itself (all quoted names):")
    dear = {}
    for label, col_ in (("opened 3%+ lower and closed down", "gap_day"), ("closed down 3%+", "down3"), ("a trigger fired", "trigger")):
        on, off = q[q[col_]], q[~q[col_]]
        c_on = chg_all[chg_all[col_]]["d"]
        dear[col_] = (1 + on["vs_line"].median()) / (1 + off["vs_line"].median()) - 1
        lines.append(f"     {label:34s} {on['vs_line'].median():+6.1%} vs the line (other days {off['vs_line'].median():+.1%}), n={len(on):,} | "
                     f"{c_on.median():+6.1%} vs the previous capture, n={len(c_on):,}")
    own = q.assign(mult=1 + q["vs_line"]).groupby("symbol")[["mult", "half_spread"]].median()
    lines.append(f"   one name's own implied vol / the line (median per name): 25th pct {own['mult'].quantile(0.25):.2f}, median "
                 f"{own['mult'].median():.2f}, 75th pct {own['mult'].quantile(0.75):.2f} (the +-20% rows in D cover this)")
    return lines, (float(a), float(b)), float(target["half_spread"].median()), float(dear["gap_day"]), own


# ---------------------------------------------------------------- A / C helpers
def per_date_ci(event: pd.Series, control: pd.Series) -> tuple[float, float, float]:
    d = (event - control).dropna()
    lo, hi = an.wf.block_ci(d)
    return float(d.mean()), float(lo), float(hi)


def payback_ci(df: pd.DataFrame, pay: str, prem: str, n: int = 2000) -> tuple[float, float]:
    g = df.groupby("date")[[pay, prem]].sum()
    s = g.groupby(g.index.year * 2 + (g.index.month > 6)).sum().values
    idx = np.random.default_rng(SEED).integers(0, len(s), (n, len(s)))
    b = s[idx].sum(axis=1)
    return tuple(np.percentile(b[:, 0] / b[:, 1], [2.5, 97.5]))


def breakeven_mult(fwd: np.ndarray, rv: np.ndarray, t: float, strike: float = 1.0) -> float:
    """Vol multiple at which the summed premium equals the summed payoff."""
    pay = np.maximum(strike - (1 + fwd), 0).sum()
    lo, hi = 0.05, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if bs_put(1.0, strike, t, mid * rv).sum() < pay else (lo, mid)
    return (lo + hi) / 2


def portfolio_section(ctx: dict, sig: dict, ext: dict, rv: pd.DataFrame, iv_of, half_spread: float) -> list[str]:
    """D. The fixed 70/30 blend plus a put on each held name when its trigger fires."""
    p, cal, col, O, C, end, dates = ctx["p"], ctx["cal"], ctx["col"], ctx["O"], ctx["C"], ctx["end"], ctx["dates"]
    names = list(ctx["raw_close"].columns)
    nm, spy = {t: i for i, t in enumerate(names)}, col["SPY"]
    hold = {d: g.sort_values("mom_12_1", ascending=False)["ticker"].head(N_TOP).tolist() for d, g in p.groupby("date")}
    sigma = iv_of(rv.values)
    T = {k: v.values for k, v in sig.items()}
    any_pct = T["red1_pct"] | T["fade_pct"] | T["gap_pct"]
    arms = {"p_red1": dict(trig=T["red1_pct"]), "p_red3": dict(trig=T["red3_pct"]), "p_fade": dict(trig=T["fade_pct"]),
            "p_gap": dict(trig=T["gap_pct"]), "p_any": dict(trig=any_pct),
            "p_any_z": dict(trig=T["red1_z"] | T["fade_z"] | T["gap_z"]),
            "p_extended": dict(trig=ext["pct"].values)}
    sens = {"p_any_cheap": dict(trig=any_pct, vol_mult=0.8), "p_any_dear": dict(trig=any_pct, vol_mult=1.2),
            "p_any_otm5": dict(trig=any_pct, strike=0.95), "p_any_20d": dict(trig=any_pct, days=20)}
    label = {"p_red1": "first red day off the high", "p_red3": "the same, down 3%+", "p_fade": "first close below the 10-day avg",
             "p_gap": "gap down 3%+ (bad-news proxy)", "p_any": "any of the three", "p_any_z": "any, extended in vol units",
             "p_extended": "no trigger: put whenever extended", "p_any_cheap": "any, vol 20% lower", "p_any_dear": "any, vol 20% higher",
             "p_any_otm5": "any, 5% out of the money", "p_any_20d": "any, 20-session put"}

    def overlay(ds, entries, base, trig, vol_mult=1.0, strike=1.0, days=PUT_DAYS):
        """Daily put P&L in the unhedged blend's own value units, and one record per put bought."""
        pnl, puts, busy = np.zeros(end + 1), [], {}
        shares = []                                   # per period: {name: shares held}
        for d, e in zip(ds, entries):
            idx = np.array([col[t] for t in hold[d]])
            rel = (1 - SHARE) * C[e, spy] / O[e, spy] + SHARE * np.mean(C[e, idx] / O[e, idx])
            v0 = base[e - entries[0]] / rel           # blend value at this period's entry open, after costs
            shares.append({t: v0 * SHARE / N_TOP / O[e, col[t]] for t in hold[d]})
        for k, (d, e) in enumerate(zip(ds, entries)):
            nxt = entries[k + 1] if k + 1 < len(entries) else None
            last = nxt - 1 if nxt is not None else end
            for t_ in hold[d]:
                j, jn = col[t_], nm[t_]
                for t in np.flatnonzero(trig[e:last + 1, jn]) + e:
                    x = t + 1
                    if x > end or x <= busy.get(t_, -1):
                        continue
                    if x > last and t_ not in shares[k + 1]:      # sold at the rebalance: nothing left to protect
                        continue
                    n_sh = shares[k][t_] if x <= last else shares[k + 1][t_]
                    s0, sg = O[x, j], sigma[t, jn] * vol_mult
                    if not (np.isfinite(sg) and np.isfinite(s0) and s0 > 0):
                        continue
                    xe, stop = x + days - 1, min(x + days - 1, end)
                    prem = float(bs_put(s0, s0 * strike, days / 252, sg)) * (1 + half_spread)
                    px = bs_put(C[x:stop + 1, j], s0 * strike, (xe - np.arange(x, stop + 1)) / 252, sg)
                    if stop == xe:
                        px[-1] = max(s0 * strike - C[xe, j], 0.0) * (1 - half_spread)
                    pnl[x:stop + 1] += n_sh * np.diff(np.r_[prem, px])
                    busy[t_] = xe
                    puts.append((x, n_sh * prem / base[x - 1 - entries[0]] if x > entries[0] else n_sh * prem,
                                 n_sh * px[-1] / base[stop - entries[0]]))
        return pnl, puts

    rets = {a: [] for a in ["b_base", *arms, *sens]}
    stats = {a: [] for a in rets}
    spend = {a: [] for a in rets}
    last12, case = {}, []
    for ph in range(STEP):
        ds = dates[ph::STEP]
        entries = [i for i in (cal.searchsorted(d, side="right") for d in ds) if i <= end]
        ds = ds[:len(entries)]
        idx = cal[entries[0]:]
        b = rb.simulate([{"SPY": 1.0}], entries[:1], O, C, col, end)[0]
        base = rb.simulate([{"SPY": 1 - SHARE, **{t: SHARE / N_TOP for t in hold[d]}} for d in ds], entries, O, C, col, end)[0]
        r_base = np.diff(np.r_[1.0, base]) / np.r_[1.0, base][:-1]
        curves, logs = {"b_base": base}, {}
        for a, kw in {**arms, **sens}.items():
            pnl, puts = overlay(ds, entries, base, **kw)
            curves[a] = np.cumprod(1 + r_base + pnl[entries[0]:] / np.r_[1.0, base][:-1])
            logs[a] = (pnl, puts)
            yrs = len(base) / 252
            spend[a].append((len(puts) / yrs, sum(x[1] for x in puts) / yrs, sum(x[2] for x in puts) / yrs))
        for a, v in curves.items():
            stats[a].append(rb.stats(v, b, idx))
            rets[a].append(pd.Series(np.diff(np.r_[1.0, v]) / np.r_[1.0, v][:-1], index=idx))
        if ctx["live"][0] in ds:
            for a, v in {**curves, "spy": b}.items():
                w = pd.Series(v, index=idx).loc["2025-10-03":]
                last12[a] = (w.iloc[-1] / w.iloc[0] - 1, (w / w.cummax() - 1).min())
            pnl, puts = logs["p_any"]
            for k, (d, e) in enumerate(zip(ds, entries)):
                if d < pd.Timestamp("2025-10-01"):
                    continue
                nxt = entries[k + 1] if k + 1 < len(entries) else None
                last = nxt - 1 if nxt is not None else end
                ii = np.array([col[t] for t in hold[d]])
                sleeve = np.mean((O[nxt, ii] if nxt is not None else C[end, ii]) / O[e, ii]) - 1
                mine = [x for x in puts if e <= x[0] <= last]
                net = float((pnl[e:last + 1] / base[e - 1 - entries[0]:last - entries[0]]).sum())
                case.append(f"   {d.date()}  sleeve {sleeve:+6.1%}  puts bought {len(mine):3d}  premium {sum(x[1] for x in mine):5.2%} of the account  "
                            f"puts' net result {net:+6.2%} of the account  (= {net / SHARE:+5.1%} of the sleeve)")

    base_r = pd.concat(rets["b_base"], axis=1)
    lines = [f"\nD. The fixed 70/30 blend plus puts on the held names (MODEL premiums; {half_spread:.1%} of premium each way) | monthly, "
             f"4 offsets | {dates.min().date()}..{cal[end].date()}",
             f"   {'put on a held name when':34s} {'CAGR med [min, max]':>24s} {'Sharpe':>6s} {'MDD med/worst':>14s} {'vs fixed 70/30 /yr [95% CI]':>30s} "
             f"{'puts/yr':>7s} {'premium/yr':>10s} {'payoff/yr':>9s} {'last 12m':>8s} {'12m MDD':>7s}"]
    for a in rets:
        df = pd.DataFrame(stats[a])
        pair = (pd.concat(rets[a], axis=1) - base_r).dropna().mean(axis=1)
        lo, hi = rb.block_ci(pair) if a != "b_base" else (0.0, 0.0)
        n, pr, po = np.mean(spend[a], axis=0) if a != "b_base" else (0.0, 0.0, 0.0)
        if a == next(iter(sens)):
            lines.append("   sensitivity:")
        lines.append(f"   {label.get(a, 'none (fixed 70/30)'):34s} {df.cagr.median():+7.1%} [{df.cagr.min():+6.1%},{df.cagr.max():+6.1%}] "
                     f"{df.sharpe.median():6.2f} {df.mdd.median():+6.1%}/{df.mdd.min():+6.1%} {pair.mean() * 252:+11.1%} [{lo:+6.1%},{hi:+6.1%}] "
                     f"{n:7.0f} {pr:10.1%} {po:9.1%} {last12[a][0]:+8.1%} {last12[a][1]:+7.1%}")
    lines.append("   'premium/yr' and 'payoff/yr' are shares of the account. Part 9 reference (theme-ETF puts every period, model): "
                 "-2.3%/yr, worst MDD -34% -> -28.5%.")
    lines.append("   the last 13 periods on the live schedule, arm 'any of the three':")
    return lines + case


def main() -> None:
    ctx = an.build_context()
    p, cal, col, O, C, end, dates = ctx["p"], ctx["cal"], ctx["col"], ctx["O"], ctx["C"], ctx["end"], ctx["dates"]
    names = list(ctx["raw_close"].columns)
    c_raw = ctx["raw_close"]
    o_raw = pd.read_parquet(pb.PIT_OUT.parent / "px_open_pit.parquet").reindex(index=cal, columns=names)
    ji = [col[t] for t in names]
    c_fill, o_fill = pd.DataFrame(C[:, ji], index=cal, columns=names), pd.DataFrame(O[:, ji], index=cal, columns=names)
    sig, rv, ext = build_signals(c_raw, o_raw)

    def as_of(rows: pd.DataFrame) -> pd.DataFrame:
        """Membership known before each session: the latest weekly decision strictly before it."""
        m = rows.assign(v=True).pivot(index="date", columns="ticker", values="v").reindex(index=dates, columns=names)
        return m.fillna(False).astype(bool).reindex(cal, method="ffill", limit=7).shift(1).fillna(False).astype(bool)

    member, held = as_of(p), as_of(p[p["picked"]])
    tradable = c_raw.notna() & c_raw.shift(-1).notna() & member
    fwd = {h: c_fill.shift(-h) / o_fill.shift(-1) - 1 for h in HORIZONS}
    fwd_close = c_fill.shift(-PUT_DAYS) / c_raw - 1          # favourable: bought on the trigger day's own close

    def frame(mask: pd.DataFrame) -> pd.DataFrame:
        m = (mask & tradable).stack()
        ix = m[m].index
        out = pd.DataFrame({"date": ix.get_level_values(0), "ticker": ix.get_level_values(1)})
        di, tj = cal.get_indexer(out["date"]), [names.index(t) for t in out["ticker"]]
        for h in HORIZONS:
            out[f"f{h}"] = fwd[h].values[di, tj]
        out["fc"], out["rv"], out["held"] = fwd_close.values[di, tj], rv.values[di, tj], held.values[di, tj]
        return out.dropna(subset=[f"f{PUT_DAYS}", "rv"])

    everything = frame(pd.DataFrame(True, index=cal, columns=names))
    controls = {"every name-day": everything,
                "extended yesterday (pct)": frame(ext["pct"].shift(1, fill_value=False)),
                "extended yesterday (z)": frame(ext["z"].shift(1, fill_value=False))}
    events = {k: frame(v) for k, v in sig.items()}
    # a gap is "earnings" when a report is dated that day or the session before (the calendar has no time of day)
    ev = pd.read_parquet(pb.CAL, columns=["ticker", "date"]).dropna()
    rep_day = pd.DataFrame(False, index=cal, columns=names)
    ok = ev["ticker"].isin(names) & ev["date"].isin(cal)
    rep_day.values[cal.get_indexer(ev.loc[ok, "date"]), [names.index(t) for t in ev.loc[ok, "ticker"]]] = True
    near_report = rep_day | rep_day.shift(1, fill_value=False)
    for tag in ("pct", "z"):
        events[f"gap_earn_{tag}"] = frame(sig[f"gap_{tag}"] & near_report)
        events[f"gap_news_{tag}"] = frame(sig[f"gap_{tag}"] & ~near_report)
    yrs = (cal[end] - dates.min()).days / 365.25

    lines = [f"=== puts on single names: first red day after an extended run | liquid-300, point-in-time | "
             f"{dates.min().date()}..{cal[end].date()} | signal on the close, entry next open ==="]
    lines.append("\nA. What the SHARES do next (no option model). '10d' = close of the 10th session / entry open - 1")
    lines.append(f"   {'set':26s} {'events':>8s} {'/yr':>6s} {'vol':>5s} {'3d mean':>8s} {'5d mean':>8s} {'10d mean':>8s} {'10d med':>8s} "
                 f"{'down':>5s} {'<-5%':>5s} {'<-10%':>6s} {'>+5%':>5s} {'>+10%':>6s}   10d vs extended days [95% CI]")
    ctrl_by_date = {k: v.groupby("date")[f"f{PUT_DAYS}"].mean() for k, v in controls.items()}

    def a_row(label: str, s: pd.DataFrame, versus: str | None) -> str:
        f = s[f"f{PUT_DAYS}"]
        tail = ""
        if versus is not None:
            m, lo, hi = per_date_ci(s.groupby("date")[f"f{PUT_DAYS}"].mean(), ctrl_by_date[versus])
            tail = f"   {m:+.2%} [{lo:+.2%}, {hi:+.2%}]"
        return (f"   {label:26s} {len(s):8,d} {len(s) / yrs:6.0f} {s['rv'].median():5.0%} {s['f3'].mean():+8.2%} {s['f5'].mean():+8.2%} "
                f"{f.mean():+8.2%} {f.median():+8.2%} {(f < 0).mean():5.0%} {(f < -0.05).mean():5.0%} {(f < -0.10).mean():6.0%} "
                f"{(f > 0.05).mean():5.0%} {(f > 0.10).mean():6.0%}{tail}")

    for k, v in controls.items():
        lines.append(a_row(k, v, None))
    for tag in ("pct", "z"):
        for t in TRIGGERS + ["gap_earn", "gap_news"]:
            lines.append(a_row(f"{t} ({tag})", events[f"{t}_{tag}"], f"extended yesterday ({tag})"))
    lines.append("   gap_earn = the gap came with an earnings report (dated that day or the one before); gap_news = it did not")
    lines.append("   the same triggers on names the sleeve held that week (top-20 momentum):")
    for tag in ("pct", "z"):
        for t in TRIGGERS:
            s = events[f"{t}_{tag}"]
            lines.append(a_row(f"{t} ({tag}) held", s[s["held"]], f"extended yesterday ({tag})"))

    liquid_now = set(p[p["date"] == dates.max()]["ticker"])
    sleeve_now = set(p[(p["date"] == dates.max()) & p["picked"]]["ticker"])
    q_lines, (iv_a, iv_b), half_spread, gap_dear, own = quote_section(liquid_now, sleeve_now)
    lines += q_lines
    lines.append(f"   -> the model below prices puts at that line (realized = max of 21d and 63d) and charges {half_spread:.1%} of the "
                 f"premium each way (extended liquid-300 names)")
    iv_of = lambda x: iv_a + iv_b * np.asarray(x, float)  # noqa: E731

    # ---- C. put economics on the events
    t_put = PUT_DAYS / 252
    lines.append(f"\nC. A {PUT_DAYS}-session at-the-money put bought at the next open (MODEL premium at the fitted implied vol, payoff from "
                 "the shares). Payback = payoff per $1 of premium before the bid/ask; 1.00 = fairly priced")
    lines.append(f"   {'set':26s} {'premium':>8s} {'payoff':>7s} {'payback [95% CI]':>24s} {'after bid/ask':>13s} {'paid off':>8s} "
                 f"{'5% OTM payback':>14s} {'break-even x quoted vol':>23s} {'if bought at the close':>22s}")

    def priced(s: pd.DataFrame, vol_mult: float = 1.0) -> pd.DataFrame:
        return s.assign(prem=bs_put(1.0, 1.0, t_put, vol_mult * iv_of(s["rv"])), pay=np.maximum(-s[f"f{PUT_DAYS}"].values, 0))

    def c_row(label: str, s: pd.DataFrame) -> str:
        s = priced(s)
        lo, hi = payback_ci(s, "pay", "prem")
        otm = np.maximum(0.95 - (1 + s[f"f{PUT_DAYS}"].values), 0).sum() / bs_put(1.0, 0.95, t_put, iv_of(s["rv"])).sum()
        be = breakeven_mult(s[f"f{PUT_DAYS}"].values, iv_of(s["rv"]), t_put)
        sc = s.dropna(subset=["fc"])
        at_close = np.maximum(-sc["fc"].values, 0).sum() / bs_put(1.0, 1.0, t_put, iv_of(sc["rv"])).sum()
        pb_ = s["pay"].sum() / s["prem"].sum()
        return (f"   {label:26s} {s['prem'].mean():8.2%} {s['pay'].mean():7.2%} {pb_:8.2f} [{lo:.2f}, {hi:.2f}] "
                f"{pb_ * (1 - half_spread) / (1 + half_spread):13.2f} {(s['pay'] > s['prem']).mean():8.0%} {otm:14.2f} {be:23.2f} {at_close:22.2f}")

    for k, v in controls.items():
        lines.append(c_row(k, v))
    for tag in ("pct", "z"):
        for t in TRIGGERS + ["gap_earn", "gap_news"]:
            lines.append(c_row(f"{t} ({tag})", events[f"{t}_{tag}"]))
    any_held = {}
    for tag in ("pct", "z"):
        s = pd.concat([events[f"{t}_{tag}"] for t in TRIGGERS]).drop_duplicates(["date", "ticker"])
        any_held[tag] = s[s["held"]]
        lines.append(c_row(f"any trigger ({tag}) held", any_held[tag]))
    lines.append(f"   gap rows repriced at the vol quoted after a gap-down day (the line x {1 + gap_dear:.2f}, section B):")
    for k in ("gap_pct", "gap_news_pct", "gap_z", "gap_news_z"):
        g = priced(events[k], 1 + gap_dear)
        lo, hi = payback_ci(g, "pay", "prem")
        pb_ = g["pay"].sum() / g["prem"].sum()
        lines.append(f"   {k.rsplit('_', 1)[0] + ' (' + k.rsplit('_', 1)[1] + ')':26s} {g['prem'].mean():8.2%} {g['pay'].mean():7.2%} {pb_:8.2f} [{lo:.2f}, {hi:.2f}] "
                     f"{pb_ * (1 - half_spread) / (1 + half_spread):13.2f} {(g['pay'] > g['prem']).mean():8.0%}")
    lines.append("   payback by calendar year (before the bid/ask):")
    by_year = {"every name-day": everything, "red1 (pct)": events["red1_pct"], "gap (pct)": events["gap_pct"],
               "gap_news (pct)": events["gap_news_pct"], "any trigger (pct) held": any_held["pct"]}
    years = sorted(everything["date"].dt.year.unique())
    lines.append(f"   {'':26s} " + " ".join(f"{y:>6d}" for y in years))
    for k, v in by_year.items():
        g = priced(v).groupby(v["date"].dt.year)[["pay", "prem"]].sum()
        lines.append(f"   {k:26s} " + " ".join(f"{g.at[y, 'pay'] / g.at[y, 'prem']:6.2f}" if y in g.index else f"{'-':>6s}" for y in years))

    lines += portfolio_section(ctx, sig, ext, rv, iv_of, half_spread)

    # ---- E. the chart the question came with
    t0 = "NVDA"
    cl = c_raw[t0]
    first_red = (cl >= cl.rolling(20).max()).shift(1, fill_value=False) & (cl.pct_change(fill_method=None) < 0)
    rows = [d for d in cl.index[first_red.values] if d >= cal[end] - pd.Timedelta(days=365)]
    sma50 = cl.rolling(50).mean()
    mult0, hs0 = (own.at[t0, "mult"], own.at[t0, "half_spread"]) if t0 in own.index else (1.0, half_spread)
    lines.append(f"\nE. {t0}, last 12 months: every first red day off a 20-day closing high (no 'extended' filter), entry next open. "
                 f"Put priced at {t0}'s own quoted level ({mult0:.2f} x the line, {hs0:.1%} half spread)")
    lines.append(f"   {'red day':10s} {'that day':>8s} {'vs 50d avg':>10s} {'3d':>7s} {'5d':>7s} {'10d':>7s} {'low within 10d':>14s} "
                 f"{'put cost (model)':>16s} {'put paid':>8s}")
    got = []
    for d in rows:
        i = cal.get_loc(d)
        if i + PUT_DAYS > end:
            continue
        e0 = o_fill[t0].values[i + 1]
        f3, f5, f10 = (c_fill[t0].values[i + h] / e0 - 1 for h in HORIZONS)
        low = c_fill[t0].values[i + 1:i + PUT_DAYS + 1].min() / e0 - 1
        prem = float(bs_put(1.0, 1.0, t_put, mult0 * iv_of(rv[t0].values[i]))) * (1 + hs0)
        got.append((f5, f10, prem, max(-f10, 0.0)))
        lines.append(f"   {d.date()!s:10s} {cl.at[d] / cl.iloc[i - 1] - 1:+8.1%} {cl.iloc[i - 1] / sma50.iloc[i - 1] - 1:+10.1%} {f3:+7.1%} {f5:+7.1%} "
                     f"{f10:+7.1%} {low:+14.1%} {prem:16.2%} {max(-f10, 0.0):8.2%}")
    if got:
        g = np.array(got)
        lines.append(f"   {len(g)} occasions: lower 5 sessions later in {(g[:, 0] < 0).sum()}, lower 10 sessions later in {(g[:, 1] < 0).sum()}; "
                     f"mean 5d {g[:, 0].mean():+.1%}, 10d {g[:, 1].mean():+.1%} | puts: paid {g[:, 2].mean():.2%} of the share price each time, "
                     f"got back {g[:, 3].mean():.2%}; the put made money on {(g[:, 3] > g[:, 2]).sum()} of {len(g)}")

    txt = "\n".join(lines)
    print(txt)
    (HERE / "21_results.txt").write_text(txt)


if __name__ == "__main__":
    main()
