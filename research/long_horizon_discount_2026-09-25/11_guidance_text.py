"""Guidance language in real earnings press releases -> drift over the next quarter (2026-10-03).

Part 3's variant (c). 07 tested EPS beat/miss x price reaction. This script tests what management
SAID about the future, read from the 8-K Item 2.02 Exhibit-99 text (sec_earnings_releases, 70k
releases, each with an exact SEC acceptance time). The user's hypothesis is the "raised + stock fell"
cell: good guidance that the market sold.

Labels (regex, interpretable baseline; snippets are printed so precision can be eyeballed):
  raised      a raise/increase verb within 80 characters of guidance|outlook|forecast in one sentence
  lowered     a lower/reduce/cut verb with the same objects, or guidance withdrawn/suspended
  mixed       both;  reaffirmed  reaffirm/reiterate/maintain only;  none  no directional guidance language
Timing: r0 = the first session that opens at or after the acceptance time. Reaction = close(r0) /
  close(r0-2) - 1 minus SPY. The two-session window also covers an 8-K accepted mid-session.
  Entry = open(r0+1). Exit = close(r0+H), for H = 21 / 63 / 126. Excess is vs SPY over the same window.
Adjusted = excess minus the same-month mean of ALL liquid-1000 releases. CIs bootstrap release months.
Sanity check before any drift number: a "raised" label must come with a better announcement
reaction than a "lowered" one. If it does not, the labels do not measure guidance.

    .venv/bin/python research/long_horizon_discount_2026-09-25/11_guidance_text.py
"""
from __future__ import annotations

import gzip
import re
from importlib.util import module_from_spec, spec_from_file_location
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = spec_from_file_location("er", HERE / "07_earnings_reaction.py")
er = module_from_spec(_spec)
_spec.loader.exec_module(er)
rb = er.rb

RAW = rb.pb.REPO / "Data" / "raw" / "sec_earnings_releases"
FLAGS = rb.pb.OUT.parent / "ex99_guidance_flags.parquet"   # derived: one row per release
HOLDS = (21, 63, 126)
OBJ = r"(?:guidance|outlook|forecasts?)"
UP = r"(?:rais(?:es|ed|ing)|increas(?:es|ed|ing)|lift(?:s|ed|ing)|boost(?:s|ed|ing)|upward(?:ly)? revis\w+|revis\w+ upward)"
DOWN = r"(?:lower(?:s|ed|ing)|reduc(?:es|ed|ing)|cut(?:s|ting)?|decreas(?:es|ed|ing)|downward(?:ly)? revis\w+|revis\w+ downward)"
GAP = r"[^.;:\n]{0,80}?"
RE = {
    "up": re.compile(rf"\b{UP}\b{GAP}\b{OBJ}\b|\b{OBJ}\b{GAP}\b{UP}\b", re.I),
    "down": re.compile(rf"\b{DOWN}\b{GAP}\b{OBJ}\b|\b{OBJ}\b{GAP}\b{DOWN}\b"
                       rf"|\b(?:withdr[ae]w\w*|suspend\w*)\b{GAP}\b{OBJ}\b", re.I),
    "same": re.compile(rf"\b(?:reaffirm\w*|reiterat\w*|maintain\w*|confirm\w*)\b{GAP}\b{OBJ}\b", re.I),
}


def scan(row: tuple[str, str]) -> dict:
    ticker, acc = row
    try:
        with gzip.open(RAW / ticker / f"{acc}.txt.gz", "rt", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
    except OSError:
        return {"ticker": ticker, "accession": acc, "read_ok": False}
    out = {"ticker": ticker, "accession": acc, "read_ok": True, "chars": len(text)}
    for k, rx in RE.items():
        hits = [m.group(0) for m in rx.finditer(text)]
        out[f"n_{k}"] = len(hits)
        out[f"ex_{k}"] = " ".join(hits[0].split())[:160] if hits else ""
    return out


def build_flags() -> pd.DataFrame:
    if FLAGS.exists():
        return pd.read_parquet(FLAGS)
    m = pd.read_parquet(RAW / "_manifest.parquet")
    m = m[m["status"] == "ok"].drop_duplicates(["ticker", "accession"])
    with Pool(8) as pool:
        rows = pool.map(scan, list(zip(m["ticker"], m["accession"])), chunksize=200)
    f = m[["ticker", "accession", "acceptance_ts", "form"]].merge(pd.DataFrame(rows), on=["ticker", "accession"])
    f = f[f["read_ok"]].copy()
    up, down, same = f["n_up"] > 0, f["n_down"] > 0, f["n_same"] > 0
    f["label"] = np.select([up & down, up, down, same], ["mixed", "raised", "lowered", "reaffirmed"], "none")
    f.to_parquet(FLAGS, index=False)
    return f


def build_events(f: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet(rb.pb.OUT, columns=["date", "ticker", "dv_rank"])
    O, C = rb.price_matrices(sorted(set(p["ticker"]) | {"SPY"}))
    cal, col = C.index, {t: j for j, t in enumerate(C.columns)}
    ev = f[f["ticker"].isin(C.columns) & f["acceptance_ts"].notna()].copy()
    et = ev["acceptance_ts"].dt.tz_convert("America/New_York")
    day = et.dt.normalize().dt.tz_localize(None)
    before_open = (et.dt.hour * 60 + et.dt.minute) <= 9 * 60 + 30
    # r0: the same session if accepted by the open on a session day, otherwise the next session
    i_same = cal.searchsorted(day.values, side="left")
    is_session = (i_same < len(cal)) & (cal[np.minimum(i_same, len(cal) - 1)] == day.values)
    r0 = np.where(is_session & before_open.values, i_same, cal.searchsorted(day.values, side="right"))
    ok = (r0 >= 2) & (r0 + 1 < len(cal))
    ev, r0 = ev[ok].copy(), r0[ok]
    ev["date"] = cal[r0]
    ev = ev.sort_values("date")
    r0 = cal.searchsorted(ev["date"].values)
    ev = pd.merge_asof(ev, p.sort_values("date"), on="date", by="ticker", direction="backward",
                       tolerance=pd.Timedelta(days=7))
    keep = ev["dv_rank"].notna().values
    ev, r0 = ev[keep].copy(), r0[keep]
    j, s = ev["ticker"].map(col).values, col["SPY"]
    Cv, Ov = C.values, O.values
    ev["react"] = (Cv[r0, j] / Cv[r0 - 2, j] - Cv[r0, s] / Cv[r0 - 2, s]) * 100
    for h in HOLDS:
        ex = r0 + h
        valid = ex < len(cal)
        exi = np.where(valid, ex, 0)
        ev[f"xs_{h}"] = np.where(valid, (Cv[exi, j] / Ov[r0 + 1, j]) - (Cv[exi, s] / Ov[r0 + 1, s]), np.nan)
    ev["r_grp"] = np.select([ev.react > er.BAND, ev.react < -er.BAND], ["up", "down"], "flat")
    ev["month"] = ev["date"].dt.to_period("M")
    # one release per ticker per 5 sessions (drops 8-K/A re-files and same-week duplicates)
    ev = ev.sort_values(["ticker", "date"])
    ev = ev[ev.groupby("ticker")["date"].diff().dt.days.fillna(99) > 7]
    return ev.dropna(subset=["react"])


def main() -> None:
    f = build_flags()
    lines = [f"=== guidance language | {len(f):,} press releases read, {f.ticker.nunique()} tickers | "
             f"labels: {f.label.value_counts(normalize=True).round(3).to_dict()} ==="]
    rng = np.random.default_rng(er.SEED)
    for lab, c in (("raised", "ex_up"), ("lowered", "ex_down"), ("reaffirmed", "ex_same")):
        lines.append(f"\nsample '{lab}' matches:")
        sub = f[f["label"] == lab]
        for i in rng.choice(len(sub), 6, replace=False):
            r = sub.iloc[i]
            lines.append(f"  {r.ticker:6s} {str(r.acceptance_ts)[:10]}  \"{r[c]}\"")
    ev = build_events(f)
    lines.append(f"\n{len(ev):,} liquid-1000 release events {ev.date.min().date()}..{ev.date.max().date()}; "
                 f"reaction up {(ev.r_grp == 'up').mean():.0%} / flat {(ev.r_grp == 'flat').mean():.0%} / "
                 f"down {(ev.r_grp == 'down').mean():.0%}")
    sc = ev.groupby("label")["react"].agg(["size", "mean", "median", lambda x: (x > 0).mean()])
    sc.columns = ["n", "mean_react_%", "median_react_%", "share_up"]
    lines.append("\nSANITY: announcement reaction by label (raised should beat lowered):\n" + sc.round(2).to_string())
    for h in HOLDS:
        lines.append(f"\n--- hold {h}: guidance label (all reactions) ---")
        lines.append(er.fmt(er.group_table(ev, ["label"], h)))
    for h in (63, 126):
        lines.append(f"\n--- hold {h}: guidance label x reaction ---")
        lines.append(er.fmt(er.group_table(ev[ev.label.isin(["raised", "lowered", "none"])], ["label", "r_grp"], h)))
    ev["yr"] = ev["date"].dt.year
    t = er.group_table(ev[ev.label == "raised"], ["r_grp", "yr"], 63)
    lines.append("\n--- hold 63: RAISED guidance by reaction and year (stability) ---")
    lines.append(t.pivot(index="yr", columns="r_grp", values="adj").map(lambda v: f"{v:+.1%}").to_string())
    txt = "\n".join(lines)
    print(txt)
    (HERE / "11_results.txt").write_text(txt)


if __name__ == "__main__":
    main()
