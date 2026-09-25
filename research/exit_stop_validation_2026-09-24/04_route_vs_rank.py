"""Step 4 -- does the OPTION book hold the same names as the TOP of the ranking?

Stage 2 measured that momentum's top-3 are median $0.85M daily dollar volume, ATR 11.18%
(3.1x universe), beta 2.05. The LIVE gates are `options_exec.ROUTE_MIN_OPEN_INTEREST = 500`
and `ROUTE_MIN_VOLUME = 100`, enforced in `route_option_or_shares`, and a name that FAILS
them is routed to SHARES rather than dropped. Hypothesis:

    the option book is systematically populated by the MORE LIQUID, LOWER-RANKED names
    while the illiquid top picks go to equity

which would be the wrong way round, since within momentum the SAME signals returned equity
+5.44% and options -31.50% (-36.93pp, p=0.006).

The right record for this is the `contract_selection` block in `live_signal_audit.jsonl`: one
entry per (bar, ticker) carrying `action` ("equity"/"option"), a `reason` that names the
routing cause verbatim (e.g. `illiquid_option(oi=0,vol=0)`), and the nested `signal_audit`
with that name's RANK at the time. That is the decision itself, not a reconstruction.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

DEPRECATED = {"dealer_ranker"}
OI_RE = re.compile(r"oi=(\d+)")
VOL_RE = re.compile(r"vol=(\d+)")


def load_selections() -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(REPO / "Data/inference/*/live_signal_audit.jsonl"))):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cs = rec.get("contract_selection")
            if not isinstance(cs, dict):
                continue
            for tkr, sel in cs.items():
                if not isinstance(sel, dict):
                    continue
                sa = sel.get("signal_audit") or {}
                reason = sel.get("reason") or ""
                oi = OI_RE.search(reason)
                vol = VOL_RE.search(reason)
                rows.append({
                    "module": rec.get("module"), "bar": rec.get("bar"), "ticker": tkr,
                    "action": sel.get("action"), "reason": reason,
                    "ref_price": sel.get("ref_price"),
                    "oi": int(oi.group(1)) if oi else np.nan,
                    "vol": int(vol.group(1)) if vol else np.nan,
                    "rank": sa.get("rank"), "score": sa.get("score"),
                })
    d = pd.DataFrame(rows)
    d["bar_ts"] = pd.to_datetime(d["bar"], utc=True, errors="coerce")
    for c in ("rank", "score", "ref_price"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["live_module"] = ~d["module"].isin(DEPRECATED)
    return d.dropna(subset=["action"])


def reason_family(r: str) -> str:
    if not r:
        return "(none)"
    if r.startswith("illiquid_option"):
        return "illiquid_option"
    if "price_floor" in r or "underlying_lt" in r:
        return "price_floor"
    if "no_contract" in r or "no_chain" in r or "unavailable" in r:
        return "no_chain"
    if "spread" in r:
        return "wide_spread"
    return r.split("(")[0][:28]


def main() -> None:
    out: dict = {}
    d = load_selections()
    print(f"contract_selection entries {len(d):,} over {d['module'].nunique()} modules, "
          f"{d['ticker'].nunique()} tickers, {d['bar_ts'].min().date()} .. "
          f"{d['bar_ts'].max().date()}")
    print(f"action split: {d['action'].value_counts().to_dict()}")
    out["n"] = int(len(d))

    d["reason_family"] = d["reason"].map(reason_family)
    d["rank_bucket"] = pd.cut(d["rank"], [0, 1, 2, 3, 5, 10, 1e9],
                              labels=["1", "2", "3", "4-5", "6-10", "11+"])

    for label, frame in (("ALL modules", d), ("LIVE modules only", d[d["live_module"]])):
        print("\n" + "=" * 78 + f"\n{label}\n" + "=" * 78)
        ct = pd.crosstab(frame["rank_bucket"], frame["action"])
        print("routing decision by RANK bucket (counts):")
        print(ct.to_string())
        if "option" in ct.columns:
            share = (ct.get("option", 0) / ct.sum(axis=1)).round(3)
            print("\nOPTION share of routed entries, by rank bucket:")
            print(share.to_string())
            out[f"option_share_{label.replace(' ', '_')}"] = {str(k): float(v)
                                                              for k, v in share.items()}
        print("\nwhy a name was sent to EQUITY instead (reason family x rank bucket):")
        eq = frame[frame["action"] == "equity"]
        if len(eq):
            print(pd.crosstab(eq["rank_bucket"], eq["reason_family"]).to_string())
        print("\nreason families overall:")
        print(frame["reason_family"].value_counts().head(8).to_string())
        out[f"reasons_{label.replace(' ', '_')}"] = frame["reason_family"].value_counts().to_dict()

    # the decisive number
    print("\n" + "=" * 78 + "\nTHE DECISIVE NUMBER\n" + "=" * 78)
    live = d[d["live_module"]]
    for label, frame in (("ALL modules", d), ("LIVE modules only", live)):
        top = frame[frame["rank"] <= 3]
        deep = frame[frame["rank"] > 3]
        if not len(top) or not len(deep):
            continue
        t_opt = float((top["action"] == "option").mean())
        d_opt = float((deep["action"] == "option").mean())
        t_ill = float((top["reason_family"] == "illiquid_option").mean())
        d_ill = float((deep["reason_family"] == "illiquid_option").mean())
        print(f"\n-- {label} --")
        print(f"  top-3   (n={len(top):>5}): option-routed {t_opt:.1%}   "
              f"blocked as illiquid {t_ill:.1%}")
        print(f"  rank 4+ (n={len(deep):>5}): option-routed {d_opt:.1%}   "
              f"blocked as illiquid {d_ill:.1%}")
        verdict = ("CONFIRMED - the option book skews to DEEPER ranks"
                   if d_opt > t_opt else
                   "NOT CONFIRMED - the option book skews to SHALLOWER ranks")
        print(f"  => {verdict}  (delta {d_opt - t_opt:+.1%})")
        out[f"decisive_{label.replace(' ', '_')}"] = {
            "top3_n": len(top), "top3_option_share": t_opt, "top3_illiquid_share": t_ill,
            "deep_n": len(deep), "deep_option_share": d_opt, "deep_illiquid_share": d_ill,
            "delta": d_opt - t_opt, "verdict": verdict}

    # liquidity of what got through vs what was blocked
    print("\n" + "=" * 78 + "\noi / volume recorded on the BLOCKED names\n" + "=" * 78)
    b = d[d["reason_family"] == "illiquid_option"]
    if len(b):
        print(b.groupby("rank_bucket", observed=True)[["oi", "vol"]]
              .agg(["count", "median"]).round(1).to_string())
        print(f"\ngates: ROUTE_MIN_OPEN_INTEREST=500, ROUTE_MIN_VOLUME=100")
        print(f"blocked entries with oi=0: {int((b['oi'] == 0).sum())} of "
              f"{int(b['oi'].notna().sum())} recorded")

    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "route_vs_rank.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {HERE / 'data' / 'route_vs_rank.json'}")


if __name__ == "__main__":
    main()
