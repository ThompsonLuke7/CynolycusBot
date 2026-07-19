"""
Phase 3 of the segmented entry/exit-policy study: entry/selection-side levers,
evaluated with FIXED exit policies (no exit re-tuning).

Two experiments, both with strict val-only selection:
  3a. Cohort-conditioned admission: extend the top-10 entry cutoff to rank<=X
      for names in high-tail-propensity cohorts (from
      exit_policy_segmentation.py's pre-val clustering). Candidates are
      evaluated on VAL ONLY; a single pre-stated rule picks the winner; TEST
      is simulated once, only for the winner + baseline.
      Pre-stated selection rule (written before any test simulation):
        maximize val TOTAL return subject to val mean-per-trade >= 80% of the
        top-10 baseline's val mean (guards against pure dilution).
  3b. Score-level gate ("do we need to fill every top-10 slot every 4H"):
      only admit rank<=10 names whose score clears an absolute threshold set
      from the VAL entering-score distribution (25th/50th pct). Winner per
      module picked on VAL by the pre-stated rule:
        highest val ret-per-bar among gates that keep >= 40% of baseline
        trades; report turnover reduction alongside. TEST run once for the
        winner + baseline.

Exit policies are FIXED at id4 (tail objective, 3a) and g284 (efficiency
objective, 3b) from the prior rounds — nothing about the exit is searched.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/exit_policy_entry_side.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/capstone"))
from exit_policy_segmentation import (  # noqa: E402
    VAL_START, VAL_END, TEST_END, TOPK, OUT_DIR, POLICIES, all_trades,
)

ID4 = POLICIES["id4"]
G284 = POLICIES["g284"]

EXT_COHORT_SETS = {
    "explosive": {"explosive"},
    "explosive+young": {"explosive", "young"},
    "explosive+young+moderate": {"explosive", "young", "moderate"},
}
EXT_DEPTHS = (15, 20, 25)
GATE_QUANTILES = (0.25, 0.50)


def load_inputs():
    streams = {m: pd.read_parquet(OUT_DIR / f"stream_{m}.parquet") for m in ("momentum", "htf", "meta")}
    cohorts = pd.read_csv(OUT_DIR / "ticker_tail_cohorts.csv")[["ticker", "tail_cohort"]]
    return streams, cohorts


def member_variant(stream: pd.DataFrame, start, end, *, ext_tickers=None, ext_depth=None,
                   score_min=None) -> pd.DataFrame:
    df = stream[(stream["timestamp"] >= start) & (stream["timestamp"] < end)].copy()
    in_top = df["rk"] <= TOPK
    if score_min is not None:
        in_top &= df["score"] >= score_min
    if ext_tickers is not None:
        in_top |= df["ticker"].isin(ext_tickers) & (df["rk"] <= ext_depth)
    df["in_top"] = in_top
    return df[["timestamp", "ticker", "in_top"]]


def stats(tr: pd.DataFrame, member: pd.DataFrame) -> dict:
    if len(tr) == 0:
        return dict(n=0)
    # avg concurrent book size = member rows flagged in_top per timestamp
    conc = member[member["in_top"]].groupby("timestamp").size()
    return dict(n=len(tr), mean=tr["ret"].mean(), median=tr["ret"].median(),
                win=(tr["ret"] > 0).mean(), total=tr["ret"].sum(),
                rpb=(tr["ret"] / np.maximum(tr["bars_held"], 1)).mean(),
                hold=tr["bars_held"].mean(), avg_book=conc.mean())


def run_3a(streams, cohorts) -> None:
    print("=" * 70)
    print("3a. cohort-conditioned admission (exit fixed = id4), VAL selection")
    print("=" * 70)
    rows = []
    for module, stream in streams.items():
        base_member = member_variant(stream, VAL_START, VAL_END)
        base_tr = all_trades(base_member, **ID4)
        base = stats(base_tr, base_member)
        rows.append(dict(module=module, variant="base_top10", window="val", **base))
        print(f"[{module}] base top10 val: n={base['n']} mean={base['mean']:.4f} "
              f"total={base['total']:.1f} rpb={base['rpb']:.4f} book={base['avg_book']:.1f}")
        for cs_name, cs in EXT_COHORT_SETS.items():
            ext_t = set(cohorts.loc[cohorts["tail_cohort"].isin(cs), "ticker"])
            for depth in EXT_DEPTHS:
                mem = member_variant(stream, VAL_START, VAL_END, ext_tickers=ext_t, ext_depth=depth)
                tr = all_trades(mem, **ID4)
                st = stats(tr, mem)
                st.update(module=module, variant=f"ext[{cs_name}]<= {depth}", window="val")
                rows.append(st)
                print(f"  ext[{cs_name:26s}] rk<={depth}: n={st['n']} mean={st['mean']:.4f} "
                      f"total={st['total']:.1f} rpb={st['rpb']:.4f} book={st['avg_book']:.1f}")
    val = pd.DataFrame(rows)

    # pre-stated rule: max val total s.t. mean >= 0.8 * base mean, per module
    winners = {}
    for module in streams:
        sub = val[val["module"] == module]
        base_mean = sub.loc[sub["variant"] == "base_top10", "mean"].iloc[0]
        cand = sub[(sub["variant"] != "base_top10") & (sub["mean"] >= 0.8 * base_mean)]
        winners[module] = None if cand.empty else cand.sort_values("total").iloc[-1]["variant"]
        print(f"[{module}] 3a val winner: {winners[module]} (rule: max total, mean>=0.8x base)")

    # single frozen TEST read: baseline + winner only
    test_rows = []
    for module, stream in streams.items():
        base_member = member_variant(stream, VAL_END, TEST_END)
        base_tr = all_trades(base_member, **ID4)
        st = stats(base_tr, base_member)
        st.update(module=module, variant="base_top10", window="test")
        test_rows.append(st)
        w = winners[module]
        if w is None:
            continue
        cs_name = w.split("[")[1].split("]")[0]
        depth = int(w.split("<=")[1])
        ext_t = set(cohorts.loc[cohorts["tail_cohort"].isin(EXT_COHORT_SETS[cs_name]), "ticker"])
        mem = member_variant(stream, VAL_END, TEST_END, ext_tickers=ext_t, ext_depth=depth)
        tr = all_trades(mem, **ID4)
        # which tickers did the extension actually add?
        added = set(tr["ticker"]) - set(base_tr["ticker"])
        stw = stats(tr, mem)
        stw.update(module=module, variant=w, window="test",
                   added_tickers=",".join(sorted(added)[:40]))
        test_rows.append(stw)
        print(f"[{module}] TEST base: n={st['n']} mean={st['mean']:.4f} total={st['total']:.1f} | "
              f"winner {w}: n={stw['n']} mean={stw['mean']:.4f} total={stw['total']:.1f} "
              f"rpb={stw['rpb']:.4f} book={stw['avg_book']:.1f}")
        print(f"    tickers added by extension on test: {sorted(added)}")
    pd.concat([val, pd.DataFrame(test_rows)], ignore_index=True).to_csv(
        OUT_DIR / "phase3a_cohort_admission.csv", index=False)


def run_3b(streams, cohorts) -> None:
    print("=" * 70)
    print("3b. score-level entry gate (exit fixed = g284), VAL selection")
    print("=" * 70)
    rows = []
    thresholds = {}
    for module, stream in streams.items():
        val_scores = stream[(stream["timestamp"] >= VAL_START) & (stream["timestamp"] < VAL_END)
                            & (stream["rk"] <= TOPK)]["score"]
        base_member = member_variant(stream, VAL_START, VAL_END)
        base_tr = all_trades(base_member, **G284)
        base = stats(base_tr, base_member)
        base.update(module=module, variant="base_top10", window="val")
        rows.append(base)
        print(f"[{module}] base top10 val: n={base['n']} mean={base['mean']:.4f} "
              f"rpb={base['rpb']:.4f} total={base['total']:.1f} book={base['avg_book']:.1f}")
        for q in GATE_QUANTILES:
            thr = float(val_scores.quantile(q))
            mem = member_variant(stream, VAL_START, VAL_END, score_min=thr)
            tr = all_trades(mem, **G284)
            st = stats(tr, mem)
            st.update(module=module, variant=f"gate_q{int(q*100)}", window="val", threshold=thr)
            rows.append(st)
            thresholds[(module, f"gate_q{int(q*100)}")] = thr
            print(f"  gate q{int(q*100)} (score>={thr:.4f}): n={st['n']} ({st['n']/base['n']:.0%} of base) "
                  f"mean={st['mean']:.4f} rpb={st['rpb']:.4f} total={st['total']:.1f} book={st['avg_book']:.1f}")
    val = pd.DataFrame(rows)

    winners = {}
    for module in streams:
        sub = val[val["module"] == module]
        base_n = sub.loc[sub["variant"] == "base_top10", "n"].iloc[0]
        base_rpb = sub.loc[sub["variant"] == "base_top10", "rpb"].iloc[0]
        cand = sub[(sub["variant"] != "base_top10") & (sub["n"] >= 0.4 * base_n)]
        if cand.empty or cand["rpb"].max() <= base_rpb:
            winners[module] = None
        else:
            winners[module] = cand.sort_values("rpb").iloc[-1]["variant"]
        print(f"[{module}] 3b val winner: {winners[module]} "
              f"(rule: max rpb among gates keeping >=40% of trades, must beat base rpb)")

    test_rows = []
    for module, stream in streams.items():
        base_member = member_variant(stream, VAL_END, TEST_END)
        base_tr = all_trades(base_member, **G284)
        st = stats(base_tr, base_member)
        st.update(module=module, variant="base_top10", window="test")
        test_rows.append(st)
        w = winners[module]
        if w is None:
            continue
        thr = thresholds[(module, w)]
        mem = member_variant(stream, VAL_END, TEST_END, score_min=thr)
        tr = all_trades(mem, **G284)
        stw = stats(tr, mem)
        stw.update(module=module, variant=w, window="test", threshold=thr)
        test_rows.append(stw)
        print(f"[{module}] TEST base: n={st['n']} mean={st['mean']:.4f} rpb={st['rpb']:.4f} | "
              f"{w}: n={stw['n']} ({stw['n']/st['n']:.0%}) mean={stw['mean']:.4f} "
              f"rpb={stw['rpb']:.4f} total={stw['total']:.1f}")
    pd.concat([val, pd.DataFrame(test_rows)], ignore_index=True).to_csv(
        OUT_DIR / "phase3b_score_gate.csv", index=False)


def main() -> None:
    streams, cohorts = load_inputs()
    run_3a(streams, cohorts)
    run_3b(streams, cohorts)


if __name__ == "__main__":
    main()
