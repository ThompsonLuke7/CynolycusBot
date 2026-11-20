"""Stage 3 evaluator: the pre-registered rule v2 on the point-in-time arms.

Pre-registration: research/execution_quality/27_stage3_preregistration.md (fixed 2026-11-08,
before the Stage 3 matrix existed). Reuses ``eval_stage2`` for alignment, per-bar statistics,
the block bootstrap, the top-k tables and the final-test gate; this module adds:

  neutral score  each arm's score, residualised within the bar on the percentile ranks of
                 daily ATR%, beta, ADV rank and price. Reported as the arm ``<arm>~n``; its
                 noise twin is ``<twin>~n``. The primary metric is its IC.
  S_new          validation bars before 2022-04-01, a period no Stage 2 read ever evaluated.
  rule v2        candidates L2_10__pit and L2_20__pit in both slots (4 tests). A candidate
                 qualifies for a slot only if, on validation:
                   1. its neutral IC CI is above 0;
                   2. its neutral-IC-minus-twin CI is above 0, with BH q <= 0.05 over the 4 tests;
                   3. its neutral IC is positive in a majority of folds;
                   4. its neutral IC point estimate is positive on S_new.
                 The pick is the qualifier with the highest neutral IC; otherwise no pick.
  test gate      the test opens only when the matrix holds >= 12 months of test bars with
                 matured 20-session labels, and the projected test MDE is logged first.
                 Confirmed = the pick's neutral IC CI on the test is above 0.

Top-k levels use fret winsorised at 1/99 within each bar (report-only; IC is rank-based and
unaffected). Corporate-action windows are already void in the labels.

Usage:
    .venv/bin/python -m strategies.momentum_expansion.ablation.eval_stage3
    .venv/bin/python -m strategies.momentum_expansion.ablation.eval_stage3 --open-final-test
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.momentum_expansion.ablation import eval_stage2 as E
from strategies.momentum_expansion.ablation.bootstrap import bh_fdr
from strategies.momentum_expansion.ablation.build_stage3_pit import EXPORT_DIR, RULE
from strategies.momentum_expansion.ablation.export_stage2 import FINAL_TEST_START

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "stage3_eval"
MANIFEST = EXPORT_DIR / "stage3_manifest.json"
DEFAULT_RESULTS = EXPORT_DIR / "stage3_results.tar"
BASELINE = "L0__pit"
CANDIDATES = ("L2_10__pit", "L2_20__pit")
NEUTRAL = "~n"
S_NEW_END = pd.Timestamp("2022-04-01", tz="UTC")
MIN_TEST_DAYS = 365
LABEL_MATURITY_DAYS = 30                 # 20 sessions plus weekends/holidays
WINSOR = (0.01, 0.99)


def neutral_name(name: str) -> str:
    """``y_L2_10`` -> ``y_L2_10~n``; ``y_L2_10__perm`` -> ``y_L2_10~n__perm`` (twins match by target)."""
    base, perm, _ = name.partition("__perm")
    return base + NEUTRAL + (perm or "")


def with_neutral(p: E.Panel, arms: list[dict]) -> tuple[E.Panel, list[dict]]:
    """Add each arm's neutral score as an extra arm."""
    ranks = {c: np.nan_to_num(E._group_rank(p.bar, p.col[c], pct=True), nan=0.5) for c in E.RESID_CONTROLS}
    score, fold, extra = dict(p.score), dict(p.fold), []
    for a in arms:
        s = p.score[a["name"]]
        n = neutral_name(a["name"])
        score[n] = E._residualise(p.bar, p.n_bars, s, np.isfinite(s), ranks).astype(np.float32)
        fold[n] = p.fold[a["name"]]
        extra.append({**a, "name": n, "target": neutral_name(a["target"])})
    return replace(p, score=score, fold=fold), arms + extra


def winsorise_returns(p: E.Panel) -> E.Panel:
    """fret_h clipped at its 1st/99th percentile within each bar (top-k levels only)."""
    col = dict(p.col)
    for h in E.TOPK_HORIZONS:
        r = pd.Series(col[f"fret_{h}"])
        g = r.groupby(p.bar)
        col[f"fret_{h}"] = r.clip(g.transform("quantile", WINSOR[0]), g.transform("quantile", WINSOR[1])) \
            .to_numpy(np.float32)
    return replace(p, col=col)


def s_new_ic(p: E.Panel, test_start: pd.Timestamp) -> dict[tuple[str, str], float]:
    """Mean neutral IC over validation bars before S_NEW_END, per candidate x slot yardstick."""
    out = {}
    for y in E.SLOTS.values():
        mask = E.partitions(p.bar_ts, E.YARDSTICKS[y], test_start)["validation"] & np.asarray(p.bar_ts < S_NEW_END)
        for a in CANDIDATES:
            n = neutral_name(a)
            if n in p.score:
                ic = E.bar_ic(p.bar, p.n_bars, p.score[n], p.col[y])
                out[(a, y)] = float(np.nanmean(ic[mask])) if np.isfinite(ic[mask]).any() else np.nan
    return out


def select_v2(ic: pd.DataFrame, paired: pd.DataFrame, s_new: dict, arms: list[dict]) -> dict:
    v_ic = ic[ic["partition"] == "validation"].set_index(["arm", "yardstick"])
    v_pair = paired[paired["partition"] == "validation"].set_index(["arm", "versus", "yardstick"])
    twin = E.twins(arms)
    tests = []
    for slot, y in E.SLOTS.items():
        for a in CANDIDATES:
            n = neutral_name(a)
            if (n, y) not in v_ic.index:
                continue
            r = v_ic.loc[(n, y)]
            pp = v_pair.loc[(n, twin[n], y)] if n in twin and (n, twin[n], y) in v_pair.index else None
            why = []
            if not r["ci_lo"] > 0:
                why.append("neutral IC CI not above 0")
            if pp is None or not pp["ci_lo"] > 0:
                why.append("does not beat its noise twin" if pp is not None else "no noise twin")
            if not r["folds_pos"] * 2 > r["n_folds"]:
                why.append(f"neutral IC positive in only {r['folds_pos']}/{r['n_folds']} folds")
            sn = s_new.get((a, y), np.nan)
            if not sn > 0:
                why.append(f"S_new neutral IC {sn:+.4f} not positive")
            tests.append({"slot": slot, "yardstick": y, "arm": a, "ic": r["mean"], "ic_ci": [r["ci_lo"], r["ci_hi"]],
                          "ic_mde": r["mde"], "vs_perm": None if pp is None else pp["mean"],
                          "vs_perm_ci": None if pp is None else [pp["ci_lo"], pp["ci_hi"]],
                          "vs_perm_p": np.nan if pp is None else pp["p"], "s_new_ic": sn,
                          "folds_pos": f"{r['folds_pos']}/{r['n_folds']}", "why_not": why})
    q = bh_fdr(pd.Series([t["vs_perm_p"] for t in tests], dtype=float))      # one family: all 4 tests
    for t, qv in zip(tests, q):
        t["vs_perm_q"] = qv
        if np.isfinite(t["vs_perm_p"]) and not qv <= E.FDR_Q:
            t["why_not"].append(f"noise-twin gap fails BH-FDR over 4 tests (q={qv:.3f})")
        t["qualifies"] = not t["why_not"]
    picks = {}
    for slot, y in E.SLOTS.items():
        cands = [t for t in tests if t["slot"] == slot]
        ok = [t for t in cands if t["qualifies"]]
        best = max(ok, key=lambda t: t["ic"]) if ok else None
        picks[slot] = {"yardstick": y, "pick": best["arm"] if best else None,
                       "candidates": sorted(cands, key=lambda t: -np.nan_to_num(t["ic"], nan=-9))}
    return picks


def test_window_days(manifest: dict, test_start: pd.Timestamp) -> int:
    """Days of test bars whose 20-session labels have matured, from the matrix's own range
    (no test outcome is read to decide whether the test may open)."""
    last = pd.Timestamp(manifest["range"][1])
    last = last.tz_localize("UTC") if last.tzinfo is None else last
    return int((last - pd.Timedelta(days=LABEL_MATURITY_DAYS) - test_start).days)


def projected_test_mde(manifest: dict, test_start: pd.Timestamp, out_dir: Path) -> dict:
    """Validation MDE of each pick's neutral IC, scaled by sqrt(validation weeks / test weeks)."""
    v = pd.read_csv(out_dir / "validation_ic.csv").set_index(["arm", "yardstick"])
    sel = json.loads((out_dir / "selection.json").read_text())
    test_weeks = max(1.0, test_window_days(manifest, test_start) / 7)
    out = {}
    for slot, s in sel["picks"].items():
        key = (neutral_name(s["pick"]), s["yardstick"]) if s["pick"] else None
        if key in v.index:
            val_weeks = v.loc[key, "n_bars"] / 10          # two 4H bars per session
            out[slot] = float(v.loc[key, "mde"] * math.sqrt(val_weeks / test_weeks))
    return out


def run(*, results: Path = DEFAULT_RESULTS, manifest_path: Path = MANIFEST, matrix: Path | None = None,
        out_dir: Path = RESULTS_DIR, n_boot: int = 1000, open_final_test: bool = False,
        test_start: str = FINAL_TEST_START) -> dict:
    manifest = json.loads(manifest_path.read_text())
    expected = [a["name"] for a in manifest["arms"]]
    results_dir = E.unpack(results)
    present = [a for a in expected if (results_dir / f"oof_{a}.parquet").exists()]
    if not present:
        raise FileNotFoundError(f"no oof_<arm>.parquet in {results_dir}")
    if len(present) < len(expected):
        logger.warning("partial run: missing %s", sorted(set(expected) - set(present)))
    arms = [a for a in manifest["arms"] if a["name"] in present]
    out_dir.mkdir(parents=True, exist_ok=True)
    r_hash = E.results_hash(results_dir, present)
    t0 = pd.Timestamp(test_start, tz="UTC")
    sel = None
    if open_final_test:
        days = test_window_days(manifest, t0)
        if days < MIN_TEST_DAYS:
            raise RuntimeError(f"the test window holds {days} days of matured labels; the pre-registration "
                               f"requires {MIN_TEST_DAYS} (rebuild the matrix on newer data first)")
        sel = E.authorise_final_test(out_dir, r_hash, expected, present,
                                     extra={"rule": RULE, "test_days": days,
                                            "projected_test_mde": projected_test_mde(manifest, t0, out_dir)})

    panel = E.load_panel(results_dir, matrix or manifest_path.parent / manifest["matrix"], present)
    if not open_final_test:
        panel = E.before(panel, t0)            # test outcomes are never computed on
    panel, arms_x = with_neutral(winsorise_returns(panel), arms)
    parts = ("test",) if open_final_test else ("validation",)
    ic, paired, topk = E.evaluate(panel, arms_x, parts=parts, test_start=t0, n_boot=n_boot, baseline=BASELINE)

    header = [f"Results: `{results_dir}` (hash {r_hash[:12]}), arms {len(present)}/{len(expected)}, "
              f"git {E._git_commit()[:10]}, n_boot {n_boot}, rule {RULE}.",
              f"Validation = OOF bars before {test_start} minus each yardstick's embargo; test = {test_start} on.",
              f"`~n` = neutral score (residualised on {', '.join(E.RESID_CONTROLS)}); it is the primary metric.",
              "Top-k levels use fret winsorised 1/99 within the bar."]
    if open_final_test:
        picks, stem = sel["picks"], "test"
        header.insert(0, f"**FINAL TEST.** Selection frozen {sel['created_utc']} (hash {sel['selection_hash'][:12]}).")
        t_ic = ic.set_index(["arm", "yardstick"])
        for s in picks.values():
            if s["pick"]:
                r = t_ic.loc[(neutral_name(s["pick"]), s["yardstick"])]
                s["confirmed"] = bool(r["ci_lo"] > 0)
                header.append(f"{s['pick']} on {s['yardstick']}: neutral IC {E._ci(r)} -> "
                              f"{'CONFIRMED' if s['confirmed'] else 'not confirmed'}")
    else:
        picks = select_v2(ic, paired, s_new_ic(panel, t0), arms_x)
        stem = "validation"
        sel = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule": RULE,
               "results_hash": r_hash, "selection_hash": E.selection_hash(picks, rule=RULE),
               "arms_present": present, "complete": len(present) == len(expected), "git_commit": E._git_commit(),
               "picks": picks}
        (out_dir / "selection.json").write_text(json.dumps(sel, indent=2, default=float))
    for name, df in (("ic", ic), ("paired", paired), ("topk", topk)):
        df[df["partition"] == stem].to_csv(out_dir / f"{stem}_{name}.csv", index=False)
    E.write_report(out_dir / f"{stem}_report.md", stem, ic, paired, topk, picks, arms_x, header,
                   baseline=BASELINE, title="Stage 3 evaluation (rule v2)")
    logger.info("picks: %s -> %s", {k: v["pick"] for k, v in picks.items()}, out_dir / f"{stem}_report.md")
    return {"ic": ic, "paired": paired, "topk": topk, "picks": picks, "results_hash": r_hash}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="stage3_results.tar or its folder")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--matrix", type=Path, default=None, help="default: the manifest's matrix next to it")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--open-final-test", action="store_true",
                    help="read the sealed test once (>= 12 months matured, complete run, frozen selection; logged)")
    a = ap.parse_args()
    run(results=a.results, manifest_path=a.manifest, matrix=a.matrix, out_dir=a.out, n_boot=a.n_boot,
        open_final_test=a.open_final_test)


if __name__ == "__main__":
    main()
