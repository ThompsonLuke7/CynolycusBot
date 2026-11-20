"""Stage 2 evaluator: score the Colab label arms out of fold, pick on validation,
confirm once on the untouched final test.

Plan: docs/superpowers/plans/2026-09-25-4h-feature-study-plan.md (Stage 2/3).
Checklist: research/year_two_experiment_checklist_2026-11-04.md (items 0.1, 0.3).

Inputs: ``stage2_results.tar`` (or its extracted folder) from
``colab/stage2_train_colab.py`` -- one ``oof_<arm>.parquet`` per arm with
(timestamp, ticker, score, fold) -- plus ``stage2_matrix.parquet`` for the
outcomes. Nothing is trained here.

Yardsticks. Every arm is scored against the same ones, whatever it trained on:
  L2_h     SPY-excess return minus same-bar vol-decile peers, h = 10/20/40
           sessions. Vol-neutral; the primary yardstick.
  fret_h   raw executable return (carries the vol/beta premium)
  fmfe_10  forward range: IC here with none on L2 = a vol forecaster
All outcomes enter at the next 4H bar's open (``stage2_labels``).

Per-bar metrics, aggregated with a block bootstrap (blocks of ceil(h/5) weeks so
overlapping outcome windows are not counted as independent), 95% CI, and an
MDE at 80% power:
  ic        within-bar Spearman(score, yardstick), the primary metric
  vs perm   ic minus the arm's noise-trained twin on the same rows (a model
            fitted to shuffled labels still tilts to vol/beta, so raw top-k
            levels are not evidence on their own)
  vs L0     ic minus the deployed-label baseline, on shared rows
  top-k     gross excess over the bar's equal-weight mean; the picks' own
            SPY-excess return after an ASSUMED cost (run_subset_eval bands);
            excess after residualising the score on vol/beta/liquidity/price;
            precision against y_L2 (base 20%); the picks' mean ATR% and beta
            percentile (0.5 = no tilt). Pseudo-arm VOL ranks by daily ATR%.

Partitions. validation = OOF bars before FINAL_TEST_START, purged by the
yardstick's embargo so no validation outcome window reaches into the test.
test = bars from FINAL_TEST_START on. Test outcomes are dropped before any
computation unless --open-final-test is given. That flag requires every arm, a
selection frozen from validation on these exact results, and it logs the read.
A selection that changes after a read is refused.

Selection rule (fixed 2026-11-07, before any Stage 2 result existed). For each
slot (momentum = L2_10, htf = L2_20) an arm qualifies if, on validation:
  1. its IC CI is above zero;
  2. its IC-minus-perm-twin CI is above zero, and that difference survives
     BH-FDR (q <= 0.05) across the slot's candidates;
  3. its IC is positive in more than half of the folds.
The pick is the qualifier with the highest IC. If nothing qualifies there is no
pick, and the arm family is reported as not beating its noise control, with
the MDE.

Reading L2 IC: L2 is mean-centred within vol deciles, but high-vol returns are
right-skewed (median below mean). Rank IC therefore marks vol down: on the
real matrix, ranking by ATR% alone scores about -0.06 on L2_10. Treat the VOL
row as the reference, not zero.

Usage:
    .venv/bin/python -m strategies.momentum_expansion.ablation.eval_stage2
    .venv/bin/python -m strategies.momentum_expansion.ablation.eval_stage2 --open-final-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
import tarfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from strategies.momentum_expansion.ablation.bootstrap import bh_fdr, week_block_bootstrap_ci
from strategies.momentum_expansion.ablation.export_stage2 import EXPORT_DIR, FINAL_TEST_START, embargo_days
from strategies.momentum_expansion.ablation.run_subset_eval import _cost

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "stage2_eval"
MANIFEST = EXPORT_DIR / "stage2_manifest.json"
DEFAULT_RESULTS = EXPORT_DIR / "stage2_results.tar"

YARDSTICKS = {"L2_10": 10, "L2_20": 20, "L2_40": 40, "fret_10": 10, "fret_20": 20, "fmfe_10": 10}
PAIRED = ("L2_10", "L2_20", "L2_40")      # arm-vs-arm differences are reported on these
SLOTS = {"momentum": "L2_10", "htf": "L2_20"}
BASELINE = "L0"                            # the deployed momentum label, retrained with the same recipe
VOL = "VOL"                                # pseudo-arm: top by daily ATR%, the tilt a noise model reproduces
TOPK_HORIZONS = (10, 20)
TOP_KS = (1, 3, 5, 10)
CI_K = 3                                   # the deployed momentum top_n; the only k given a CI
RESID_CONTROLS = ("daily_atr_pct", "beta_spy_60", "adv_rank", "px")
MIN_PAIRS = 50                             # fewer scored names on a bar -> that bar is skipped
FDR_Q = 0.05
SELECTION_RULE = "v1-2026-11-07"


# ---------------------------------------------------------------------------
# Panel: matrix outcomes aligned with every arm's OOF scores
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Panel:
    bar: np.ndarray               # bar code per row; rows = matrix rows scored by any arm
    bar_ts: pd.DatetimeIndex      # UTC timestamp of each bar code, ascending
    col: dict[str, np.ndarray]    # outcome / control columns per row
    score: dict[str, np.ndarray]  # per arm, NaN where the arm did not score the row
    fold: dict[str, np.ndarray]   # per arm, fold of each BAR (-1 = arm has no rows there)

    @property
    def n_bars(self) -> int:
        return len(self.bar_ts)


def _matrix_columns() -> list[str]:
    cols = list(YARDSTICKS) + list(RESID_CONTROLS)
    for h in TOPK_HORIZONS:
        cols += [f"L2_{h}", f"fret_{h}", f"spy_fret_{h}", f"y_L2_{h}"]
    return list(dict.fromkeys(cols))


def _ts_ns(col: pa.ChunkedArray) -> np.ndarray:
    if col.type.tz is None:
        raise ValueError("timestamps must be tz-aware (UTC)")
    return pc.cast(col, pa.timestamp("ns", tz="UTC")).cast(pa.int64()).to_numpy()


def load_panel(results_dir: Path, matrix_path: Path, arms: list[str]) -> Panel:
    """Align each ``oof_<arm>.parquet`` with the matrix rows it scored.

    Fails loudly on OOF rows missing from the matrix, duplicate OOF rows or
    non-finite scores -- any of those means the results and the matrix do not
    belong together.
    """
    t = pq.read_table(matrix_path, columns=["timestamp", "ticker", *_matrix_columns()])
    ts = _ts_ns(t.column("timestamp"))
    tick = t.column("ticker").combine_chunks().dictionary_encode()
    names = tick.dictionary
    bars, bcode = np.unique(ts, return_inverse=True)
    key = bcode.astype(np.int64) * len(names) + tick.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    order = np.argsort(key, kind="stable")
    skey = key[order]
    if np.any(skey[1:] == skey[:-1]):
        raise ValueError(f"{matrix_path} has duplicate (timestamp, ticker) rows")

    hits = {}
    for arm in arms:
        o = pq.read_table(results_dir / f"oof_{arm}.parquet", columns=["timestamp", "ticker", "score", "fold"])
        ots = _ts_ns(o.column("timestamp"))
        ob = np.searchsorted(bars, ots)
        oti = pc.fill_null(pc.index_in(o.column("ticker"), value_set=names), -1).to_numpy()
        okey = ob.astype(np.int64) * len(names) + oti
        pos = np.minimum(np.searchsorted(skey, okey), len(skey) - 1)
        found = (ob < len(bars)) & (bars[np.minimum(ob, len(bars) - 1)] == ots) & (oti >= 0) & (skey[pos] == okey)
        if not found.all():
            raise ValueError(f"{arm}: {int((~found).sum()):,} OOF rows are not in {matrix_path.name}")
        idx = order[pos]
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"{arm}: duplicate (timestamp, ticker) rows in the OOF file")
        s = o.column("score").to_numpy().astype(np.float32)
        if not np.isfinite(s).all():
            raise ValueError(f"{arm}: non-finite OOF scores")
        hits[arm] = (idx, s, o.column("fold").to_numpy().astype(np.int16))
        logger.info("%-22s %9d rows  %s .. %s  folds=%s", arm, len(idx),
                    pd.Timestamp(ots.min(), tz="UTC").date(), pd.Timestamp(ots.max(), tz="UTC").date(),
                    sorted(np.unique(hits[arm][2]).tolist()))

    rows = np.unique(np.concatenate([h[0] for h in hits.values()]))
    used, bar = np.unique(bcode[rows], return_inverse=True)
    col = {c: t.column(c).to_numpy().astype(np.float32)[rows] for c in _matrix_columns()}
    col["cost"] = _cost(pd.Series(col["px"])).to_numpy(dtype=np.float32)
    score, fold = {}, {}
    for arm, (idx, s, f) in hits.items():
        at = np.searchsorted(rows, idx)
        score[arm] = np.full(len(rows), np.nan, dtype=np.float32)
        score[arm][at] = s
        fold[arm] = np.full(len(used), -1, dtype=np.int16)
        fold[arm][bar[at]] = f
    return Panel(bar=bar.astype(np.int64), bar_ts=pd.to_datetime(bars[used], utc=True), col=col,
                 score=score, fold=fold)


def before(p: Panel, cutoff: pd.Timestamp) -> Panel:
    """The panel restricted to bars strictly before ``cutoff`` (bar codes are kept)."""
    n = int(np.searchsorted(p.bar_ts, cutoff))
    keep = p.bar < n
    return replace(p, bar=p.bar[keep], bar_ts=p.bar_ts[:n], col={k: v[keep] for k, v in p.col.items()},
                   score={k: v[keep] for k, v in p.score.items()}, fold={k: v[:n] for k, v in p.fold.items()})


# ---------------------------------------------------------------------------
# Per-bar statistics (vectorised over all bars at once)
# ---------------------------------------------------------------------------

def _group_rank(bar: np.ndarray, x: np.ndarray, *, pct: bool = False) -> np.ndarray:
    return pd.Series(x).groupby(bar).rank(method="average", pct=pct).to_numpy()


def _bar_mean(bar: np.ndarray, n_bars: int, x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    n = np.bincount(bar[rows], minlength=n_bars)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.bincount(bar[rows], x[rows].astype(np.float64), minlength=n_bars) / n


def bar_ic(bar: np.ndarray, n_bars: int, score: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman(score, y) within each bar on rows where both are present.
    NaN for bars with fewer than MIN_PAIRS pairs or no variance."""
    m = np.isfinite(score) & np.isfinite(y)
    b = bar[m]
    rs, ry = _group_rank(b, score[m]), _group_rank(b, y[m])
    n = np.bincount(b, minlength=n_bars).astype(np.float64)
    sx, sy = np.bincount(b, rs, minlength=n_bars), np.bincount(b, ry, minlength=n_bars)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.bincount(b, rs * ry, minlength=n_bars) - sx * sy / n
        vx = np.bincount(b, rs * rs, minlength=n_bars) - sx * sx / n
        vy = np.bincount(b, ry * ry, minlength=n_bars) - sy * sy / n
        ic = cov / np.sqrt(vx * vy)
    ic[(n < MIN_PAIRS) | ~(vx > 1e-9) | ~(vy > 1e-9)] = np.nan
    return ic


def paired_ic_diff(bar: np.ndarray, n_bars: int, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-bar IC(a) - IC(b), both on the rows that a, b and y all cover."""
    both = np.isfinite(a) & np.isfinite(b)
    return bar_ic(bar, n_bars, np.where(both, a, np.nan), y) - bar_ic(bar, n_bars, np.where(both, b, np.nan), y)


def _positions(bar: np.ndarray, score: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rows ordered by bar then score descending, position within its bar).
    Ties keep row order (stable), as in ``metrics.top_n_per_group``."""
    rows = np.flatnonzero(valid & np.isfinite(score))
    order = rows[np.lexsort((-score[rows], bar[rows]))]
    b = bar[order]
    start = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    pos = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
    return order, pos


def _residualise(bar: np.ndarray, n_bars: int, score: np.ndarray, valid: np.ndarray,
                 ranks: dict[str, np.ndarray]) -> np.ndarray:
    """Within-bar OLS residual of the score's percentile rank on the controls'
    percentile ranks (one regression per bar, solved in one batch)."""
    rows = np.flatnonzero(valid)
    b = bar[rows]
    y = _group_rank(b, score[rows], pct=True)
    X = np.column_stack([np.ones(len(rows))] + [ranks[c][rows] for c in RESID_CONTROLS])
    k = X.shape[1]
    xtx, xty = np.empty((n_bars, k, k)), np.empty((n_bars, k))
    for i in range(k):
        xty[:, i] = np.bincount(b, X[:, i] * y, minlength=n_bars)
        for j in range(i, k):
            xtx[:, i, j] = xtx[:, j, i] = np.bincount(b, X[:, i] * X[:, j], minlength=n_bars)
    beta = np.linalg.solve(xtx + 1e-9 * np.eye(k), xty[..., None])[..., 0]
    out = np.full(len(score), np.nan)
    out[rows] = y - np.einsum("ij,ij->i", X, beta[b])
    return out


def topk_bars(p: Panel, score: np.ndarray, h: int, ranks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-bar top-k series for one arm at horizon ``h`` (see module docstring)."""
    valid = np.isfinite(score) & np.isfinite(p.col[f"L2_{h}"])
    rows = np.flatnonzero(valid)
    ret, hit = p.col[f"fret_{h}"], p.col[f"y_L2_{h}"]
    own_net = ret - p.col[f"spy_fret_{h}"] - p.col["cost"]
    thin = np.bincount(p.bar[rows], minlength=p.n_bars) < MIN_PAIRS
    ew, base_hit = _bar_mean(p.bar, p.n_bars, ret, rows), _bar_mean(p.bar, p.n_bars, hit, rows)
    order, pos = _positions(p.bar, score, valid)
    r_order, r_pos = _positions(p.bar, _residualise(p.bar, p.n_bars, score, valid, ranks), valid)
    out = {}
    for k in TOP_KS:
        pk, rk = order[pos < k], r_order[r_pos < k]
        out[f"ex_k{k}"] = _bar_mean(p.bar, p.n_bars, ret, pk) - ew
        out[f"net_k{k}"] = _bar_mean(p.bar, p.n_bars, own_net, pk)
        out[f"rex_k{k}"] = _bar_mean(p.bar, p.n_bars, ret, rk) - ew
        out[f"prec_k{k}"] = _bar_mean(p.bar, p.n_bars, hit, pk) - base_hit
        out[f"atr_k{k}"] = _bar_mean(p.bar, p.n_bars, ranks["daily_atr_pct"], pk)
        out[f"beta_k{k}"] = _bar_mean(p.bar, p.n_bars, ranks["beta_spy_60"], pk)
    for v in out.values():
        v[thin] = np.nan
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def partitions(bar_ts: pd.DatetimeIndex, h: int, test_start: pd.Timestamp) -> dict[str, np.ndarray]:
    """validation ends ``embargo_days(h)`` before the test so no h-session
    outcome window crosses into it."""
    cut = test_start - pd.Timedelta(days=embargo_days(h))
    return {"validation": np.asarray(bar_ts < cut), "test": np.asarray(bar_ts >= test_start)}


def _blocks(bar_ts: pd.DatetimeIndex, h: int) -> np.ndarray:
    weeks = max(1, math.ceil(h / 5))
    return bar_ts.tz_convert(None).to_period("W").asi8 // weeks


def summarize(values: np.ndarray, bar_ts: pd.DatetimeIndex, h: int, bar_fold: np.ndarray,
              mask: np.ndarray, *, n_boot: int) -> dict:
    keep = mask & np.isfinite(values)
    v = values[keep]
    out = {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "mde": np.nan, "p": np.nan,
           "n_bars": int(keep.sum()), "folds_pos": 0, "n_folds": 0}
    if len(v) < 2:
        return out
    b = week_block_bootstrap_ci(pd.Series(v), pd.Series(_blocks(bar_ts[keep], h)), n_boot=n_boot, alpha=0.05)
    se = (b["ci_hi"] - b["ci_lo"]) / (2 * 1.96)
    fm = pd.Series(v).groupby(bar_fold[keep]).mean()
    out.update(mean=b["point"], ci_lo=b["ci_lo"], ci_hi=b["ci_hi"], mde=2.8 * se,
               p=math.erfc(abs(b["point"]) / se / math.sqrt(2)) if se > 0 else np.nan,
               folds_pos=int((fm > 0).sum()), n_folds=int(len(fm)))
    out.update({f"y{yr}": m for yr, m in pd.Series(v).groupby(bar_ts[keep].year).mean().items()})
    return out


def twins(arms: list[dict]) -> dict[str, str]:
    """{real arm: its noise-trained control} -- matched by target, so feature
    variants share their label's control."""
    by_target = {a["target"]: a["name"] for a in arms if a["control"]}
    return {a["name"]: by_target[a["target"] + "__perm"] for a in arms
            if not a["control"] and a["target"] + "__perm" in by_target}


def evaluate(p: Panel, arms: list[dict], *, parts: tuple[str, ...], test_start: pd.Timestamp,
             n_boot: int, baseline: str = BASELINE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(ic, paired, topk) tables, one row per arm x yardstick/metric x partition."""
    real = [a["name"] for a in arms if not a["control"] and a["name"] in p.score]
    twin = {a: c for a, c in twins(arms).items() if a in p.score and c in p.score}
    scores = dict(p.score)
    base_rows = np.isfinite(scores[baseline]) if baseline in scores else np.ones(len(p.bar), bool)
    scores[VOL] = np.where(base_rows, p.col["daily_atr_pct"], np.nan)
    folds = {**p.fold, VOL: p.fold.get(baseline, np.zeros(p.n_bars, np.int16))}
    ranks = {c: np.nan_to_num(_group_rank(p.bar, p.col[c], pct=True), nan=0.5) for c in RESID_CONTROLS}

    def rec(values, horizon, fold, part, mask, **ids):
        return {**ids, "partition": part, **summarize(values, p.bar_ts, horizon, fold, mask, n_boot=n_boot)}

    ic_rows, pair_rows, topk_rows = [], [], []
    for y, h in YARDSTICKS.items():
        masks = partitions(p.bar_ts, h, test_start)
        own = {arm: bar_ic(p.bar, p.n_bars, s, p.col[y]) for arm, s in scores.items()}
        for arm, series in own.items():
            ic_rows += [rec(series, h, folds[arm], part, masks[part], arm=arm, yardstick=y) for part in parts]
        if y not in PAIRED:
            continue
        for arm in real:
            for other in {twin.get(arm), baseline} - {None, arm}:
                if other in scores:
                    if np.array_equal(np.isfinite(scores[arm]), np.isfinite(scores[other])):
                        d = own[arm] - own[other]          # same rows: the own series are already paired
                    else:
                        d = paired_ic_diff(p.bar, p.n_bars, scores[arm], scores[other], p.col[y])
                    pair_rows += [rec(d, h, folds[arm], part, masks[part], arm=arm, versus=other, yardstick=y)
                                  for part in parts]
        logger.info("yardstick %s done", y)
    for h in TOPK_HORIZONS:
        masks = partitions(p.bar_ts, h, test_start)
        per = {arm: topk_bars(p, s, h, ranks) for arm, s in scores.items()}
        for arm, series in per.items():
            for part in parts:
                m = masks[part]
                for metric, v in series.items():
                    if metric.endswith(f"_k{CI_K}"):
                        topk_rows.append(rec(v, h, folds[arm], part, m, arm=arm, h=h, metric=metric))
                    else:
                        topk_rows.append({"arm": arm, "h": h, "metric": metric, "partition": part,
                                          "mean": float(np.nanmean(v[m])) if np.isfinite(v[m]).any() else np.nan,
                                          "n_bars": int(np.isfinite(v[m]).sum())})
                if arm in twin:
                    d = series[f"ex_k{CI_K}"] - per[twin[arm]][f"ex_k{CI_K}"]
                    topk_rows.append(rec(d, h, folds[arm], part, m, arm=arm, h=h, metric=f"ex_vs_perm_k{CI_K}"))
        logger.info("top-k h=%d done", h)
    return pd.DataFrame(ic_rows), pd.DataFrame(pair_rows), pd.DataFrame(topk_rows)


# ---------------------------------------------------------------------------
# Selection and the final-test gate
# ---------------------------------------------------------------------------

def select(ic: pd.DataFrame, paired: pd.DataFrame, arms: list[dict]) -> dict:
    """Apply the pre-registered rule (module docstring) to the validation tables."""
    v_ic = ic[ic["partition"] == "validation"].set_index(["arm", "yardstick"])
    v_pair = paired[paired["partition"] == "validation"].set_index(["arm", "versus", "yardstick"])
    twin = twins(arms)
    picks = {}
    for slot, y in SLOTS.items():
        cands = []
        for a in [a["name"] for a in arms if not a["control"]]:
            if (a, y) not in v_ic.index:
                continue
            r = v_ic.loc[(a, y)]
            pp = v_pair.loc[(a, twin[a], y)] if a in twin and (a, twin[a], y) in v_pair.index else None
            why = []
            if not r["ci_lo"] > 0:
                why.append("IC CI not above 0")
            if pp is None or not pp["ci_lo"] > 0:
                why.append("does not beat its noise twin" if pp is not None else "no noise twin")
            if not r["folds_pos"] * 2 > r["n_folds"]:
                why.append(f"IC positive in only {r['folds_pos']}/{r['n_folds']} folds")
            cands.append({"arm": a, "ic": r["mean"], "ic_ci": [r["ci_lo"], r["ci_hi"]], "ic_mde": r["mde"],
                          "vs_perm": None if pp is None else pp["mean"],
                          "vs_perm_ci": None if pp is None else [pp["ci_lo"], pp["ci_hi"]],
                          "vs_perm_p": np.nan if pp is None else pp["p"],
                          "folds_pos": f"{r['folds_pos']}/{r['n_folds']}", "why_not": why})
        q = bh_fdr(pd.Series([c["vs_perm_p"] for c in cands], dtype=float))
        for c, qv in zip(cands, q):
            c["vs_perm_q"] = qv
            if np.isfinite(c["vs_perm_p"]) and not qv <= FDR_Q:
                c["why_not"].append(f"noise-twin gap fails BH-FDR (q={qv:.3f})")
            c["qualifies"] = not c["why_not"]
        ok = [c for c in cands if c["qualifies"]]
        best = max(ok, key=lambda c: c["ic"]) if ok else None
        picks[slot] = {"yardstick": y, "pick": best["arm"] if best else None,
                       "candidates": sorted(cands, key=lambda c: -np.nan_to_num(c["ic"], nan=-9))}
    return picks


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def results_hash(results_dir: Path, arms: list[str]) -> str:
    return hashlib.sha256(json.dumps(
        {a: _sha256(results_dir / f"oof_{a}.parquet") for a in sorted(arms)}).encode()).hexdigest()


def selection_hash(picks: dict, rule: str = SELECTION_RULE) -> str:
    frozen = {"rule": rule, "picks": {slot: v["pick"] for slot, v in sorted(picks.items())}}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def authorise_final_test(out_dir: Path, r_hash: str, expected: list[str], present: list[str],
                         extra: dict | None = None) -> dict:
    """Refuse unless a complete run was selected on validation first; log the read
    (with ``extra`` fields, e.g. the projected test MDE recorded before opening)."""
    missing = sorted(set(expected) - set(present))
    if missing:
        raise RuntimeError(f"the final test needs every arm; missing {missing}")
    sel_path = out_dir / "selection.json"
    if not sel_path.exists():
        raise RuntimeError("no selection.json: run the validation pass first")
    sel = json.loads(sel_path.read_text())
    if sel["results_hash"] != r_hash:
        raise RuntimeError("selection.json was made on different results; rerun the validation pass")
    log = out_dir / "final_test_access.jsonl"
    prior = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    for rec in prior:
        if rec["results_hash"] == r_hash and rec["selection_hash"] != sel["selection_hash"]:
            raise RuntimeError(f"the final test was read on {rec['opened_utc']} under a different selection; "
                               "a selection may not change after the test is opened")
    entry = {"opened_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "results_hash": r_hash,
             "selection_hash": sel["selection_hash"], "picks": {k: v["pick"] for k, v in sel["picks"].items()},
             "git_commit": _git_commit(), "prior_reads": len(prior), **(extra or {})}
    with log.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.warning("FINAL TEST OPENED (read #%d for these results); logged to %s", len(prior) + 1, log)
    return sel


def unpack(results: Path) -> Path:
    """The results folder; a .tar is extracted next to itself, never over a
    different file of the same name (experiment outputs are not overwritten)."""
    if results.is_dir():
        return results
    if results.suffix != ".tar" or not results.exists():
        raise FileNotFoundError(f"{results} is neither a results folder nor a .tar")
    dest = results.with_suffix("")
    dest.mkdir(exist_ok=True)
    with tarfile.open(results) as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            target = dest / m.name
            if target.exists():
                if target.stat().st_size != m.size:
                    raise FileExistsError(f"{target} differs from the copy in {results.name}; move the old "
                                          "results folder aside before unpacking a new run")
                continue
            tar.extract(m, dest, filter="data")
    return dest


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _ci(r) -> str:
    if r is None or not np.isfinite(r["mean"]):
        return "n/a"
    return f"{r['mean']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"


def write_report(path: Path, part: str, ic: pd.DataFrame, paired: pd.DataFrame, topk: pd.DataFrame,
                 picks: dict, arms: list[dict], header: list[str], *, baseline: str = BASELINE,
                 title: str = "Stage 2 evaluation") -> None:
    ic_i = ic[ic["partition"] == part].set_index(["arm", "yardstick"])
    pr_i = paired[paired["partition"] == part].set_index(["arm", "versus", "yardstick"])
    tk_i = topk[topk["partition"] == part].set_index(["arm", "h", "metric"])
    twin = twins(arms)
    order = [a["name"] for a in arms if not a["control"]] + [VOL]

    def get(idx, key):
        return idx.loc[key] if key in idx.index else None

    lines = [f"# {title} -- {part}", "", *header, ""]
    for slot, y in SLOTS.items():
        h = YARDSTICKS[y]
        lines += [f"## {slot} slot: yardstick {y}; pick = **{picks[slot]['pick']}**", "",
                  f"| arm | IC vs {y} | MDE | folds+ | IC - perm twin | IC - {baseline} | top-{CI_K} excess (gross) "
                  f"| top-{CI_K} own net (assumed cost) | top-{CI_K} residualised | top-{CI_K} excess - perm "
                  f"| precision - base | ATR / beta pctile |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in order:
            r = get(ic_i, (a, y))
            if r is None:
                continue
            tilt = [get(tk_i, (a, h, f"{m}_k{CI_K}")) for m in ("atr", "beta")]
            lines.append(" | ".join([
                f"| {a}", _ci(r), f"{r['mde']:.4f}", f"{r['folds_pos']}/{r['n_folds']}",
                _ci(get(pr_i, (a, twin.get(a), y))), _ci(get(pr_i, (a, baseline, y))),
                *[_ci(get(tk_i, (a, h, f"{m}_k{CI_K}"))) for m in ("ex", "net", "rex", "ex_vs_perm", "prec")],
                "/".join(f"{t['mean']:.2f}" if t is not None else "n/a" for t in tilt)]) + " |")
        lines.append("")
        for c in picks[slot]["candidates"]:
            if not c["qualifies"]:
                lines.append(f"- {c['arm']}: {'; '.join(c['why_not'])}")
        lines.append("")
    lines += ["## Top-k depth (gross excess / own net), mean over bars", "",
              "| arm | h | " + " | ".join(f"k={k}" for k in TOP_KS) + " |",
              "|---|---|" + "---|" * len(TOP_KS)]
    for h in TOPK_HORIZONS:
        for a in order:
            cells = []
            for k in TOP_KS:
                ex, net = get(tk_i, (a, h, f"ex_k{k}")), get(tk_i, (a, h, f"net_k{k}"))
                cells.append("n/a" if ex is None else f"{ex['mean']:+.4f} / {net['mean']:+.4f}")
            lines.append(f"| {a} | {h} | " + " | ".join(cells) + " |")
    lines += ["", "## IC on every yardstick (mean [95% CI])", "",
              "| arm | " + " | ".join(YARDSTICKS) + " |", "|---|" + "---|" * len(YARDSTICKS)]
    for a in order + [x["name"] for x in arms if x["control"]]:
        if (a, "L2_10") in ic_i.index:
            lines.append(f"| {a} | " + " | ".join(_ci(get(ic_i, (a, y))) for y in YARDSTICKS) + " |")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(*, results: Path = DEFAULT_RESULTS, manifest_path: Path = MANIFEST, matrix: Path | None = None,
        out_dir: Path = RESULTS_DIR, n_boot: int = 1000, open_final_test: bool = False,
        test_start: str = FINAL_TEST_START) -> dict:
    manifest = json.loads(manifest_path.read_text())
    expected = [a["name"] for a in manifest["arms"]]
    results_dir = unpack(results)
    present = [a for a in expected if (results_dir / f"oof_{a}.parquet").exists()]
    if not present:
        raise FileNotFoundError(f"no oof_<arm>.parquet in {results_dir}")
    if len(present) < len(expected):
        logger.warning("partial run: missing %s", sorted(set(expected) - set(present)))
    arms = [a for a in manifest["arms"] if a["name"] in present]
    out_dir.mkdir(parents=True, exist_ok=True)
    r_hash = results_hash(results_dir, present)
    t0 = pd.Timestamp(test_start, tz="UTC")
    sel = authorise_final_test(out_dir, r_hash, expected, present) if open_final_test else None

    panel = load_panel(results_dir, matrix or manifest_path.parent / manifest["matrix"], present)
    if not open_final_test:
        panel = before(panel, t0)             # test outcomes are never computed on
    parts = ("test",) if open_final_test else ("validation",)
    ic, paired, topk = evaluate(panel, arms, parts=parts, test_start=t0, n_boot=n_boot)

    header = [f"Results: `{results_dir}` (hash {r_hash[:12]}), arms {len(present)}/{len(expected)}, "
              f"git {_git_commit()[:10]}, n_boot {n_boot}, rule {SELECTION_RULE}.",
              f"Validation = OOF bars before {test_start} minus each yardstick's embargo; test = {test_start} on.",
              "Levels on raw returns carry a vol/beta premium: read them against the VOL row and the perm twin."]
    if open_final_test:
        picks = sel["picks"]
        stem = "test"
        header.insert(0, f"**FINAL TEST.** Selection frozen {sel['created_utc']} (hash {sel['selection_hash'][:12]}).")
    else:
        picks = select(ic, paired, arms)
        stem = "validation"
        sel = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule": SELECTION_RULE,
               "results_hash": r_hash, "selection_hash": selection_hash(picks), "arms_present": present,
               "complete": len(present) == len(expected), "git_commit": _git_commit(), "picks": picks}
        (out_dir / "selection.json").write_text(json.dumps(sel, indent=2, default=float))
    for name, df in (("ic", ic), ("paired", paired), ("topk", topk)):
        df[df["partition"] == stem].to_csv(out_dir / f"{stem}_{name}.csv", index=False)
    write_report(out_dir / f"{stem}_report.md", stem, ic, paired, topk, picks, arms, header)
    logger.info("picks: %s -> %s", {k: v["pick"] for k, v in picks.items()}, out_dir / f"{stem}_report.md")
    return {"ic": ic, "paired": paired, "topk": topk, "picks": picks, "results_hash": r_hash}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="stage2_results.tar or its folder")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--matrix", type=Path, default=None, help="default: the manifest's matrix next to it")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--open-final-test", action="store_true",
                    help="read the untouched final test (needs a complete run and a frozen selection; logged)")
    a = ap.parse_args()
    run(results=a.results, manifest_path=a.manifest, matrix=a.matrix, out_dir=a.out, n_boot=a.n_boot,
        open_final_test=a.open_final_test)


if __name__ == "__main__":
    main()
