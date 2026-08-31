"""Is the LABEL wrong, or are the FEATURES wrong? Two questions, two tests.

Stage 6 showed the models are graded at ~15 trading days while the policy exits at
4-5, and that raw forward MFE barely separates the top of the ranking. That is
not yet a diagnosis, because forward MFE in ATR is NOT what these models were
trained to predict. The target is a cross-sectional COMPOSITE of four ranks:

    0.40 * rank(fwd_max_alpha)        # forward max return MINUS SPY's
    0.25 * rank(fwd_atr_adj_return)
    0.20 * rank(trend_persistence)
    0.15 * rank(fwd_max_drawdown, ascending=False)

reproduced here exactly (`LABEL_CONFIG["composite_weights"]`, ranked within each
decision bar, which is what `_cross_sectional_expansion_survival_score` does).

Two separable failures:

  A. The model does NOT rank its own composite  -> the FEATURES do not carry the
     signal. Relabelling changes nothing; a new label on the same features fails
     the same way.
  B. The model DOES rank its composite, but the composite does not translate
     into tradeable move -> the LABEL is the defect. Relabelling is the fix.

Also reported: alpha vs raw return. The composite's biggest weight is already
SPY-relative, so "everything ranked well because the index was up" is a
hypothesis this can settle rather than assume.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from strategies.momentum_expansion.config.momentum_config import LABEL_CONFIG  # noqa: E402

DATA = REPO_ROOT / "research/execution_quality/data"
DAILY = REPO_ROOT / "Data/shared/bars/1d"
HORIZONS = (5, 10, 15, 20)
_cache: dict[str, pd.DataFrame | None] = {}


def daily(ticker: str):
    if ticker in _cache:
        return _cache[ticker]
    path = DAILY / f"{ticker}.parquet"
    df = None
    if path.exists():
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        prev = df["close"].shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                        (df["low"] - prev).abs()], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(14, min_periods=5).mean()
        df["atr_pct"] = df["atr14"] / df["close"]
        df["date"] = df["timestamp"].dt.tz_convert("America/New_York").dt.date
    _cache[ticker] = df
    return df


def components(ticker: str, when: datetime, horizon: int, bench: pd.DataFrame):
    """The label's raw components, computed forward from the session AFTER the
    decision so the decision day's own bar can never leak in."""
    df = daily(ticker)
    if df is None or len(df) < 30:
        return None
    idx = df.index[df["date"] > when.date()]
    if len(idx) == 0:
        return None
    i0 = int(idx[0])
    w = df.iloc[i0:i0 + horizon]
    if len(w) < max(2, horizon // 2):
        return None
    ref = float(df["close"].iloc[i0 - 1]) if i0 > 0 else float(w["close"].iloc[0])
    atr_pct = float(df["atr_pct"].iloc[max(0, i0 - 1)])
    atr = float(df["atr14"].iloc[max(0, i0 - 1)])
    if not (ref > 0 and np.isfinite(atr_pct) and atr_pct > 0 and np.isfinite(atr) and atr > 0):
        return None

    fwd_max_return = float(w["high"].max()) / ref - 1.0
    fwd_max_drawdown = (ref - float(w["low"].min())) / ref
    trend_persistence = float((w["close"] > ref).mean())

    bidx = bench.index[bench["date"] > when.date()]
    if len(bidx) == 0:
        return None
    b0 = int(bidx[0])
    bw = bench.iloc[b0:b0 + horizon]
    bref = float(bench["close"].iloc[b0 - 1]) if b0 > 0 else float(bw["close"].iloc[0])
    bench_fwd_max = float(bw["high"].max()) / bref - 1.0

    return {
        "fwd_max_return": fwd_max_return,
        "fwd_max_alpha": fwd_max_return - bench_fwd_max,
        "fwd_atr_adj_return": fwd_max_return / atr_pct,
        "fwd_max_drawdown": fwd_max_drawdown,
        "trend_persistence": trend_persistence,
        "mfe_atr": (float(w["high"].max()) - ref) / atr,
        "ret_atr": (float(w["close"].iloc[-1]) - ref) / atr,
    }


def composite(group: list[dict], key_prefix: str) -> None:
    """Cross-sectional composite within ONE decision bar — the same construction
    as `_cross_sectional_expansion_survival_score`, which ranks within timestamp."""
    wts = LABEL_CONFIG["composite_weights"]
    n = len(group)
    if n < 4:
        return

    def ranks(col, ascending=True):
        vals = np.array([g[col] for g in group], dtype=float)
        order = pd.Series(vals).rank(pct=True, ascending=ascending).to_numpy()
        return order

    r = (wts["fwd_max_alpha"] * ranks("fwd_max_alpha")
         + wts["fwd_atr_adj_return"] * ranks("fwd_atr_adj_return")
         + wts["trend_persistence"] * ranks("trend_persistence")
         + wts["fwd_max_drawdown"] * ranks("fwd_max_drawdown", ascending=False))
    for g, v in zip(group, r):
        g[key_prefix] = float(v)


def spearman_within_bar(rows, xkey, ykey):
    """Rank correlation computed WITHIN each decision bar, then pooled.

    Pooling raw values across bars would let a market-wide up day masquerade as
    ranking skill; within-bar holds the tape fixed.
    """
    out = []
    bybar = defaultdict(list)
    for r in rows:
        if r.get(xkey) is not None and r.get(ykey) is not None:
            bybar[r["bar"]].append(r)
    for _, g in bybar.items():
        if len(g) < 4:
            continue
        x = pd.Series([v[xkey] for v in g]).rank()
        y = pd.Series([v[ykey] for v in g]).rank()
        if x.std() == 0 or y.std() == 0:
            continue
        out.append(float(np.corrcoef(x, y)[0, 1]))
    if not out:
        return float("nan"), 0
    return float(np.mean(out)), len(out)


def main() -> None:
    bench = daily("SPY")
    if bench is None:
        raise SystemExit("SPY daily bars are required for the alpha component")

    rows = []
    for line in (DATA / "stage2_signal_spine.jsonl").open():
        r = json.loads(line)
        if not r.get("submit") or r.get("score") is None:
            continue
        rows.append(r)

    for h in HORIZONS:
        for r in rows:
            when = datetime.fromisoformat(r["available_at"].replace("Z", "+00:00"))
            c = components(r["ticker"], when, h, bench)
            if c:
                for k, v in c.items():
                    r[f"{k}_{h}"] = v
        bybar = defaultdict(list)
        for r in rows:
            if r.get(f"fwd_max_alpha_{h}") is not None:
                bybar[(r["module"], r["bar"])].append(
                    {**{k: r[f"{k}_{h}"] for k in
                        ("fwd_max_alpha", "fwd_atr_adj_return", "trend_persistence",
                         "fwd_max_drawdown")},
                     "_ref": r})
        for _, g in bybar.items():
            composite(g, "composite")
            for item in g:
                if "composite" in item:
                    item["_ref"][f"composite_{h}"] = item["composite"]

    print(f"signals scored: {len(rows)}\n")
    print("=" * 100)
    print("A. Does the model rank ITS OWN TRAINING TARGET?  (within-bar Spearman,")
    print("   model score vs the reproduced label composite)")
    print("=" * 100)
    print(f"{'module':24s} " + "  ".join(f"{f'{h}d':>10s}" for h in HORIZONS) + "   bars")
    for m in sorted({r["module"] for r in rows}):
        sub = [r for r in rows if r["module"] == m]
        cells, nb = [], 0
        for h in HORIZONS:
            rho, n = spearman_within_bar(sub, "score", f"composite_{h}")
            cells.append(f"{rho:+10.3f}" if not math.isnan(rho) else f"{'-':>10s}")
            nb = max(nb, n)
        print(f"{m:24s} " + "  ".join(cells) + f"   {nb}")

    print("\n" + "=" * 100)
    print("B. Does the model rank TRADEABLE move?  (score vs forward MFE in ATR)")
    print("=" * 100)
    print(f"{'module':24s} " + "  ".join(f"{f'{h}d':>10s}" for h in HORIZONS))
    for m in sorted({r["module"] for r in rows}):
        sub = [r for r in rows if r["module"] == m]
        cells = []
        for h in HORIZONS:
            rho, _ = spearman_within_bar(sub, "score", f"mfe_atr_{h}")
            cells.append(f"{rho:+10.3f}" if not math.isnan(rho) else f"{'-':>10s}")
        print(f"{m:24s} " + "  ".join(cells))

    print("\n" + "=" * 100)
    print("C. Does the COMPOSITE itself predict tradeable move?")
    print("   (if this is high but A is low, the features are the defect;")
    print("    if this is low, the label is the defect however well A scores)")
    print("=" * 100)
    print(f"{'module':24s} " + "  ".join(f"{f'{h}d':>10s}" for h in HORIZONS))
    for m in sorted({r["module"] for r in rows}):
        sub = [r for r in rows if r["module"] == m]
        cells = []
        for h in HORIZONS:
            rho, _ = spearman_within_bar(sub, f"composite_{h}", f"mfe_atr_{h}")
            cells.append(f"{rho:+10.3f}" if not math.isnan(rho) else f"{'-':>10s}")
        print(f"{m:24s} " + "  ".join(cells))

    print("\n" + "=" * 100)
    print("D. Raw return vs ALPHA — is 'it went up' just 'the index went up'?")
    print("=" * 100)
    for h in HORIZONS:
        raw = [r.get(f"fwd_max_return_{h}") for r in rows if r.get(f"fwd_max_return_{h}") is not None]
        alpha = [r.get(f"fwd_max_alpha_{h}") for r in rows if r.get(f"fwd_max_alpha_{h}") is not None]
        if not raw:
            continue
        print(f"  {h:2d}d  median fwd_max_return={np.median(raw):+.3%}   "
              f"median fwd_max_alpha={np.median(alpha):+.3%}   "
              f"share alpha>0 = {np.mean([a > 0 for a in alpha]):.1%}")

    out = DATA / "stage7_label_diagnosis.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
