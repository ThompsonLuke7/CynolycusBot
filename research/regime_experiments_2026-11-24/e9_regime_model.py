"""E9: a trained regime model vs the hand vote (PLAN.md section 4, pre-registered 2026-11-25
~00:47 ET; primaries pinned ~00:52 ET, before any result here was computed).

Models (walk-forward, yearly refit on an expanding window, nothing fit on the test year):
  HMM-R K=2 (PRIMARY), K=3   Gaussian HMM on Nasdaq-100 daily log returns (x100), 1986+
  HMM-M K=3                  diagonal Gaussian HMM on [NDX return, 5d change in log VIX, Baa
                             spread, 10y-3m curve], each standardised on the training rows, 1991+
  SUP-LGB-T1 (PRIMARY), SUP-LR-T1, SUP-LGB-T2, SUP-LR-T2, SUP-LGB-T1+sectors
                             supervised score: P(target) from base + vix, credit, rates, dollar,
                             relative, claims (+ sector correlation), E3 features; LR is clipped
HMM states use FILTERED probabilities only (forward algorithm on data <= t with parameters fit
before the test year), never smoothed or Viterbi paths. States are ordered by training return
variance. EM: 5 seeded random starts at the first fit, warm start afterwards.

Books (E2 execution: decide at close t, trade at close t+1, 2 bp, cash earns bills):
  HMM  size = 1 - 0.75 * P(highest-variance state); long/flat = flat when that P > 0.5
  SUP  training-window percentile of the score -> 1.0 (<80th), 0.8 (80-90), 0.5 (90-97),
       0.25 (>=97th); long/flat = flat above the 80th
Benchmarks on each model's own out-of-sample span: buy-and-hold, LH vote x live multipliers,
LH vote long/flat, 15% vol target.

Decision (fixed): a model is PREFERRED over the vote if its sized book (i) beats the vote book's
Sharpe with the 95% CI of the difference above zero, or (ii) beats the vote book's MDD in >= 3 of 4
out-of-sample crashes with a Sharpe-difference CI not below zero; it must also do so against the
15% vol target to be worth its complexity. A secondary variant passing alone does not displace the
vote (about 8 variants).

Fixes and follow-ups after the first run (2026-11-25 ~00:52 ET), labelled:
  * BUG FIX: books were selected by name prefix, so "sup_lgb_t1_" also caught "sup_lgb_t1_sect_*"
    and the primary SUP model was scored on 2004+ only. Now exact names.
  * FOLLOW-UP fu_oofcut_*: the in-sample percentile cutoffs left the SUP books ~94-99% invested
    (in-sample scores, LightGBM's especially, are more extreme than out-of-sample ones). The
    follow-up takes the cutoffs from blocked 4-fold out-of-fold scores inside each training window
    (purged by the horizon). Not part of the pre-registered decision.

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e9_regime_model.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_general_policy as e2  # noqa: E402
import e3_factor_completeness as e3  # noqa: E402
import regime_common as rc  # noqa: E402

SEED = 20261125
LONG_FAMS = ("vix", "credit", "rates", "dollar", "relative", "claims")
SUP_CUTS = (0.80, 0.90, 0.97)
SUP_SIZES = (1.0, 0.8, 0.5, 0.25)


# ---------------------------------------------------------------- Gaussian HMM (diagonal)
class GaussianHMM:
    def __init__(self, k: int, seed: int):
        self.k, self.rng = k, np.random.default_rng(seed)

    def _logb(self, X: np.ndarray) -> np.ndarray:
        var = self.var[None, :, :]
        d = X[:, None, :] - self.mu[None, :, :]
        return -0.5 * (np.log(2 * np.pi * var) + d * d / var).sum(axis=2)

    def init_random(self, X: np.ndarray) -> None:
        k, n = self.k, len(X)
        vol = pd.Series(np.abs(X[:, 0])).rolling(20, min_periods=1).mean().to_numpy()
        cuts = np.quantile(vol, np.linspace(0, 1, k + 1))
        lab = np.clip(np.searchsorted(cuts, vol, side="right") - 1, 0, k - 1)
        lab = np.where(self.rng.random(n) < 0.1, self.rng.integers(0, k, n), lab)
        self.mu = np.array([X[lab == j].mean(axis=0) if (lab == j).any() else X.mean(axis=0) for j in range(k)])
        self.var = np.array([X[lab == j].var(axis=0) + 1e-6 if (lab == j).sum() > 1 else X.var(axis=0) for j in range(k)])
        self.A = np.full((k, k), 0.02 / max(k - 1, 1)) + np.eye(k) * (0.98 - 0.02 / max(k - 1, 1))
        self.pi = np.full(k, 1 / k)

    def _forward(self, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, k = B.shape
        alpha = np.empty((n, k))
        c = np.empty(n)
        a = self.pi * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        A = self.A
        for t in range(1, n):
            a = (alpha[t - 1] @ A) * B[t]
            s = a.sum()
            c[t] = s
            alpha[t] = a / s
        return alpha, c

    def _emis(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lb = self._logb(X)
        m = lb.max(axis=1, keepdims=True)
        return np.exp(lb - m), m[:, 0]

    def fit(self, X: np.ndarray, iters: int = 60, tol: float = 1e-4) -> float:
        floor = X.var(axis=0) * 1e-3
        prev = -np.inf
        for _ in range(iters):
            B, m = self._emis(X)
            alpha, c = self._forward(B)
            ll = float(np.log(c).sum() + m.sum())
            n, k = B.shape
            beta = np.empty((n, k))
            beta[-1] = 1.0
            A = self.A
            for t in range(n - 2, -1, -1):
                beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = (alpha[:-1, :, None] * A[None, :, :] * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
            self.pi = gamma[0]
            self.A = xi.sum(axis=0) / xi.sum(axis=(0, 2))[:, None]
            w = gamma.sum(axis=0)
            self.mu = (gamma.T @ X) / w[:, None]
            self.var = np.maximum((gamma.T @ (X * X)) / w[:, None] - self.mu ** 2, floor)
            if ll - prev < tol * abs(ll):
                break
            prev = ll
        return ll

    def filtered(self, X: np.ndarray) -> np.ndarray:
        B, _ = self._emis(X)
        return self._forward(B)[0]

    def order(self) -> np.ndarray:
        return np.argsort(self.var[:, 0])  # ascending return variance


def hmm_walk_forward(X: np.ndarray, idx: pd.DatetimeIndex, k: int, first_test: int,
                     window_years: int | None = None) -> tuple[np.ndarray, list]:
    """P(highest-variance state) filtered through each test year with parameters fit before it."""
    years = idx.year.to_numpy()
    p_hi = np.full(len(X), np.nan)
    model, log = None, []
    for yr in range(first_test, years.max() + 1):
        tr = years < yr
        if window_years is not None:  # E10: most recent N years only
            tr &= years >= yr - window_years
        te = years == yr
        if te.sum() == 0:
            continue
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
        Z = (X - mu) / sd
        if model is None:
            best = None
            for s in range(5):
                m = GaussianHMM(k, SEED + s)
                m.init_random(Z[tr])
                ll = m.fit(Z[tr])
                if best is None or ll > best[0]:
                    best = (ll, m)
            model = best[1]
        else:
            model.fit(Z[tr], iters=30)
        upto = np.where(years <= yr)[0]
        post = model.filtered(Z[upto])
        hi = model.order()[-1]
        p_hi[te] = post[te[upto]][:, hi]
        log.append({"year": yr, "k": k, "hi_state_sd": float(np.sqrt(model.var[hi, 0]) * sd[0]),
                    "lo_state_sd": float(np.sqrt(model.var[model.order()[0], 0]) * sd[0]),
                    "hi_persistence": float(model.A[hi, hi])})
    return p_hi, log


# ---------------------------------------------------------------- supervised score
def _fit_score(model: str, Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    if model == "lr":
        lo, hi = np.percentile(Xtr, 1, axis=0), np.percentile(Xtr, 99, axis=0)
        Xtr, Xte = np.clip(Xtr, lo, hi), np.clip(Xte, lo, hi)
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xtr))[:, 1], m.predict_proba(sc.transform(Xte))[:, 1]
    m = lgb.LGBMClassifier(**{**e3.LGB_PARAMS, "n_jobs": 2}).fit(Xtr, ytr)
    return m.predict_proba(Xtr)[:, 1], m.predict_proba(Xte)[:, 1]


def inner_oof_scores(model: str, X: np.ndarray, y: np.ndarray, purge: int, folds: int = 4) -> np.ndarray:
    """Follow-up: out-of-fold scores on the training window (blocked folds, purged both sides)."""
    n = len(X)
    edges = np.linspace(0, n, folds + 1).astype(int)
    out = np.full(n, np.nan)
    for a, b in zip(edges[:-1], edges[1:]):
        tr = np.r_[np.arange(0, max(a - purge, 0)), np.arange(min(b + purge, n), n)]
        if len(tr) < 500 or len(np.unique(y[tr])) < 2:
            continue
        out[a:b] = _fit_score(model, X[tr], y[tr], X[a:b])[1]
    return out[~np.isnan(out)]


def sup_walk_forward(F: pd.DataFrame, y: pd.Series, cols: list[str], model: str, purge: int,
                     cut_source: str = "insample"):
    sub, ys, years, test_years, _ = e3.span_setup(F, y, cols)
    X = sub[cols].to_numpy()
    score = np.full(len(sub), np.nan)
    size = np.full(len(sub), np.nan)
    for yr in test_years:
        te = np.where(years == yr)[0]
        if len(te) == 0:
            continue
        tr = np.arange(0, max(te[0] - purge, 0))
        tr = tr[~np.isnan(ys[tr])]
        Xtr, Xte = X[tr], X[te]
        s_tr, s_te = _fit_score(model, Xtr, ys[tr], Xte)
        if cut_source == "oof":
            s_tr = inner_oof_scores(model, Xtr, ys[tr], purge)
        cuts = np.quantile(s_tr, SUP_CUTS)
        score[te] = s_te
        size[te] = np.array(SUP_SIZES)[np.searchsorted(cuts, s_te, side="right")]
    return pd.DataFrame({"score": score, "size": size}, index=sub.index)


# ---------------------------------------------------------------- evaluation
def books_on_span(weights: dict[str, pd.Series], start, end) -> tuple[dict, pd.DataFrame]:
    p, _ = rc.latest_panel()
    fr, _ = rc.build_frame("ndx")
    idx = fr.loc[start:end].index
    vote = e2.target_weights(fr, p["ndx"]).reindex(idx)
    w = pd.DataFrame({"bh": 1.0, "vote_size": vote["size_regime"], "vote_longflat": vote["longflat_v4"],
                      "voltarget15": vote["voltarget15"]}, index=idx)
    for k, v in weights.items():
        w[k] = v.reindex(idx)
    w = w.dropna()
    close = p["ndx"].reindex(w.index)
    r = close.pct_change().fillna(0.0).to_numpy()
    rf = (p["tbill3m"].shift(1).reindex(w.index) / 100 / 252).to_numpy()
    rets = pd.DataFrame({c: e2.run_book(w[c].to_numpy(), r, rf) for c in w.columns}, index=w.index)
    rets["bh"] = r
    rng = np.random.default_rng(SEED)
    out = {}
    crashes = e2.crash_table(close, rets, e2.CRASH_WINDOWS)
    for c in w.columns:
        sig = w[c].to_numpy()
        pos = np.concatenate([[sig[0]] * 2, sig[:-2]])
        m = e2.metrics(rets[c].to_numpy(), rf, pos)
        if c in weights:
            ex = rets[c].to_numpy() - rf
            m["sharpe_minus_vote_ci"] = e2.block_ci_sharpe_diff(ex, rets["vote_size"].to_numpy() - rf, rng)
            m["sharpe_minus_voltarget_ci"] = e2.block_ci_sharpe_diff(ex, rets["voltarget15"].to_numpy() - rf, rng)
            m["sharpe_minus_bh_ci"] = e2.block_ci_sharpe_diff(ex, rets["bh"].to_numpy() - rf, rng)
            m["crash_wins_vs_vote"] = f"{int((crashes[f'{c}_mdd'] > crashes['vote_size_mdd']).sum())}/{len(crashes)}"
            m["crash_wins_vs_voltarget"] = f"{int((crashes[f'{c}_mdd'] > crashes['voltarget15_mdd']).sum())}/{len(crashes)}"
            shifts = rng.integers(250, len(sig) - 250, 500)
            null = [e2.mdd(e2.run_book(np.roll(sig, s), r, rf)) for s in shifts]
            m["mdd_placebo_pct"] = float((np.array(null) < m["mdd"]).mean() * 100)
        out[c] = m
    return out, crashes


def decide(m: dict, vs: str) -> bool:
    """(i) Sharpe CI vs `vs` above zero, or (ii) >= 3/4 crash wins with the CI not below zero."""
    lo, hi = m[f"sharpe_minus_{vs}_ci"]
    wins, n = map(int, m[f"crash_wins_vs_{vs}"].split("/"))
    return bool(lo > 0 or (wins >= 0.75 * n and hi >= 0))


def description(score_state: pd.Series, start, end) -> dict:
    fr, _ = rc.build_frame("ndx")
    g = fr.loc[start:end].join(score_state.rename("st"), how="inner").dropna(subset=["st", "fwd_mdd20", "vote7"])
    rng = np.random.default_rng(SEED)
    out = {}
    for dec, d in g.groupby(rc.decade_of(g.index).to_numpy()):
        if len(d) < 500:
            continue
        codes = pd.factorize(d["st"])[0]
        vcodes = pd.factorize(d["state"])[0]
        y = d["fwd_mdd20"].to_numpy()
        null = rc.circular_shift_null(codes, y, rc.eta2, 500, rng, 60)
        out[dec] = {"model_eta2": rc.eta2(codes, y), "model_null_p95": float(np.percentile(null, 95)),
                    "vote_eta2": rc.eta2(vcodes, y)}
    return out


def main() -> int:
    warnings.simplefilter("ignore")
    rc.OUT.mkdir(exist_ok=True)
    p, panel_name = rc.latest_panel()
    weights, states, logs, spans = {}, {}, [], {}
    # HMMs
    r = np.log(p["ndx"]).diff() * 100
    hmm_r = pd.DataFrame({"r": r}).dropna()
    for k in (2, 3):
        ph, lg = hmm_walk_forward(hmm_r.to_numpy(), hmm_r.index, k, 1991)
        s = pd.Series(ph, index=hmm_r.index).dropna()
        name = f"hmm_r{k}"
        weights[f"{name}_size"], weights[f"{name}_longflat"] = 1 - 0.75 * s, (s <= 0.5).astype(float)
        states[name] = (s > 0.5).astype(int)
        logs += [{**x, "model": name} for x in lg]
        spans[name] = s.index[0]
        print(name, "done", s.index[0].date(), flush=True)
    m_feat = pd.DataFrame({"r": r, "dvix": np.log(p["vix"]).diff(5), "dbaa": p["baa_10y_spread"].diff(5),
                           "dcurve": (p["ust10y"] - p["tbill3m"]).diff(5)}).dropna()
    ph, lg = hmm_walk_forward(m_feat.to_numpy(), m_feat.index, 3, 1995)
    s = pd.Series(ph, index=m_feat.index).dropna()
    weights["hmm_m3_size"], weights["hmm_m3_longflat"] = 1 - 0.75 * s, (s <= 0.5).astype(float)
    states["hmm_m3"] = (s > 0.5).astype(int)
    logs += [{**x, "model": "hmm_m3"} for x in lg]
    spans["hmm_m3"] = s.index[0]
    print("hmm_m3 done", flush=True)
    # supervised
    F, fam, _, _ = e3.build_features("ndx")
    close = p["ndx"].dropna()
    ys = {}
    for tname, h, thr in (("T1", 20, -0.05), ("T2", 60, -0.10)):
        mdd = rc.forward_targets(close, h).reindex(F.index)["fwd_mdd20"]
        ys[tname] = ((mdd <= thr).astype(float).where(mdd.notna()), h)
    long_cols = fam["base"] + [c for k in LONG_FAMS for c in fam[k]]
    variants = [("sup_lgb_t1", "T1", "lgb", long_cols, "insample"), ("sup_lr_t1", "T1", "lr", long_cols, "insample"),
                ("sup_lgb_t2", "T2", "lgb", long_cols, "insample"), ("sup_lr_t2", "T2", "lr", long_cols, "insample"),
                ("sup_lgb_t1_sect", "T1", "lgb", long_cols + fam["sectors_corr"], "insample")]
    # follow-up (labelled, after the first run): cutoffs from inner out-of-fold training scores
    variants += [(f"fu_oofcut_{n[4:]}", t, m, c, "oof") for n, t, m, c, _ in variants[:4]]
    for name, tname, model, cols, cut_source in variants:
        y, h = ys[tname]
        out = sup_walk_forward(F, y, cols, model, h, cut_source).dropna()
        weights[f"{name}_size"] = out["size"]
        weights[f"{name}_longflat"] = (out["size"] >= 1.0).astype(float)
        states[name] = out["size"]
        spans[name] = out.index[0]
        print(name, "done", out.index[0].date(), flush=True)
    pd.DataFrame(logs).to_csv(rc.OUT / "e9_hmm_fit_log.csv", index=False)
    # evaluate each model family on its own out-of-sample span
    results, rows = {}, []
    end = p.index[-1]
    for name, start in spans.items():
        ws = {k: weights[k] for k in (f"{name}_size", f"{name}_longflat")}  # exact (prefix matching mixed t1/t1_sect)
        books, crashes = books_on_span(ws, start, end)
        crashes.to_csv(rc.OUT / f"e9_crashes_{name}.csv", index=False)
        desc = description(states[name], start, end)
        primary = name in ("hmm_r2", "sup_lgb_t1")
        sized = books[f"{name}_size"]
        results[name] = {"start": str(start.date()), "primary": primary,
                         "PREFERRED_vs_vote": decide(sized, "vote"),
                         "beats_voltarget": decide(sized, "voltarget"),
                         "books": books, "description": desc}
        for b, m in books.items():
            rows.append({"model": name, "book": b, "start": str(start.date()),
                         **{k: (str(v) if isinstance(v, tuple) else v) for k, v in m.items()}})
        print(name, "preferred_vs_vote", results[name]["PREFERRED_vs_vote"], "beats_voltarget",
              results[name]["beats_voltarget"], flush=True)
    pd.DataFrame(rows).to_csv(rc.OUT / "e9_books.csv", index=False)
    (rc.OUT / "e9_summary.json").write_text(json.dumps({"panel": panel_name, "models": results}, indent=2, default=str))
    pd.set_option("display.width", 250)
    t = pd.DataFrame(rows)
    print(t[["model", "book", "start", "cagr", "sharpe", "mdd", "invested", "sharpe_minus_vote_ci",
             "crash_wins_vs_vote", "sharpe_minus_voltarget_ci", "crash_wins_vs_voltarget", "mdd_placebo_pct"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
