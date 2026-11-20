"""Theme grouping stability experiment (2026-11-18). No API calls, CPU only.

Question: would clustering on stable business inputs stop the weekly theme churn
(19/20 mega caps changed primary theme in 5 weeks; consecutive-week ARI 0.24-0.48)
without losing grouping quality?

Variants (universe = the 2026-11-16 weekly run's 2,903 tickers):
  PROD  production embedding as written on 2026-11-16 (ticker_embeddings.parquet):
        concat[0.6 * (0.6 desc + 0.4 news), 0.4 * 60d co-movement PCA]
  B1    BGE embedding of the business description only (what PROD's desc part uses)
  B2    BGE embedding of "Sector: S. Industry: I. " + description

Each variant is clustered exactly as step03 does (UMAP 20-d, n_neighbors 15,
min_dist 0, cosine, random_state; HDBSCAN leaf, min_cluster_size 5, min_samples 2)
and every ticker is assigned to its nearest cluster centroid in embedding space,
as step08 does (primary theme = argmax cosine).

Stability (inputs held fixed, so this isolates the algorithm):
  * UMAP seed 42 vs 7 and 123
  * 5 runs dropping a random 3% of tickers (promotions/delistings), seed 42
  Measured as ARI vs the base run on common tickers, plus top-20-by-market-cap
  group-mate Jaccard (a "flip" = Jaccard < 0.5).
Quality:
  * agreement with the hand-curated theme_map_v4 (label_checks.v4_agreement)
  * forward cohesion: mean within-group pairwise correlation of daily returns over
    the 20 trading days after 09-21, 09-28, 10-05, 10-12, against Yahoo-sector and
    size-matched random groupings. PROD is represented there by the grouping that
    was actually published that week. Daily |return| > 40% is set to NaN
    (corporate actions / stale prints in the mixed-adjustment bar cache) and a
    ticker needs >= 15 valid days in the window.
  * HDBSCAN noise share, group count, top1-top2 centroid margins, mega-cap mates.

Caveats: B1/B2 use today's descriptions (business text rarely changes; no price
information, so no return leakage). Published PROD groupings include trailing 60d
co-movement, which mechanically favours forward correlation.

Writes only under research/theme_stability_2026-11-18/.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from themes.dynamic_theme import config as C  # noqa: E402
from themes.dynamic_theme import label_checks as lc  # noqa: E402
from themes.dynamic_theme.profiles import load_profiles  # noqa: E402
from themes.dynamic_theme.sklearn_compat import load_hdbscan, load_umap  # noqa: E402
from themes.dynamic_theme.stages.step02_embed import _clean_text  # noqa: E402

OUT = Path(__file__).resolve().parent
BASE_SEED = 42
ALT_SEEDS = [7, 123]
DROP_FRAC = 0.03
DROP_TRIALS = 5
COHESION_DATES = ["2026-09-21", "2026-09-28", "2026-10-05", "2026-10-12"]
WINDOW = 20
MIN_VALID_DAYS = 15
MAX_ABS_RET = 0.40
MAX_PAIRS_PER_GROUP = 300
MEGA = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "JPM", "XOM", "LLY", "WMT"]

log = logging.getLogger("theme_stability")


# ── inputs ─────────────────────────────────────────────────────────────────────
def load_inputs():
    clusters = pd.read_parquet(C.TICKER_CLUSTERS_PATH)
    universe = clusters["ticker"].astype(str).tolist()
    docs = pd.read_parquet(C.TICKER_DOCUMENTS_PATH).drop_duplicates("ticker").set_index("ticker")
    prof = load_profiles().drop_duplicates("ticker").set_index("ticker")
    emb = pd.read_parquet(C.TICKER_EMBEDDINGS_PATH).drop_duplicates("ticker").set_index("ticker")
    return universe, docs, prof, emb


def texts_for(universe, docs, prof):
    desc, pref = {}, {}
    for t in universe:
        d = docs["description"].get(t) if "description" in docs.columns else None
        d = _clean_text(str(d)) if isinstance(d, str) else ""
        if not d and isinstance(prof["longBusinessSummary"].get(t), str):
            d = _clean_text(prof["longBusinessSummary"].get(t))
        if not d:
            continue
        sector = prof["sector"].get(t) if t in prof.index else None
        industry = prof["industry"].get(t) if t in prof.index else None
        head = " ".join(
            f"{k}: {v}." for k, v in (("Sector", sector), ("Industry", industry)) if isinstance(v, str) and v
        )
        desc[t] = d
        pref[t] = f"{head} {d}".strip()
    return desc, pref


def embed(texts: dict[str, str]) -> tuple[list[str], np.ndarray]:
    from signals.news.nlp import embed_texts_bge

    tickers = sorted(texts)
    vecs = embed_texts_bge([texts[t] for t in tickers], model_name=C.BGE_MODEL, device="cpu")
    return tickers, np.asarray(vecs, dtype=np.float32)


# ── clustering exactly as step03 + assignment as step08 ───────────────────────
def cluster_and_assign(tickers: list[str], matrix: np.ndarray, seed: int) -> dict:
    umap, hdbscan_lib = load_umap(), load_hdbscan()
    n_components = min(C.UMAP_N_COMPONENTS, matrix.shape[1] - 1, len(tickers) - 2)
    reduced = umap.UMAP(
        n_components=n_components, n_neighbors=C.UMAP_N_NEIGHBORS, min_dist=C.UMAP_MIN_DIST,
        metric=C.UMAP_METRIC, random_state=seed, low_memory=False,
    ).fit_transform(matrix).astype(np.float32)
    labels = hdbscan_lib.HDBSCAN(
        min_cluster_size=C.HDBSCAN_MIN_CLUSTER_SIZE, min_samples=C.HDBSCAN_MIN_SAMPLES,
        metric=C.HDBSCAN_METRIC, cluster_selection_method=C.HDBSCAN_CLUSTER_SELECTION_METHOD,
    ).fit(reduced).labels_
    ids = sorted(set(labels) - {-1})
    norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    cents = np.stack([norm[labels == i].mean(axis=0) for i in ids])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-10
    sims = norm @ cents.T
    order = np.argsort(-sims, axis=1)
    primary = np.array(ids)[order[:, 0]]
    rows = np.arange(len(tickers))
    margin = sims[rows, order[:, 0]] - sims[rows, order[:, 1]]
    return {
        "primary": pd.Series(primary, index=tickers),
        "margin": pd.Series(margin, index=tickers),
        "noise_share": float((labels == -1).mean()),
        "n_clusters": len(ids),
    }


def mates(primary: pd.Series, ticker: str, common: set[str]) -> set[str]:
    if ticker not in primary.index:
        return set()
    g = primary[ticker]
    return set(primary.index[primary == g]) & common - {ticker}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def compare(base: pd.Series, other: pd.Series, top20: list[str]) -> dict:
    common = sorted(set(base.index) & set(other.index))
    cs = set(common)
    jac = [jaccard(mates(base, t, cs), mates(other, t, cs)) for t in top20 if t in cs]
    return {
        "ari": float(adjusted_rand_score(base[common], other[common])),
        "top20_mean_jaccard": float(np.mean(jac)) if jac else None,
        "top20_flips": int(sum(j < 0.5 for j in jac)),
    }


# ── quality ───────────────────────────────────────────────────────────────────
def load_returns(tickers: list[str]) -> pd.DataFrame:
    closes = {}
    for t in tickers:
        path = C.DAILY_BARS_DIR / f"{t}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["timestamp", "close"])
        except Exception:
            continue
        ts = pd.to_datetime(df["timestamp"])
        ts = ts.dt.tz_convert(None) if ts.dt.tz is not None else ts
        closes[t] = pd.Series(df["close"].values, index=ts.dt.normalize()).groupby(level=0).last()
    prices = pd.DataFrame(closes).sort_index()
    rets = prices.pct_change(fill_method=None)
    return rets.mask(rets.abs() > MAX_ABS_RET)


def cohesion(groups: pd.Series, window: pd.DataFrame, rng: np.random.Generator) -> dict:
    valid = window.columns[window.notna().sum() >= MIN_VALID_DAYS]
    g = groups[groups.index.isin(valid)]
    pair_vals, group_means = [], []
    for _, members in g.groupby(g):
        m = list(members.index)
        if len(m) < 3:
            continue
        corr = window[m].corr(min_periods=MIN_VALID_DAYS).to_numpy()
        iu = np.triu_indices(len(m), k=1)
        vals = corr[iu]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        if len(vals) > MAX_PAIRS_PER_GROUP:
            vals = rng.choice(vals, MAX_PAIRS_PER_GROUP, replace=False)
        pair_vals.extend(vals.tolist())
        group_means.append(float(vals.mean()))
    return {"pair_mean": float(np.mean(pair_vals)) if pair_vals else None,
            "group_mean": float(np.mean(group_means)) if group_means else None,
            "groups": len(group_means), "tickers": int(len(g))}


def published_groupings() -> dict[str, pd.Series]:
    h = pd.read_parquet(C.OUTPUTS_DIR / "ticker_theme_membership_history.parquet",
                        columns=["as_of", "ticker", "theme", "membership_score"])
    h["as_of"] = pd.to_datetime(h["as_of"]).dt.strftime("%Y-%m-%d")
    top = h.sort_values("membership_score", ascending=False).drop_duplicates(["as_of", "ticker"])
    return {d: g.set_index("ticker")["theme"] for d, g in top.groupby("as_of")}


def v4_score(primary: pd.Series, v4) -> float | None:
    df = pd.DataFrame({"ticker": primary.index, "primary_theme": primary.astype(str).values})
    return lc.v4_agreement(df, v4).get("agreement")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    universe, docs, prof, emb = load_inputs()
    desc, pref = texts_for(universe, docs, prof)
    log.info("universe %d | with description %d", len(universe), len(desc))

    variants: dict[str, tuple[list[str], np.ndarray]] = {}
    prod_t = [t for t in universe if t in emb.index]
    variants["PROD"] = (prod_t, np.array(emb.loc[prod_t, "embedding"].tolist(), dtype=np.float32))
    variants["B1"] = embed(desc)
    variants["B2"] = embed(pref)
    log.info("embedded B1/B2 in %.0fs", time.time() - t0)

    caps = pd.to_numeric(prof["marketCap"], errors="coerce").reindex(universe)
    top20 = caps.dropna().sort_values(ascending=False).head(20).index.tolist()
    rng = np.random.default_rng(BASE_SEED)
    v4 = lc.load_v4_map()

    results: dict = {"top20": top20, "variants": {}}
    runs = []
    base_runs = {}
    for name, (tickers, matrix) in variants.items():
        log.info("== %s: %d tickers, dim %d", name, len(tickers), matrix.shape[1])
        base = cluster_and_assign(tickers, matrix, BASE_SEED)
        base_runs[name] = base
        for seed in ALT_SEEDS:
            other = cluster_and_assign(tickers, matrix, seed)
            runs.append({"variant": name, "test": f"seed_{seed}", **compare(base["primary"], other["primary"], top20)})
        for trial in range(DROP_TRIALS):
            keep = np.sort(rng.choice(len(tickers), int(round(len(tickers) * (1 - DROP_FRAC))), replace=False))
            other = cluster_and_assign([tickers[i] for i in keep], matrix[keep], BASE_SEED)
            runs.append({"variant": name, "test": f"drop3pct_{trial}", **compare(base["primary"], other["primary"], top20)})
        m = base["margin"]
        results["variants"][name] = {
            "tickers": len(tickers), "noise_share": base["noise_share"], "n_clusters": base["n_clusters"],
            "groups_used": int(base["primary"].nunique()),
            "margin_median": float(m.median()), "margin_share_lt_0.01": float((m < 0.01).mean()),
            "mega_margins": {t: round(float(m[t]), 4) for t in MEGA if t in m.index},
            "v4_agreement": v4_score(base["primary"], v4),
        }
        log.info("   %s done (%.0fs elapsed)", name, time.time() - t0)

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(OUT / "stability_runs.csv", index=False)
    results["stability"] = (runs_df.assign(kind=runs_df["test"].str.extract(r"^(seed|drop3pct)")[0])
                            .groupby(["variant", "kind"])[["ari", "top20_mean_jaccard", "top20_flips"]]
                            .mean().round(3).reset_index().to_dict("records"))

    # published production groupings: observed weekly churn + quality
    pub = published_groupings()
    weeks = sorted(pub)[-6:]
    results["published_weekly"] = [
        {"from": a, "to": b, **compare(pub[a], pub[b], top20)} for a, b in zip(weeks, weeks[1:])
    ]
    results["published_v4_agreement"] = {w: v4_score(pub[w], v4) for w in weeks}

    rets = load_returns(universe)
    sector = prof["sector"].reindex(universe).dropna()
    sector = sector[sector.astype(str).str.len() > 0]
    coh = []
    for d in COHESION_DATES:
        after = rets.index[rets.index > pd.Timestamp(d)][:WINDOW]
        if len(after) < WINDOW:
            continue
        window = rets.loc[after]
        groupings = {"PUBLISHED": pub.get(d), "B1": base_runs["B1"]["primary"],
                     "B2": base_runs["B2"]["primary"], "SECTOR": sector}
        for name, grp in groupings.items():
            if grp is not None:
                coh.append({"date": d, "grouping": name, **cohesion(grp, window, rng)})
        b2 = base_runs["B2"]["primary"]
        for k in range(3):
            shuffled = pd.Series(rng.permutation(b2.values), index=b2.index)
            coh.append({"date": d, "grouping": f"RANDOM_{k}", **cohesion(shuffled, window, rng)})
    coh_df = pd.DataFrame(coh)
    coh_df.to_csv(OUT / "cohesion.csv", index=False)
    coh_df["g"] = coh_df["grouping"].str.replace(r"_\d$", "", regex=True)
    results["cohesion_mean_over_dates"] = (coh_df.groupby("g")[["pair_mean", "group_mean", "groups", "tickers"]]
                                           .mean().round(4).reset_index().to_dict("records"))

    # mega-cap mates (largest 6 co-members) per grouping
    mate_rows = []
    groupings = {"PUBLISHED_11-03": pub[weeks[-1]], "PROD_11-16": base_runs["PROD"]["primary"],
                 "B1": base_runs["B1"]["primary"], "B2": base_runs["B2"]["primary"]}
    for t in MEGA:
        for name, grp in groupings.items():
            if t not in grp.index:
                continue
            co = grp.index[(grp == grp[t]) & (grp.index != t)]
            top_mates = caps.reindex(co).sort_values(ascending=False).head(6).index.tolist()
            mate_rows.append({"ticker": t, "grouping": name, "group_size": int(len(co) + 1),
                              "largest_mates": " ".join(top_mates)})
    pd.DataFrame(mate_rows).to_csv(OUT / "megacap_mates.csv", index=False)

    results["params"] = {"base_seed": BASE_SEED, "alt_seeds": ALT_SEEDS, "drop_frac": DROP_FRAC,
                         "drop_trials": DROP_TRIALS, "window": WINDOW, "min_valid_days": MIN_VALID_DAYS,
                         "max_abs_ret": MAX_ABS_RET, "bge": C.BGE_MODEL, "runtime_s": round(time.time() - t0)}
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    log.info("done in %.0fs -> %s", time.time() - t0, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
