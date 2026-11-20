"""Follow-up 2: how much price co-movement to keep in the identity vector?

Themes are meant to catch cross-industry trades (AI infrastructure, defensives,
vertically integrated chains), which business text alone cannot see. This sweeps
the co-movement weight w in the production blend
    identity = normalize(concat[(1 - w) * B2_text, w * comovement_pca20])
with B2_text = BGE("Sector: S. Industry: I. " + description) and HDBSCAN eom /
min_cluster_size 5, everything else as step03/step08.

Co-movement is built as of 2026-10-05 (60-day lookback ending then), so the forward
cohesion window (20 trading days after 10-05) does not overlap it. "Monthly re-fit
churn" re-builds co-movement as of 2026-11-03 and compares the seed-42 groupings.

Basket co-membership = share of basket pairs placed in the same group (higher means
the cross-industry theme exists as a group). Cross-sector share = share of grouped
tickers whose group's largest sector holds < 60% of its members.
Writes comovement_sweep.csv and comovement_mates.csv next to this file.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

import run_experiment as rx  # noqa: E402
from themes.dynamic_theme import config as C  # noqa: E402
from themes.dynamic_theme import label_checks as lc  # noqa: E402
from themes.dynamic_theme.sklearn_compat import load_hdbscan, load_umap  # noqa: E402
from themes.dynamic_theme.stages.step02_embed import _build_comovement_vectors  # noqa: E402

WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5]
SEEDS = [42, 7, 123]
AS_OF = pd.Timestamp("2026-10-05")
AS_OF_NEXT = pd.Timestamp("2026-11-03")
AI_INFRA = ["NVDA", "AVGO", "AMD", "TSM", "MU", "ANET", "SMCI", "DELL", "VRT", "ETN", "GEV",
            "CEG", "VST", "ORCL", "CRWV", "NBIS", "COHR", "CLS"]
DEFENSIVE = ["PG", "KO", "PEP", "CL", "KMB", "MO", "PM", "JNJ", "MRK", "WMT", "COST", "DUK",
             "SO", "AEP", "ED", "GIS", "HSY", "MDLZ"]
SHOW = ["NVDA", "VRT", "CEG", "AAPL", "TSLA", "META", "KO", "DUK"]


def blend(text: np.ndarray, beh: dict[str, np.ndarray], tickers: list[str], w: float) -> np.ndarray:
    out = []
    for t, tv in zip(tickers, text):
        bv = beh.get(t)
        vec = np.concatenate([(1 - w) * tv, w * bv]) if bv is not None else np.concatenate([tv, np.zeros(20)])
        out.append(vec / (np.linalg.norm(vec) or 1.0))
    return np.asarray(out, dtype=np.float32)


def groups_for(matrix: np.ndarray, seed: int) -> np.ndarray:
    red = load_umap().UMAP(n_components=C.UMAP_N_COMPONENTS, n_neighbors=C.UMAP_N_NEIGHBORS,
                           min_dist=C.UMAP_MIN_DIST, metric=C.UMAP_METRIC, random_state=seed,
                           low_memory=False).fit_transform(matrix)
    labels = load_hdbscan().HDBSCAN(min_cluster_size=5, min_samples=C.HDBSCAN_MIN_SAMPLES,
                                    metric=C.HDBSCAN_METRIC, cluster_selection_method="eom").fit(red).labels_
    ids = sorted(set(labels) - {-1})
    norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    cents = np.stack([norm[labels == i].mean(axis=0) for i in ids])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-10
    return np.array(ids)[np.argmax(norm @ cents.T, axis=1)], float((labels == -1).mean()), len(ids)


def basket_rate(primary: pd.Series, basket: list[str]) -> float:
    names = [t for t in basket if t in primary.index]
    pairs = list(combinations(names, 2))
    return float(np.mean([primary[a] == primary[b] for a, b in pairs])) if pairs else float("nan")


def cross_sector_share(primary: pd.Series, sector: pd.Series) -> float:
    df = pd.DataFrame({"g": primary, "s": sector.reindex(primary.index)}).dropna()
    sizes = df.groupby("g").size()
    big = sizes[sizes >= 5].index
    top_share = df[df.g.isin(big)].groupby("g")["s"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    crossing = top_share[top_share < 0.6].index
    return float(df.g.isin(crossing).sum() / df.g.isin(big).sum())


def main() -> None:
    universe, _docs, prof, _ = rx.load_inputs()
    tickers = pd.read_csv(HERE / "b2_tickers.csv").iloc[:, 0].astype(str).tolist()
    text = np.load(HERE / "b2_embeddings.npy")
    beh_now = _build_comovement_vectors(tickers, as_of=AS_OF)
    beh_next = _build_comovement_vectors(tickers, as_of=AS_OF_NEXT)
    caps = pd.to_numeric(prof["marketCap"], errors="coerce")
    sector = prof["sector"].reindex(tickers)
    v4 = lc.load_v4_map()
    rets = rx.load_returns(tickers)
    window = rets.loc[rets.index[rets.index > AS_OF][: rx.WINDOW]]
    rng = np.random.default_rng(42)
    rows, mates = [], []
    for w in WEIGHTS:
        mat = blend(text, beh_now, tickers, w)
        runs = {s: groups_for(mat, s) for s in SEEDS}
        base = pd.Series(runs[42][0], index=tickers)
        nxt = groups_for(blend(text, beh_next, tickers, w), 42)[0]
        row = {
            "comovement_weight": w, "n_groups": runs[42][2], "noise_share": round(runs[42][1], 3),
            "seed_ari": round(float(np.mean([adjusted_rand_score(runs[42][0], runs[s][0]) for s in SEEDS[1:]])), 3),
            "monthly_refit_ari": round(float(adjusted_rand_score(runs[42][0], nxt)), 3),
            "v4_agreement": round(rx.v4_score(base, v4), 3),
            "cohesion_pair_mean": round(rx.cohesion(base, window, rng)["pair_mean"], 4),
            "ai_infra_comembership": round(basket_rate(base, AI_INFRA), 3),
            "defensive_comembership": round(basket_rate(base, DEFENSIVE), 3),
            "cross_sector_share": round(cross_sector_share(base, sector), 3),
        }
        rows.append(row)
        print(row, flush=True)
        for t in SHOW:
            co = base.index[(base == base[t]) & (base.index != t)]
            top = caps.reindex(co).sort_values(ascending=False).head(7).index.tolist()
            mates.append({"w": w, "ticker": t, "size": len(co) + 1, "largest_mates": " ".join(top)})
    pd.DataFrame(rows).to_csv(HERE / "comovement_sweep.csv", index=False)
    pd.DataFrame(mates).to_csv(HERE / "comovement_mates.csv", index=False)


if __name__ == "__main__":
    main()
