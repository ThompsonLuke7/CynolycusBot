"""Follow-up: is coarser HDBSCAN (bigger / eom clusters) inherently more stable?

Same B2 inputs ("Sector: S. Industry: I. " + description) and UMAP as run_experiment.py.
UMAP depends only on the seed, so it runs once per seed (42, 7, 123) and each HDBSCAN
setting is fit on those three layouts. Stability = mean ARI of the nearest-centroid
assignment between seed 42 and seeds 7 / 123. Quality = v4 agreement and forward
20-day cohesion after 2026-10-05. Writes granularity.csv next to this file.
"""
from __future__ import annotations

import sys
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

SETTINGS = [("leaf", 5), ("leaf", 10), ("leaf", 15), ("eom", 5), ("eom", 10), ("eom", 15)]
SEEDS = [42, 7, 123]


def assign(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    ids = sorted(set(labels) - {-1})
    norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    cents = np.stack([norm[labels == i].mean(axis=0) for i in ids])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-10
    return np.array(ids)[np.argmax(norm @ cents.T, axis=1)]


def main() -> None:
    universe, docs, prof, _ = rx.load_inputs()
    _, pref = rx.texts_for(universe, docs, prof)
    tickers, matrix = rx.embed(pref)
    np.save(HERE / "b2_embeddings.npy", matrix)
    pd.Series(tickers).to_csv(HERE / "b2_tickers.csv", index=False)
    umap, hdb = load_umap(), load_hdbscan()
    layouts = {s: umap.UMAP(n_components=C.UMAP_N_COMPONENTS, n_neighbors=C.UMAP_N_NEIGHBORS,
                            min_dist=C.UMAP_MIN_DIST, metric=C.UMAP_METRIC, random_state=s,
                            low_memory=False).fit_transform(matrix) for s in SEEDS}
    v4 = lc.load_v4_map()
    rets = rx.load_returns(tickers)
    after = rets.index[rets.index > pd.Timestamp("2026-10-05")][: rx.WINDOW]
    window = rets.loc[after]
    rng = np.random.default_rng(42)
    rows = []
    for method, mcs in SETTINGS:
        prim = {}
        for s, red in layouts.items():
            labels = hdb.HDBSCAN(min_cluster_size=mcs, min_samples=C.HDBSCAN_MIN_SAMPLES,
                                 metric=C.HDBSCAN_METRIC, cluster_selection_method=method).fit(red).labels_
            prim[s] = (assign(matrix, labels), float((labels == -1).mean()), len(set(labels) - {-1}))
        base = pd.Series(prim[42][0], index=tickers)
        coh = rx.cohesion(base, window, rng)
        rows.append({
            "method": method, "min_cluster_size": mcs, "n_groups": prim[42][2],
            "noise_share": round(prim[42][1], 3),
            "seed_ari": round(float(np.mean([adjusted_rand_score(prim[42][0], prim[s][0]) for s in SEEDS[1:]])), 3),
            "v4_agreement": round(rx.v4_score(base, v4), 3),
            "cohesion_pair_mean": round(coh["pair_mean"], 4),
        })
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(HERE / "granularity.csv", index=False)


if __name__ == "__main__":
    main()
