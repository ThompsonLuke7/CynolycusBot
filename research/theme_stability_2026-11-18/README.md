# Theme grouping stability — 2026-11-18

**Question.** Dynamic themes churn weekly: 19 of the top 20 by market cap changed primary
theme at least once over the five weekly runs 09-21 → 11-03, and the rename-proof
agreement between consecutive weeks' groupings (adjusted Rand index, ARI) was only
0.24–0.48. Would clustering on stable business inputs fix that without losing quality?

**Setup.** No API calls, CPU only. Universe = the 2,903 tickers of the 2026-11-16 run
(the universe was identical in every weekly snapshot, so all observed churn came from
inputs and algorithm, not membership changes). Three embeddings, each clustered exactly
as step03 (UMAP 20-d / 15 neighbours / min_dist 0 / cosine; HDBSCAN leaf, min_cluster_size
5, min_samples 2), each ticker assigned to its nearest centroid as step08 does:

| variant | identity vector |
|---|---|
| PROD | production 11-16 embedding: 0.6·(0.6 desc + 0.4 news) ⊕ 0.4·60d co-movement PCA |
| B1 | BGE of the business description |
| B2 | BGE of `Sector: S. Industry: I. ` + description |

Code: `run_experiment.py` (main), `granularity_check.py`, `seed_stability.py` (HDBSCAN settings),
`comovement_sweep.py` (co-movement weight). Raw output:
`results.json`, `stability_runs.csv`, `cohesion.csv`, `megacap_mates.csv`, `granularity.csv`,
`seed_stability.csv`, `comovement_sweep.csv`, `comovement_mates.csv`.

## 1. The clustering itself is unstable, whatever the inputs

Inputs held fixed; only the UMAP seed changes, or a random 3% of tickers is dropped
(what promotions/delistings do). ARI vs the base run, and top-20 "flips" (group-mate
Jaccard < 0.5):

| variant | seed change ARI | 3% drop ARI | top-20 flips (seed / drop) |
|---|---|---|---|
| PROD | 0.57 | 0.55 | 14.5 / 13.4 |
| B1 | 0.56 | 0.52 | 11.0 / 12.2 |
| B2 | 0.55 | 0.55 | 11.5 / 10.6 |
| *published, real consecutive weeks* | *0.24–0.48* | | *15–19* |

With identical inputs, half the grouping changes. Stable inputs alone cannot stop the
churn if the taxonomy is re-clustered every week; the observed weekly churn is this
algorithmic instability compounded by the ~64% of the vector (news, co-movement) that
changes weekly.

## 2. Business inputs (B2) group better

| | v4 agreement | fwd 20d cohesion (pair / group mean) | noise | median top1–top2 margin |
|---|---|---|---|---|
| PROD (11-16) | 0.43 | — | 34% | 0.029 |
| published (09-21…11-03) | 0.35–0.42 | 0.425 / 0.441 | — | — |
| B1 | 0.52 | 0.416 / 0.417 | 32% | 0.041 |
| **B2** | **0.62** | **0.431 / 0.447** | 31% | 0.038 |
| Yahoo sector (11 groups) | — | 0.290 | — | — |
| random, B2 sizes | — | 0.142 | — | — |

Cohesion = mean within-group correlation of daily returns over the 20 trading days after
09-21, 09-28, 10-05 and 10-12 (|ret| > 40% masked; ≥ 15 valid days). B2 uses no price data
yet matches the published groupings, which include trailing 60-day co-movement. Caveats:
B2 uses today's descriptions (business text rarely changes; no return leakage), and v4
agreement favours finer groups mechanically.

Mega-cap group-mates (largest 6), published 11-03 vs B2:

| | published 11-03 | B2 |
|---|---|---|
| AAPL | DE CTVA EA ADM NTR BG | GRMN EA TTWO PL SRAD GRVY |
| TSLA | CCJ MP OKLO NXE UEC PLUG | TM GM RACE F HMC RIVN |
| META | GOOGL GOOG MSFT PLTR PANW CRWD | GOOGL GOOG SPOT PINS SNAP TTD |
| JPM | WFC PNC USB TFC FITB HBAN | DB TFC MTB CFG RF WBS |
| NVDA | AMD ORCL INTC ARM DELL QCOM | AMD INTC ARM DELL CDNS SNPS |
| AMZN | BABA PDD NTES SE BIDU JD | BABA SONY PDD MELI DASH SE |

(PROD 11-16 put AAPL with HCA/CNC hospitals and JPM with insurers.)

## 3. Coarser clusters are NOT reliably more stable (corrected)

A first two-seed check suggested HDBSCAN `eom`/5 was far more stable (ARI 0.86 vs
0.55). It was a lucky draw: the identical inputs re-run later gave 0.58. With 6 seeds
and all 15 pairs (`seed_stability.py`, B2 + 0.2 co-movement):

| HDBSCAN | groups | noise | seed ARI mean (min–max) |
|---|---|---|---|
| leaf, 5 (kept) | 189 | 30% | 0.59 (0.54–0.62) |
| eom, 5 | 139 | 16% | 0.57 (0.42–0.88) |

`eom` halves the unassigned share but is erratic (some seeds merge a 180-name blob).
Whatever the setting, a re-fit reshuffles ~40% of the grouping, so groups must be fixed
between re-fits. `leaf`/5 is kept because it is predictable. (`granularity.csv` holds
the single-pair numbers for larger min_cluster_size; treat them as indicative only.)

## 4. Keep some price co-movement: cross-industry trades need it

Themes are meant to track cross-industry trades (AI infrastructure, defensives,
vertically integrated chains), which business text alone cannot see.
`comovement_sweep.py` blends B2 text with the 60-day co-movement PCA,
`normalize(concat[(1-w)·text, w·comovement])`, co-movement built as of 2026-10-05 so it
does not overlap the cohesion window:

| w | fwd cohesion | v4 | cross-sector share | re-fit ARI (10-05 → 11-03) |
|---|---|---|---|---|
| 0.0 | 0.424 | 0.61 | 2.5% | 1.00 |
| **0.2** | **0.473** | **0.58** | 4.0% | **0.84** |
| 0.3 | 0.486 | 0.55 | 12.5% | 0.58 |
| 0.4 (old production weight) | 0.478 | 0.55 | 14.7% | 0.55 |
| 0.5 | 0.458 | 0.46 | 24.6% | 0.48 |

What w = 0.2 adds over business text alone (largest group-mates, `comovement_mates.csv`):
NVDA moves from a 180-name software/semis blob to MSFT ORCL ANET CDNS SNOW NBIS CRWV
(AI compute); CEG splits from the regulated utilities into VST NRG TLN AES (AI power),
while DUK stays with NEE SO AEP (defensive utilities); VRT joins ETN NVT HUBB GNRC
POWL (power equipment); TSLA joins RIVN XPEV LI NIO MBLY (EV/autonomy) instead of
legacy autos. At w ≥ 0.3 AAPL lands with toys (HAS, MAT) or autos again.

## Conclusion (implemented 2026-11-18)

1. Identity vector = B2 text (sector + industry + business description, no news) with a
   **0.2** share of 60-day price co-movement (`DESC_BLEND_WEIGHT` 1.0, `COMOVEMENT_WEIGHT`
   0.2; was 0.6 / 0.4).
2. **Fixed groups**: step03 re-fits only every `THEME_REFIT_DAYS` (91), when the recipe
   changes, or on request (`WEEKLY_THEMES_REFIT=1`); otherwise it reuses
   `ticker_cluster_anchor.parquet` and new tickers wait unassigned.
3. **Frozen primary themes**: step09 keeps each ticker's primary theme by cluster id for the
   life of a re-fit (`ticker_primary_anchor.parquet`), so near-ties cannot flip it; a
   corrected label follows its cluster.
4. HDBSCAN stays `leaf`/5.

Rehearsal on the 2026-11-16 inputs (real step02/step03 code, scratch paths): 187 groups,
31% unassigned (incl. TSLA, AAPL, MSFT, AMZN, JPM, DUK — placed by industry peers, then
frozen); NVDA with PLTR ANET EQIX SNOW NBIS DOCN, CEG with VST NRG TLN OKLO SMR, VRT
with ETN NVT HUBB POWL; a simulated next week kept all 2,883 continuing tickers in their
group. Not yet tested: labels under the new groups (needs API credit) and real universe
turnover over a quarter.
