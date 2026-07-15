# Cross-Signal Confluence Discovery — Findings Report

**Date:** 2026-07-07 (study run 2026-07-06/07)
**Agent:** Claude
**Status:** Research only. Nothing wired into live/paper trading. Test set has been consumed — do not re-mine against it.

---

## Executive verdict

**No cross-signal confluence survives the pre-registered rigor gauntlet. The honest headline
is a null result, with a measured reason: the dataset is too short to certify thin
interactions.**

- 524 orthogonal-axis signal pairs were mined on TRAIN (8 months) across two binary
  targets (`meta_upside`, `meta_good`) and a continuous target (mean forward alpha).
  The p-value distribution of the monthly-block interaction test is consistent with a
  global null (2.4–4.9% of pairs at p≤0.05 vs 5% expected by chance). Best
  BH-FDR q-value: 0.56. Zero pairs met the pre-registered bar (q≤0.10 + ≥6 stable
  months + positive val replication).
- **Power check (planted synthetic interactions):** the gauntlet catches a pure +10pp
  precision interaction (q=0.074, zero false positives elsewhere) but misses +5pp
  (q=0.25) and +3pp (q=0.98). With ~11.5 months of labeled data, the certification
  floor is roughly a **+7–10pp joint-precision interaction**. Edges of the size this
  system actually trades (momentum OOS lift ~1.06) are far below that floor — so
  "not certifiable" ≠ "does not exist," but nothing here can honestly be called found.
- A relaxed, explicitly *uncertified* screen (raw p≤0.05 + ≥300 de-overlapped episodes
  + ≥75% positive months + val replication) produced 16 unique candidate pairs — a count
  fully consistent with chance (≈23–29 expected). Each got exactly one TEST read:
  **6 failed outright on test, most of the rest are redundant momentum stacks, and one
  pair replicated direction in every split on both targets** (see ranked list).
- Triples were not mined: with the pair-level certification floor already at +7–10pp and
  triple cells 3–10× smaller, any triple "finding" would be uncertifiable by construction.

---

## Data substrate and method

**Spine:** `signals/meta_context/meta_ranker/meta_ranker_matrix.parquet` (554,858 rows,
(timestamp, ticker), 2 bars/day, 2025-05-29 → 2026-07-02; forward labels present through
2026-05-14 because they come from the momentum OOF build of that date).
Joined point-in-time by `scripts/confluence_discovery/build_dataset.py`:

- Technical states from `features_4h.parquet` (2026-07-06 rebuild), exact
  (timestamp, ticker) join — 98.0% coverage.
- FINRA daily short-sale volume → `short_ratio`, `short_ratio_z20`, `short_ratio_pct63`,
  `short_vol_surge20`, joined **strictly prior-day** (`merge_asof`,
  `allow_exact_matches=False`) — 99.5% coverage.

**Universe (per house realism):** label-valid rows with `dollar_vol_pctile_252 ≥ 0.40`,
`low_price_flag == 0`, no ETFs → 295,328 rows, 1,070 tickers.

**Split (house convention, `family_backtest.compute_test_cutoff`):** row-fraction 60/20/20
over timestamp-sorted label-valid rows →
TRAIN < 2025-12-29, VAL < 2026-03-11, TEST ≤ 2026-05-14.
Quantile thresholds for all conditions were fit on TRAIN only. Mining + significance on
TRAIN; selection required VAL replication; TEST touched exactly once on a frozen shortlist.

**Interaction test:** per-pair monthly delta of joint precision vs **best single marginal**
(blocks absorb the overlapping 25-bar forward windows and cross-sectional correlation),
one-sided t-test across months, BH-FDR across all pairs per family. Super-additivity is
tracked separately as a log-odds interaction term
(`logit(pJ) − [logit(pA)+logit(pB)−logit(base)]`), because beating the best marginal can be
achieved by merely stacking two correlated positives — that is *not* confluence.
Sample size is reported both as rows and as de-overlapped per-ticker **episodes**.

**Outcomes:** `meta_upside` (fwd_max_return ≥ +15% in 25 4H bars, liquid),
`meta_good` (≥ +12%, max drawdown ≤ 8%, positive alpha, liquid), mean `fwd_max_alpha`.

---

## Signal inventory

**Used (point-in-time verified):**

| Axis | Signals | PIT basis |
|---|---|---|
| technical | `mom_score`/`mom_xs_rank` (ExpansionRanker, walk-forward OOF), `htf_score`/`htf_xs_rank` (HTF swing, OOF), near-52w-high, breakout20, compression, rvol, ATR expansion, volume spike | OOF predictions from walk-forward folds; features causal by construction |
| theme | heat, rank, acceleration, breadth, newness, crowding, membership | step09 per-date values, strictly-prior-day asof join |
| news/catalyst | `news_catalyst_score`, counts, breaking, bull alignment, forward-guidance score | dated records, strictly-prior-day asof join |
| calendar | days-to/since earnings, pre/post-3d flags, macro-event proximity | calendar known in advance (see caveats) |
| flow | FINRA daily short volume ratio/z/percentile/surge | daily files published after the close; strictly-prior-day join |
| regime | SPY trend, VIX z, treasury level/5d-change | causal market history |

**Excluded (and why):**

| Source | Reason |
|---|---|
| Dealer positioning / gamma (Schwab snapshots) | 2 days of history (2026-07-02+). Unusable for any split. Forward collection is the fix. |
| CBOE options summary / unusual strikes | Dense snapshots only since ~2026-05-31 (~1 month); a few stray earlier rows. Too short. |
| NASDAQ/NYSE bi-monthly short interest | Fetcher is yfinance snapshot-only; **no stored point-in-time history**; reconstructing from current values would be lookahead. |
| USAspending contracts | No dated, ticker-mapped historical store; award events flow into news records (already represented by the news axis). |
| Social attention (Reddit) | Module code exists, no data collected. |
| `trend_persistence` | **Forward label** (known repo lesson) — banned as feature. |
| `earnings_in_fwd_window` | Defined over the label window — banned as feature. |
| `signal_agreement` (matrix col) | Degenerate coverage (~0.1% non-null in usable form). |
| `theme_newness_score`, `spy_uptrend` as conditions | Degenerate on train (q80 = min value / 96% coverage) — auto-dropped. |

**Leakage audit notes:**
- `theme_crowding_frac` tripped the auto-tripwire (corr 0.29 with fwd alpha). Audit verdict:
  **causal, not leakage** — it is the share of the ticker's theme with same-bar
  `mom_xs_rank > 0.8` ([build_meta_ranker_matrix.py:514-516]). But it is momentum-derived:
  pairs with mom/htf are same-family stacks, and their consistently *negative* interaction
  terms (−0.6 to −1.0) confirm redundancy, not synergy.
- Known second-order caveats (accepted, flagged): theme taxonomy (v4) is applied
  retroactively over history; the news scorer version postdates some of the news it scores;
  `days_to_earnings` uses the fetched calendar (confirmed dates can shift by a day or two).
  None of these can manufacture a pair interaction by themselves, but they add optimism to
  theme/news marginals.

---

## Ranked shortlist (all UNCERTIFIED — best-of-noise until re-tested on new data)

Test-set context first: TEST (2026-03-11 → 2026-05-14) was a violent recovery tape —
`meta_upside` base rate 23.9% vs 16.5% train / 14.9% val. Absolute PnL numbers on test are
regime-flattered; only deltas vs marginals are meaningful.

**1. `theme_rank_top10 & short_vol_surge`** — *the only candidate worth anything*
Rule: ticker's primary theme in top-10 by heat AND prior-day FINRA short volume ≥ 1.69×
its 20-day mean (train q90). Rationale: heavy shorting *into* a leading theme = squeeze
fuel / disagreement, orthogonal origins (price-derived theme heat × flow).
- delta vs best marginal: **+3.7pp train, +5.5pp val, +9.7pp test**; positive in 8/8 train
  months and 3/3 test months; log-odds interaction positive in every split
  (+0.09 / +0.18 test); replicates on both targets (meta_good test delta +4.5pp).
- Sample: 3,078 train rows (1,194 episodes), 980 test rows (365 episodes, 255 tickers,
  52 themes, max theme share 9.7% — not concentration-driven).
- Realism: ~21 candidate rows/day on test; joint-cell mean fwd close return 13.7% gross /
  13.4% net of 30bps (tape-inflated), win rate 70%, mean max-drawdown 8.6%.
- Honesty: raw p 0.029 (binary) / 0.0034 (alpha family) but **FDR q ≈ 0.86** — as a member
  of a 524-pair search this is indistinguishable from the best of noise. Its distinguishing
  evidence is directional consistency in *every* split, month, and target.
- **Verdict: worth a zero-cost shadow log / scheduled re-test once ≥6 more labeled months
  accumulate. Not worth paper capital yet, and far from live.**

**2. `theme_cold & news_bull`** (meta_good) — bullish news alignment in bottom-quintile-heat
themes (contrarian rotation story). Delta +5.4pp train / +2.9pp val / +2.2pp test,
interaction positive in all splits (+0.05→+0.12), n large (4,877 test rows, 1,461 episodes).
Effect shrinks monotonically across splits. **Verdict: marginal — directionally alive,
decaying magnitude; re-check later, do not act.**

**3. `htf_top20 & news_breaking` / `mom_htf_agree & news_breaking`** — breaking-news flag on
technical leaders. Deltas positive in all three splits (+4.2/+2.4/+5.4pp and
+3.8/+2.4/+4.0pp) but test cells are small (390 and 316 rows; ~150–200 episodes) and the
news-coverage regime changed mid-sample (live collection ramped). **Verdict: marginal /
underpowered — cannot be separated from the coverage shift.**

**4. Momentum × `theme_crowded` stacks** (`mom_top20/mom_top10/htf_top20/mom_htf_agree`
× theme_crowded) — joint precision is high (test lifts 2.3–2.5) and deltas positive, but the
log-odds interaction is **strongly negative everywhere** (−0.56 to −0.95): these are two
measurements of the same momentum phenomenon. Stacking them concentrates picks; it does not
create conditional edge. The meta ranker already consumes these inputs. **Verdict: additive
stacking, not confluence. No action.**

**5. `atr_expanding & short_low`, `theme_rank_top10 & short_low`** — small positive test
deltas on one target, sign flips on the other. **Verdict: noise-leaning.**

---

## What did NOT survive (the graveyard — equally important)

| Pair | Seduction | Death |
|---|---|---|
| `mom_htf_agree & short_pct_high` | Best raw p of the whole search (0.0014), positive delta in 8/8 train months, val +0.8pp | **Test −1.0pp** — the single sharpest train statistic in the study failed its one OOS read; textbook mined noise |
| `mom_top10 & short_pct_low` | Train delta +5.4pp, lift 3.4 | Val −3.6pp (never reached test) |
| `theme_crowded & news_breaking` | Train +5.4pp AND val +5.4pp | **Test −3.2pp** |
| `mom_top10 & earnings_far` | Train +3.3pp, val +1.3pp, huge n | Test −1.3pp (mom alone did better) |
| `theme_accel & pre_earnings3d` | Train +3.7pp, val +2.1pp | **Test −5.7pp** |
| `theme_rank_top10 & pre_earnings3d` | Val +8.8pp | Test −1.8pp |
| `mom_top20 & news_none` | Train +3.0pp, val +7.0pp | **n=4 on test** — "no news" measures collection coverage, not market state; structurally broken condition |
| All regime conditioners (vix_stress, spy_downtrend, rates_*) | Train lifts up to 1.6 | One-episode regimes: val/test sign flips; 8 months cannot distinguish a regime effect from a single drawdown event |
| RETRACTED precedent honored | `trend_persistence` breadth gate | Never entered: forward label, banned at inventory |

Global honesty check: 466 pairs carried enough months for the block test. Raw p≤0.05
counts — 23 (meta_upside precision), 11 (meta_good precision), 19 (mean-alpha family) —
never exceeded the ≈23 expected by chance. **The search as a whole is consistent with "no
detectable pair interactions in this dataset."**

---

## Reproducibility

- `scripts/confluence_discovery/build_dataset.py` — rebuilds the PIT dataset
  (`research/confluence/confluence_dataset.parquet`).
- `scripts/confluence_discovery/search.py mine --target meta_upside [--triples]` — fits
  thresholds on train, mines pairs, writes `singles_*/pairs_*/shortlist_*.csv` +
  `fitted_thresholds_*.json`. `confirm --shortlist <csv>` runs the one-shot test read.
- `scripts/confluence_discovery/power_check.py` — planted-interaction power calibration.
- All intermediate outputs: `research/confluence/`. Run log: `research/confluence/RUNLOG.md`.

## Next step (roadmap-aligned)

Re-run `search.py mine` when ≥6 additional labeled months exist (early 2027), which roughly
halves the certification floor. Keep collecting dealer positioning + CBOE snapshots — those
are the genuinely orthogonal axes this study could not test, and 6–9 months of either would
make a materially better-powered second pass. Until then: no confluence rule should be wired
anywhere, including as a "soft" filter.
