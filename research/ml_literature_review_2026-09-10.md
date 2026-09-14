# ML / AI trading literature review — Sep 2025 → Sep 2026, mapped to CynolycusBot

Date: 2026-09-10. Scope: papers from the past ~12 months that could improve **labels, features,
model training, or systems engineering** in this repo. Read via abstracts/HTML full text; numbers
below are **as reported by the authors, not reproduced here**. Most results are on Chinese A-shares
or monthly US data, so every item is a hypothesis to test on our own OOF matrices, not a result.

Repo facts this review was mapped against (checked 2026-09-10):
- Every `colab_competition.py` trainer ranks with `rank:ndcg` (XGB) / `lambdarank` (LGBM), plus
  `reg:squarederror` and classifier arms. Classifiers beat rankers in momentum and HTF.
- Momentum `top_n=3` (`strategies/momentum_expansion/config/momentum_config.py:220`); Meta `top_k=10`
  (`signals/meta_context/meta_ranker/live_runner.py:97`). The ordering edge is real but concentrated
  at shallow depth (`research/execution_quality/23_rank_depth_and_options.md`).
- No ranker uses uncertainty/abstention gating. `gate_regime_backtest.py` exists; the breadth gate was
  retracted (forward label).
- Four retractions in 2026 (stale option prints, the `trend_persistence` forward label, +$5,110 thesis
  cell, WOLF unadjusted corporate action) all came from **pipeline artifacts, not models**.

---

## Tier 1 — cheap, directly applicable to the Meta retrain / ranking roadmap

### 1. Label horizon sweep — "The Label Horizon Paradox" (arXiv 2602.03395, v5 Aug 2026)
- **Claim:** the best *training* label is often a shorter horizon δ\* than the *evaluation* horizon Δ.
  Signal saturates early while noise keeps accumulating. Interday: δ\* ≪ Δ. Intraday 30-min: δ\* ≈ Δ.
  Intraday 90-min: hump-shaped, with an intermediate optimum.
- **Evidence:** 10 neural architectures on CSI 300/500/1000 plus S&P 500. Train 2019–23, val 2023–24,
  test 2024–25. The LSTM on CSI 500 went from IC 0.845 to 1.029. Improvement held for every architecture.
- **Fit:** our labels are forward MFE/return at the traded horizon (momentum/HTF/Meta `horizon_bars`,
  e.g. 53 × 4h ≈ 21d). We have never trained at one horizon and scored at another.
- **Experiment (no new infra):** in the Meta retrain bundle, add label columns at δ ∈ {~2, 5, 10, 15}d.
  Train one model per δ and score all of them on the **fixed** target Δ, using the rank-depth
  harness's precision@3 plus the permutation test. Pick δ\* on validation only. Their "cheap
  approximation" is exactly this grid. The bi-level method isn't needed.
- **Caveats:** tested on neural nets, not GBDT. One test year. For SPY daytrader, the intraday
  results suggest little gain at short targets.

### 2. LambdaRankIC objective (arXiv 2605.00501, May 2026)
- **Claim:** a lambda gradient that weights each pairwise swap by its exact ΔRankIC,
  `12·|r̂ⱼ−r̂ᵢ|·|ỹᵢ−ỹⱼ| / n(n²−1)`, is implemented as an **XGBoost custom objective**.
- **Evidence:** US monthly data, 94 characteristics, 1964–2024, rolling 120/60/12. Rank IC was 0.115
  vs 0.086 for NDCG. Decile long-short Sharpe was 0.92 vs 0.50. No code release, but Algorithm 1 is
  short.
- **Fit:** a drop-in third ranking arm for every `colab_competition.py`.
- **Important nuance for us:** NDCG is top-heavy and LambdaRankIC is whole-list. Their win is on
  *decile long-short*; our edge is at *top-3 long-only*, where NDCG's top weighting may be the right
  alignment. Score it on precision@k/top-3 forward return, not on Rank IC, or it will win on the wrong
  metric. Given classifiers already beat rankers here, treat it as a challenger, not a fix.

### 3. Regime-trust gate + tail cap — "When Alpha Breaks" (arXiv 2603.13252, Feb 2026)
- **Setup:** a LightGBM ranker at a 20-day horizon, the closest published match to our 4H modules.
  A second model predicts **the ranker's own rank displacement** (DEUP epistemic uncertainty) against
  a point-in-time baseline.
- **Policy:** trade only when gate G(t) ≥ 0.2, use vol sizing on active dates, and cap the most
  uncertain tail. Gate AUROC was about 0.72 overall and 0.75 in the final test period.
- **Key negative result:** inverse-uncertainty sizing **degraded** performance, because uncertainty
  correlates ~0.6 with |score|. Use it as a binary gate plus a tail cap, never as a continuous sizer.
- **Fit / difference from the retracted breadth gate:** the target is "will *this ranker's* ordering
  hold on this bar", i.e. per-bar realized rank IC / top-3 hit, computed only from data available
  at the bar. It is not market breadth. We can test it on the 1,739 OOF bars (2022-11..2026-05) in
  the Meta research matrix, using `mom_score`.
- **Caveat:** we already found entry-day breadth unforecastable with current features. Expect low
  power, and pre-register the gate threshold before looking.

### 4. Pipeline falsification audit — "Spurious Predictability in Financial ML" (arXiv 2604.15531, Apr 2026)
- **Method:** run the *entire* pipeline (features, labels, tuning, selection, execution timing)
  on five synthetic nulls: white noise, regime-switching vol, bid-ask bounce placebo, factor null,
  and GARCH. Any significant walk-forward result on a null falsifies the pipeline. Then measure
  selection inflation (ΔZ, effective multiplicity K_eff).
- **Second finding:** ML cross-sectional models often just reproduce known factor exposures.
  Returns are significant gross, but alpha is zero after factor adjustment.
- **Fit:** every 2026 retraction was a pipeline artifact. A null-environment harness would have
  flagged the `trend_persistence` forward label mechanically. The factor-null test is the right check
  for "is momentum's ordering edge just the momentum/size/vol factor?".
- **Experiment:** wrap the existing rank-depth harness (`scripts/` rank-depth +
  permutation) with (a) a shuffled/synthetic-returns run and (b) style-residualized forward returns
  (momentum, size, beta, vol, liquidity) before computing top-k lift. This is research
  infrastructure, and it protects every later item on this list.

---

## Tier 2 — features

### 5. Option-implied primitives — "Option-Implied Signals and Crash Risk, 2015–2026" (arXiv 2608.26115)
- 12.4M firm-days from the OPRA EOD feed. Signals that **survive 2023–2026:** Cremers–Weinbaum IV
  spread (ATM call − put IV, |t|>4 in every regime), risk-neutral skewness (t>13 at 63d), and the
  30–60d term-structure slope. **Decayed:** the smirk is insignificant in the AI/mega-cap regime.
  **Weak everywhere:** put/call ratios.
- XGBoost beat linear only in 2023–26 (R²_OOS +1.29% vs +0.07%), and 2025 drives it. It
  underperformed linear in 2020–22.
- **Fit:** dealer_ranker shows no ranking lift from GEX-style features. The durable signals are
  IV-surface shape, not positioning. These can be computed from the Schwab chain snapshots we already
  pull for dealer gamma.
- **Blocker:** there is no historical IV surface (OPRA agreement unsigned; Schwab is live-only). This
  is **forward capture only**, so start snapshotting daily IV spread/skew/term slope now, because the
  clock is the constraint.
- **Implemented 2026-09-10** as nightly stage 1c: `strategies/dealer_positioning/iv_surface.py` +
  `scripts/capture_iv_surface.py` → `Data/dealer_positioning/iv_surface/YYYYMMDD/`. It also archives
  the raw per-contract bid/ask, the first option-quote history we have. Schwab's `volatility`
  field is one value per strike (call == put in 99.94% of pairs), so the CW spread is backed out of the
  mids instead. The live 20-name check matched CBOE iv30 at corr 0.999.

### 6. Point-in-time language models + a lookahead test for LLM-derived features
- **PIT LMs** (Kelly et al., arXiv 2607.11889, Jul 2026): 1.5B/4B decoders with **monthly
  checkpoints Dec-2013..Dec-2024** on HuggingFace. News embeddings from them produce out-of-sample
  Sharpe 1.0–1.5 with no training leakage. This is a leak-free alternative or complement to FinBERT
  in the live news path (per_record + catalyst_signal + embeddings).
- **Lookahead Propensity** (arXiv 2512.23847, Dec 2025): a date-only recall probe that estimates
  whether an LLM "knows" a firm-date outcome. Contamination shows as predictive power concentrated
  in high-LAP firm-dates, and it vanishes after the model's cutoff.
- **Risk this exposes in our repo:** dynamic theme assignments use Claude plus embeddings. Any
  *historical* theme membership assigned by a 2026 model may encode knowledge of which names later
  became theme winners. That is lookahead and survivorship inside theme research. Audit it
  before using theme features in any backtest before 2026. **Look-Ahead-Bench** (arXiv 2601.13770)
  is a standardized version of this check.

### 7. Cross-stock / theme-graph spillover features
- **LLM-Augmented Semantic Networks** (arXiv 2604.19476): 10-K embeddings produce a candidate graph,
  and an LLM filters the edges by economic relation. S&P 500 long-short Sharpe went 0.74 → 0.82
  (2011–19).
- **Supply-chain propagation of text signals** (arXiv 2606.29290): network-augmented embedding factor,
  NW t = −2.64 after momentum/vol/size controls.
- **Fit:** we already have an 85–88-theme graph. Lagged peer-return and peer-news features
  (theme-mates' returns and catalyst scores over the last 1–5 bars) are cheap and PIT-safe, *if*
  theme membership itself is PIT (see #6). The gains reported are modest.

### 8. LLM-proposed features into GBDT — "Generative AI for Stock Selection" (arXiv 2602.00196, Jan 2026)
- The LLM proposes economically motivated features (with RAG), and a gradient-boosted tabular model
  consumes them. Reported Sharpe 1.14–1.63 in ensembles. AlphaAgent (KDD 2025) adds
  originality/complexity regularization against alpha decay.
- **Fit:** a feature-idea generator for the "find features that rank forward moves" roadmap step.
  **Must** be paired with #4, because LLM search raises effective multiplicity, which is exactly
  what that paper quantifies.

---

## Tier 3 — sizing, model class, engineering (lower priority or optional)

- **Conformal Kelly** (arXiv 2608.01494, Aug 2026): the conformal interval scales a fractional-Kelly
  size. Honest negative result: in a sealed 2022–24 lockbox, coverage held (74.5% vs 75%) but growth
  **failed** to beat unlevered passive. Takeaway: conformal intervals are reliable for risk bands and
  tail caps, not for return sizing. Consistent with #3.
- **FinPFN / TabPFN-3** (FinPFN Nov 2025; TabPFN-3 arXiv 2605.13986): a tabular foundation model
  fine-tuned for regime-aware returns. CSI 500 IR 0.85 vs LightGBM 0.70, with the largest gap during
  regime changes. It is plausibly suited to our small effective samples (~1.7k OOF bars).
  **Optional / Outside Roadmap.** It likely needs a GPU for fine-tuning, and context-size limits
  apply to the cross-section.
- **Decision-induced ranking** (arXiv 2605.01176): score-driven top-k selection inflates predictions
  and turnover. Fixes are clipping, rescaling, and partial rebalancing. Relevant once top-k is
  concentrated to 1–3: add a hysteresis/hold-band so a name at rank 4 doesn't churn out.
- **Decision-focused sparse tangent portfolios** (arXiv 2607.00581): a differentiable top-k
  operator trains prediction end-to-end on Sharpe. Heavier lift. **Optional / Outside Roadmap.**

---

## Evidence against (do not pursue now)

| Direction | Evidence | Why skip |
|---|---|---|
| General time-series foundation models (TimesFM, Chronos, Moirai) | Zero-shot TimesFM R² −2.8%, Chronos −1.4% (Rahimikia et al., arXiv 2511.18578); gains over random walk "small and sparse", 2/10 significant (arXiv 2606.27100) | Only models **pre-trained from scratch on financial data** help. Kronos (arXiv 2508.02739, AAAI 2026) is that, but its RankIC claims are on its own benchmark |
| LLM trading agents | Of 19 closed-loop studies: 2 have time-consistent splits, 1 models costs, 0 fully reproducible (arXiv 2605.19337) | No credible after-cost evidence |
| Deep models on raw intraday bars | STRATA (arXiv 2608.28060) reports Sharpe 12.85 close-to-close on A-shares and admits it degrades from the first executable price | This is the same failure as our 22 → 23 rank-depth retraction |
| Replacing engineered features with deep nets | Feature engineering + GBDT beat DL (arXiv 2601.07131) | Supports staying GBDT |
| Importing published stop/TP grids | Agent-swarm replay favoured *tighter* stops (arXiv 2604.27150) | Contradicts our measured result (winners' pre-peak MAE, wide stops beat tight). Our own fills rule |

---

## Suggested order (roadmap-aligned)

1. **#4 falsification + factor residualization** on the rank-depth harness. It guards everything else.
2. **#1 label-horizon sweep**, folded into the pending Meta retrain bundle (and momentum).
3. **#2 LambdaRankIC** as a third arm in `colab_competition.py`, scored on top-3 metrics.
4. **#3 regime-trust gate** on the 1,739 OOF bars, as a binary gate only.
5. **#5 start forward IV-surface capture** now (data-bound; cheap to start, slow to mature).
6. **#6 LAP/PIT audit** of theme membership and news features before any pre-2026 theme backtest.

## Sources
- Label Horizon Paradox — https://arxiv.org/abs/2602.03395
- LambdaRankIC — https://arxiv.org/html/2605.00501v1
- When Alpha Breaks — https://arxiv.org/abs/2603.13252
- Spurious Predictability in Financial ML — https://arxiv.org/html/2604.15531v1
- Option-Implied Signals and Crash Risk — https://arxiv.org/html/2608.26115
- Scaling Point-in-Time Language Models — https://arxiv.org/html/2607.11889v2
- Detecting Lookahead Bias in LLM Forecasts — https://arxiv.org/abs/2512.23847
- Look-Ahead-Bench — https://arxiv.org/abs/2601.13770
- Cross-Stock Predictability via LLM-Augmented Semantic Networks — https://arxiv.org/abs/2604.19476
- Supply Chain Propagation of Textual Signals — https://arxiv.org/pdf/2606.29290
- Generative AI for Stock Selection — https://arxiv.org/html/2602.00196v1
- AlphaAgent — https://arxiv.org/html/2502.16789v2
- Conformal Kelly — https://arxiv.org/html/2608.01494v1
- TabPFN-3 — https://arxiv.org/pdf/2605.13986 ; FinPFN — https://www.sciencedirect.com/science/article/abs/pii/S1386418125000825
- Decision-Induced Ranking — https://arxiv.org/pdf/2605.01176
- Decision-focused Sparse Tangent Portfolio — https://arxiv.org/abs/2607.00581
- Re(Visiting) TSFMs in Finance — https://arxiv.org/abs/2511.18578
- Pretrained TSFMs for Financial Return Forecasting — https://arxiv.org/abs/2606.27100
- Kronos — https://arxiv.org/abs/2508.02739
- Agentic Trading evidence map — https://arxiv.org/html/2605.19337v1
- STRATA — https://arxiv.org/abs/2608.28060
- Limits of Complexity (feature engineering vs DL) — https://arxiv.org/pdf/2601.07131
- Optimal SL/TP for agent swarm — https://arxiv.org/abs/2604.27150
