# Year-two experiment checklist: where does an obvious setup stop being obvious?

Written 2026-11-04. Source: an external chat analysis, pasted by the user, that
synthesised several papers (Financial World Modeling, HARN, AURA, a Hawkes
market-making paper, SciExam, GPN-Star) and applied them to CynolycusBot. Every
claim in it was checked against this repo. The paper findings are as that chat
summarised them; they were **not independently verified** in this pass (see the
last section).

Status legend: `[ ]` open, `[~]` partly done in the repo already, `[x]` done.
Keep this file current: tick items, link the write-up, and add the date.

---

## 1. Verdict on the pasted analysis

**The core question is right.** It asks "at which stage does edge die:
representation, prediction, selection/abstention, instrument choice, or
execution/exit?" That is better than asking whether the model or the system is
to blame. Five parts hold up against the repo: the decision-state archive,
abstention as a modelled action, splitting alpha from option expression,
freezing the evaluator, and the "not now" list.

**Claims checked and confirmed**

- `fwd_max_alpha` is a no-op: within-bar rho vs raw forward max is +1.000
  (`research/execution_quality/10_label_diagnosis.md`). The live label still uses it
  (`strategies/momentum_expansion/labels/expansion_labels.py:108`).
- Entry vectors were never archived, so historical feature parity cannot be certified
  (`research/profit_investigation_2026-09-19/report.md:122,193`).
- The option-selection score is not a probability of profit
  (`core/nervous_system/config/options.py:3-4`).
- A triple-barrier comparison already exists
  (`research/regime_coverage_2026-09-21/label_compare.py`).

**Already done or superseded (do not rebuild)**

| Paste item | What already exists |
|---|---|
| "Baseline B: fix `fwd_max_alpha`" | Superseded by the Stage 2 L2 arms (vol-matched realised excess). The bundle was built 2026-10-06 and **has never been run**. |
| Top 50%..1% selectivity curve | Rank-depth study, `top_n=3` (`research/execution_quality/23_*`, `25_*` §B). It used MFE and returns, not net-of-cost expectancy. |
| ENTER/PASS abstention | RL Stage 2: the incumbent is decidable, but nothing fitted survives its permuted null (`research/rl_reintroduction_2026-09-23/02_decidability_findings.md`). |
| "Not discovered" stage | RL Stage 1: funnel tail lift is 1.06x at the gate and 1.18x at the universe, a measured absence (`01_tail_retrieval_findings.md`). |
| `P(teacher takes setup)` | Positive-unlabeled ACE timing experiment (`research/discord_ledger_2026-09-21/ace_pu_timing_experiment_*`). |
| Freshness and causal state | Snapshot-level rules in `core/nervous_system/config/freshness.py`; Meta staleness companions; runner staleness guards. **Missing:** per-timeframe state-age *features* inside the model matrix. |
| Decision archive | `decision_records`, `decision_outcomes` and `context_snapshots` (Postgres, governed path only, has a `failure_stage` column); `core/live_signal_audit.py` (`signal_audit_v1`, score/rank buckets). **Missing:** the full candidate cross-section with full feature vectors, for every module, independent of the DB. |

**Where our own evidence contradicts the paste**

1. **ACE's "80-90% win rate" is not supported by our ledger.** His recaps report
   peak gain rather than exit, and they omit same-day losers. 9 of 15 scored free
   alerts expired worthless; the mean was -0.9% (CI -35..+34). Our reproduction
   recall is 5/25 at ±10 min, which is below the shifted baseline. The teacher
   policy is therefore **gated** (3.2).
2. **"The negatives are the charts Ace looked at and didn't trade."** We cannot
   observe what he looked at. A non-post is *unlabeled*, not a negative. This is
   the positive-unlabeled framing the repo already uses.
3. **Theme features hurt Meta as ranking inputs** (-0.0167 rho), and weekly theme
   churn blocks live use. The relative-structure experiment (2F) must use stable
   groupings and clear that prior.
4. **"The bottleneck is information" is a hypothesis, not a finding.** The
   deployed momentum score is a volatility ranker (L2 IC -0.054). Freshly fitted
   models on interpreted features lost to their own permuted-label nulls.
5. **Option expression cannot rescue the account.** Both legs lose: equity
   episodes average -7.5%, options -16.2% and negative at every DTE bucket. The
   round-trip hurdle is about 35% on the names we trade.

**Safeguards the paste leaves out** (each one has cost us a wrong answer before)

- Marks, not stale trade prints, for any option P&L.
- The OPRA cap: unexpired contracts return 403.
- SIP is 15-minute delayed.
- A permuted-*label* null arm for any trained ranker.
- A random-refusal placebo for any gate.
- Hindsight bias in the human benchmark.

## 2. Rules for every item (from AGENTS.md and past retractions)

- [ ] Trained rankers: within-bar rank IC as the primary metric, plus a
      trained-on-permuted-label `__perm` arm. A shuffled *score* does not catch
      the vol/beta tilt.
- [ ] Gates and abstention: compare against a random-refusal placebo at the same
      trade count. Under negative edge, any throttle "wins".
- [ ] Measure forward outcomes from an executable entry (next open or next bar),
      never from the prior close.
- [ ] Apply a corporate-action guard across the whole holding window (the daily
      bar cache is unadjusted). Apply a liquidity screen using per-date ADV rank,
      not absolute floors (the volume feed switched from IEX to SIP).
- [ ] Rank ATR-unit and vol-matched labels within vol decile.
- [ ] Count embargoes in decision bars, at least the label horizon. 4H bars come
      2 per day.
- [ ] Mark portfolios daily across all phase offsets; period-end MDD understates.
- [ ] Report the MDE with every null, and end every failed test with the next
      variant.
- [ ] State which modules are in scope. The test window stays sealed: Stage 2 =
      2026-05-15+; the confluence test is burned until about 2027.

---

## Phase 0: unblock (do first)

### 0.1 Run Stage 2. These are Baselines A and B.

The bundle is in `strategies/momentum_expansion/data/training_export_stage2/`.
It has 15 arms, 8.74M rows and permuted controls. The trainer
(`ablation/colab/stage2_train_colab.py`) resumes per fold from Drive.

- [x] Upload the matrix, manifest and trainer to `MyDrive/cynolycus_stage2/` and
      run on Colab (GPU, est. 2-4 h on a T4). Return `stage2_results.tar`.
      Done 2026-11-07, 15/15 arms.
- [x] Validation read (2026-11-07,
      `research/execution_quality/26_stage2_label_bakeoff_validation.md`):
      - **No arm qualifies in either slot** under rule v1. Best ICs are +0.012 and
        +0.018, against an MDE of 0.020 and 0.028, so the null is underpowered.
      - L0 is a pure vol ranker.
      - The final test stays **sealed**: it holds only about 4 months, so its MDE is
        about 3.5× validation's.
      - Exploratory: L2_20 without calendar features shows positive residualised
        top-3 excess. Pre-register it as v2.
- [x] Write the local OOF evaluator (2026-11-07):
      `strategies/momentum_expansion/ablation/eval_stage2.py`, tested in
      `tests/test_eval_stage2.py`.
      - Rehearsed on the real 8.7M-row matrix with stub OOF files (about 15 min,
        4.5 GB peak).
      - The selection rule is pre-registered in the module docstring and uses
        BH-FDR on the gap to the noise twin.
      - The test window is opened only via `--open-final-test`, after a selection
        is frozen, and every read is logged.

      This is the seed of the frozen evaluator in 0.3.
- [ ] Baseline A = L0, the deployed momentum label retrained with the same
      recipe. The deployed OOF itself predates the current universe. Baseline
      B = L2_10 / L2_20. Report every arm against its `__perm` arm on
      2026-05-15+.
- [ ] Retire `fwd_max_alpha` from `expansion_labels.py` only as part of the
      retrain that replaces the live label. Do not change the live label on its own.
- [ ] Then follow the existing order: momentum, then HTF (rebuild HTF on all rows,
      not swing-zone rows), then Meta.

**Done when:** an arm is picked on validation and the final test has been opened
once. *As of 2026-11-07 no arm was picked, so this is still open.* Next:
- pre-register v2 (no-calendar L2_20, residualised score) and open the test once it
  holds at least 12 months, about 2027-06;
- or extend the matrix back before 2020 for power.

**Momentum parity fixed and checklist 1.1 run for the pick, 2026-11-19.**
- **Parity:** four live-parity defects fixed (`28_stage3_validation.md`).
- **Selectivity:** no long-only book beats SPY after costs (`29_stage3_selectivity.md`).
- **Final booster:** the Colab script is ready, and shadow scoring is wired into the nightly job.

**Stage 3 validation result, 2026-11-19** (`research/execution_quality/28_stage3_validation.md`).
- **Momentum slot:** picks L2_20__pit (neutral IC +0.0093, q = 0.028). It is regime-dependent and net negative at top-k.
- **HTF slot:** no pick (underpowered).
- **Final test:** sealed until about 2027-06.

### 0.2 Decision-state archive ("never again unarchived")

Start this first. Every day that goes unarchived is lost for good.

**Built 2026-11-13.**
- **Writer:** `core/decision_state_archive.py`.
- **Outcomes:** `core/decision_state_outcomes.py`, run by `scripts/decision_state/attach_outcomes.py`.
- **Parity:** `scripts/decision_state/parity_check.py`.
- **Scheduling:** the outcome and parity stages run after the stamp in `nightly_data_readiness.sh`, and neither is fatal.
- **When it takes effect:**
  - momentum, HTF, Meta and dealer start archiving on their next scheduled subprocess run;
  - 30m swing and SPY run inside the server process, so they need a combined-server restart.
- **Real-data rehearsal:**
  - momentum picks were identical old vs new on 40/40 bars;
  - parity passed, and a 5%-corrupted copy failed it;
  - outcomes vs daily close-to-close returns: Spearman 0.975;
  - storage is about 15–25 MB a day.
- **Still open:**
  - parity for 30m swing and SPY is NOT_CHECKED (no research reference is wired);
  - exits and SPY/30m/governed entries are not joined to fills (the known hole below);
  - the "done" bar needs a week of archive.

- [x] Schema `decision_state_v1`, one row per *scored candidate* (not only picks),
      per module and decision bar:
  - decision timestamp, bar-close timestamp, and the as-of/availability
    timestamp of each input;
  - universe version hash, feature schema hash, and the full feature vector;
  - model bundle hash, score, rank and regime state;
  - each gate that fired, with its reason;
  - chosen action, contracts considered, and the quote snapshot at decision;
  - intent/order id, git commit, and inference wall-clock start/end.
- [x] One shared append-only writer in `core/`. Write parquet partitions under
      `Data/research/decision_state/<module>/<date>/`, immutable and never
      rewritten. It must not depend on the state DB, which was down 10-07..10-24.
- [x] Wire it into momentum, HTF, Meta, 30m swing, dealer, SPY and intraday
      structure. Intraday structure already writes `abstention_v2`; map it, do
      not duplicate it.
- [~] Join to the fill registry and closed-trade ledgers (partial: `order_plan_actions` joins 4H-engine and dealer picks to route, contract, quote and disposition by `decision_run_id`). SPY, 30m and governed
      entries are still not registered (fill-reconciliation note); close that gap
      or record it as a known hole.
- [~] Parity test (built; momentum/HTF/Meta/dealer wired, 30m/SPY NOT_CHECKED; green needs a week of live archive): rebuild features with the research builder at archived
      timestamps and compare with the archived vector within a tolerance. Fail
      loudly on mismatch. This is the research/live parity check AGENTS.md
      requires.
- [x] Nightly outcome attacher, stored separately from the raw archive:
      - forward return, MFE and MAE from an executable entry;
      - first-passage outcome and time-to-MFE;
      - horizons 1/5/10/20 sessions.
- [x] Check the storage budget (measured 2026-11-13: momentum 1.2 MB/run, Meta/HTF/dealer 0.4 MB/run, all modules about 15–25 MB/day, 4–6 GB/yr; option chains are not archived here). Estimate momentum at about 2.9k rows × 134
      columns × 2 bars/day, and measure option chain snapshots separately.

**Done when:** every live module has a week of archive and the parity test is green.

### 0.3 Freeze the evaluator and the experiment manifest

- [ ] Manifest fields:
  - dataset cutoff, universe version, feature schema hash and target version;
  - model family, hyperparameters and seed, plus calibration;
  - cost model version and execution policy;
  - git commit, and a log of every test-window read.
- [ ] Extend `strategies/momentum_expansion/ablation/{metrics,folds,bootstrap}.py`
      rather than starting a new harness. It should report:
  - within-bar IC, precision@1/3/5/10, and net expected return@K;
  - first-passage success, MFE/MAE and large-winner recall;
  - calibration and finite-capital P&L marked daily;
  - results by regime, the `__perm` delta, beta/liquidity-residualised top-k,
    and the MDE.
- [ ] Define sealed chronological validation blocks (for example 2024 trend,
      2025 chop, 2026 rotation). Regime and month dominate seed variance, so
      report per block.
- [ ] Add regression tests for bugs we have already hit:
  - `run_bakeoff` embargo counted in hours instead of bars;
  - competition `choose_best` picking by `val_log_loss`;
  - HTF OOF future selection;
  - MFE measured from the prior close;
  - stray thin timestamps (<200 names per bar);
  - earnings NaN rows dropped by drop-any.

**Done when:** the evaluator refuses to run without a manifest.

---

## Phase 1: cheap diagnostics on data we already have

### 1.1 Net-of-cost selectivity curve with placebo (the paste's item 8)

- [~] Rank depth is done on returns and MFE. Stage 6 found Kelly f* monotone by
      `mom_score` decile and negative growth in the bottom decile.
- [ ] For each module's frozen OOF predictions, take the top
      50/25/10/5/2/1% and report net expectancy after costs, with the equity and
      option cost models separate. Include trade count and CI.
- [ ] Run a random-refusal placebo at each matched count, and a
      beta/liquidity-residualised version.
- [ ] Pick the threshold on validation only, without retraining.

**Kill:** if no cutoff beats the placebo net of costs, selectivity on this score
is exhausted. Go to Phase 2 features rather than more gates.

### 1.2 Abstention / setup-quality layer

Start this only after 1.1 shows usable headroom.

- [ ] Target: does *the deployed policy's* trade clear costs?
- [ ] Inputs:
  - score level and dispersion, and regime;
  - agreement across modules (the confluence study found zero certified
    interactions, so the prior is low);
  - vol-forecaster output.
- [ ] It must beat a matched random refusal *and* its own `__perm` arm. The
      post-mortem found overlays roughly equal to random refusal.

### 1.3 Split alpha from option expression (the paste's item 9)

- [ ] Validity first. Per module, corr(option return, direction-adjusted
      underlying return) should be at least about 0.6; the profit investigation
      found +0.74 on fills. Use fills, quotes and marks, never stale prints.
- [ ] For every option episode in the post-mortem ledger
      (`research/account_postmortem_2026-11-03/`, 1,309 episodes):
      - compute hypothetical shares P&L at the same entry and exit timestamps;
      - build a 2×2 of alpha good/bad × expression good/bad, by module and DTE.
- [ ] Write a rule-based route gate before any model. Breakeven move =
      round-trip spread ÷ leverage, compared with the move predicted by the
      forward-range forecaster. Outputs: `shares | call | spread | none`.
      Validate on the local OPRA cache (2024-02..2026-09).
- [ ] Learned option-expression model: only once some underlying signal shows
      positive net edge (Phase 2). Expression cannot fix negative alpha.

### 1.4 Failure-stage attribution: "why did we miss this trade?"

- [ ] Taxonomy, in order:
      1. `not_in_universe`
      2. `not_discovered` (data missing or stale)
      3. `features_stale_or_wrong`
      4. `scored_low`
      5. `vetoed:<gate>`
      6. `wrong_instrument`
      7. `bad_entry_timing`
      8. `bad_exit`

      Keep it consistent with `decision_records.failure_stage`.
- [ ] Target sets:
      - (a) each week's top-1% movers, screened for liquidity and corporate
        actions;
      - (b) setups the user flags;
      - (c) ACE alerts.
- [ ] Report the share of winners lost at each stage against the random-row base
      rate.
- [ ] Before 0.2 exists, stages 3-5 can only be *reconstructed* from today's
      matrices. Label them as reconstructions (risk of revision and timing
      leakage).

**Done when:** a weekly stage table is generated automatically from the archive.

### 1.5 Blind human-vs-machine benchmark

- [ ] Blind labelling tool:
  - charts are truncated at the decision timestamp and shown in random order,
    with no future bars;
  - reuse `core/shared_plotting/` (`candles.py`, `time_axis.py`);
  - the user records take/skip, direction, invalidation and target before the
    reveal.
- [ ] Sample: about 100 setups the user calls "clearly good", plus 300-500
      matched lookalikes. Match on date, sector, vol decile and trailing
      run-up (the design in `scripts/intraday_audit/matched_control.py`).
- [ ] Hindsight guard: setups chosen after seeing the outcome are biased. Mix
      them with the lookalikes and re-label everything blind; the blind labels
      are the benchmark. Add a prospective arm in which the user logs setups in
      real time.
- [ ] Compare the user's blind precision with each model's rank of the same
      rows. Report the MDE (about ±10pp with 100 positives).

**Kill:** if the user's blind labels do not beat the lookalikes, the "obvious
setup" gap is not measurable here.

---

## Phase 2: representation × label bakeoff (the paste's A-F experiment)

**Mapping:**

| Arm | Contents | Status |
|---|---|---|
| A | Deployed score | Covered by 0.1 |
| B | L2 label (fixed `fwd_max_alpha`) | Covered by 0.1 |
| C | Path-aware targets | Handcrafted |
| D | Freshness-aware features | Handcrafted |
| F | Relative structure | Handcrafted |
| E | Learned latent | GPU, conditional |

Run in the order C → D → F → E. F comes before E because it is cheap. Keep the
paste's cumulative stack, but also run each block alone.

**Shared protocol for every arm:**

- identical purged chronological folds, PIT universe and costs;
- the same final order policy;
- a `__perm` arm for each;
- the evaluator from 0.3.

### 2C. Path-aware multi-output targets

- [ ] Add these columns to `ablation/stage2_labels.py`:
  - L2 relative return, MFE and MAE;
  - first-passage outcome (ATR barriers plus a vertical barrier) and time-to-MFE;
  - an adverse-before-favourable flag;
  - a tail-move flag (top 1% within vol decile).
- [ ] Pre-register how the heads are combined; decide it on validation only. ATR-unit labels pick
      low-vol names, so rank them within vol decile.

### 2D. Hierarchical, freshness-aware state (HARN's causal-state idea)

- [ ] Per-timeframe blocks (15m/30m/1h/4h/1d). A block changes only when its bar
      has closed and become *available*. Add `state_age`, `fresh_this_decision`
      and `bars_since_update` to each.
- [ ] Traps:
      - 4H bars start at 10:00 ET;
      - the last 4H close can be an after-hours print;
      - daily bars are unadjusted;
      - intraday SIP is delayed 15 minutes.
- [ ] Bonus: staleness features would have surfaced the 61-day IEI/VIXY freeze.

### 2F. Relative and residualised structure

- [ ] market → sector → industry → ticker. Features:
  - beta-residual momentum;
  - ticker vs sector and sector vs SPY;
  - percentile within sector and industry;
  - breadth participation, plus dispersion and concentration.
- [ ] Use stable groupings: sector/industry, or trailing-60d correlation
      clusters, which recover 73% of theme cohesion. Test a theme block
      separately; it must beat the -0.0167 prior.
- [ ] Magnitude arm: the forward-range forecaster as an input. OOS log-range R²
      is 0.44-0.51. This mirrors AURA's gain in move magnitude rather than
      direction.

### 2E. Multi-task market-state encoder (conditional)

- [ ] **Gate:** start only if 2D or 2F shows a block gain over `__perm` on
      validation.
- [ ] Small supervised multi-head encoder. Heads:
  - relative return and realised vol;
  - spread/liquidity and cross-sectional rank;
  - sector-relative return, MFE/MAE, and regime.

  Fit on the train fold only. Feed the latent into the XGBoost ranker.
- [ ] Controls: a random-initialised encoder (the paper reports a random ViT as
      a strong baseline) and `__perm`. Compare handcrafted, handcrafted + latent,
      and latent only.
- [ ] Train on Colab GPU with per-unit Drive checkpoints. Prior: Qlib's Alpha158
      XGBoost beats Transformer (`research/execution_quality/14_what_good_looks_like.md`).

### Phase 2 decision

- [ ] Pick on validation and open the final test once. Report the full 0.3
      metric set by regime.
- [ ] Re-rank the 1.5 benchmark rows under each arm: do the obvious setups rise
      to the top?

---

## Phase 3: gated or data-limited

### 3.1 Hawkes-style event intensity for Intraday Structure

- **Data gate:** there is no live NBBO or OPRA flow (OPRA agreement unsigned;
  SIP delayed 15 minutes), so sweep and call/put flow intensity is not available
  intraday.
- **Feasible now:** price acceleration, volume bursts and level tests. News
  arrivals can be used for research only, because their observation time is not
  certified.
- [ ] Multi-decay EWMA intensities (for example half-lives of 5, 30 and 120
      min). Add acceleration and persistence, per ticker and per sector.
- [ ] Evaluate on the intraday decision funnel against matched controls: same
      ticker, different day, ±15 min, ±0.25 ATR run-up.
- Prior: the engine shows no edge over matched controls (83 confirmed setups,
  0 fills). Declined setups do underperform, so the gate is informative.

### 3.2 ACE as a teacher policy

**Gate first:**

- [x] Add the `Zach_ace` alias (1557910141632782409) to the legacy
      `scripts/discord_ledger` pipeline; otherwise new alerts drop out silently.
      (2026-11-06: shared `scripts/discord_ledger/identities.py`, which also adds the new alerts channel;
      ledger rebuilt at `export_refresh_20261103_01/ledger_01`, 249 ACE option entries, 26 from Zach_ace.)
- [x] Re-measure the *exit-based* win rate on the full alert population:
  - include the omitted losers, and score at exit, not at peak;
  - use marks or prints, with the derivative/underlying correlation check.

  (2026-11-06, `export_refresh_20261103_01/exit_scoring_02/`, pre-registered in
  `scripts/discord_ledger/score_ace_lifecycles.py`.) Of the 249 entries, 157 were priced. The rest lacked
  a full contract (82), were ledger duplicates (6) or had no bars (4).
  - **Following his stated exits:** **win 16.1% vs 14.7%** for the same contract bought at a random
    minute that session. Paired difference CI is −0.9..+3.9pp. Mean −24% (CI −60..+24), median −100%.
  - 71% lose ≥90%. The 87 contracts whose prints track the underlying (corr ≥ 0.5) give the same
    picture.
  - **Direction:** the underlying moved his way on 46.5% of entries.

**Stop** if it is not materially above the base rate with a CI.
**→ STOPPED 2026-11-06: the gate failed.** The steps below are not to be built.

**If it passes:**

- [ ] ~~Build the teacher dataset as a typed chain: watchlist → zone tag →
      entry → trim → exit. Non-alerts are *unlabeled*.~~ Gate failed.
- [ ] ~~Extend `ace_pu_timing_experiment` with a `__perm` arm, then analyse where
      the bot's ranks disagree with the teacher.~~ Gate failed.
- Power today is 201 decisions with a timing-recall MDE of about 29pp. That is
  not enough, so accumulate prospective alerts.

### 3.3 Learned option-expression model

- [ ] Gated by 1.3 and Phase 2. Candidate inputs:
  - predicted move size and time-to-move;
  - DTE, delta and IV level/term structure;
  - theta, spread and breakeven distance;
  - liquidity and expected IV behaviour.

---

## Not now: Optional / Outside Roadmap

The paste lists these and we agree:

- AURA-style RL tool selection;
- full world-model planning and RoboJEPA scaling;
- MaD-RL and SMART regime memory;
- score-centering and Wasserstein portfolio optimisation.

They are downstream of edge. The post-mortem showed that overlays roughly match
random refusal under negative edge, and robust sizing does not create alpha.

## Recommended order

0.1 and 0.2 in parallel (0.2 starts now) → 0.3 → 1.1 → 1.3 → 1.4 → 1.5 → 2C →
2D → 2F → 2E (conditional) → 3.x (gated).

None of this is expected to fix live P&L in the near term. Every module still has
negative measured edge, and entries were reopened on 2026-11-04.

## Source papers (read before 2E and 3.1; claims unverified here)

- Towards Financial World Modeling: https://arxiv.org/html/2610.09048v2
- HARN (event time vs representation time): https://arxiv.org/html/2609.26822v1
- AURA (RL tool selection; move-magnitude gains): https://arxiv.org/html/2609.36914v1
- Hawkes-process market making: https://arxiv.org/abs/2610.16524
- SciExam (freeze, then expose to hidden future): https://arxiv.org/abs/2610.10513
- The "AI Frontier Research Tracker" Google Sheet (`Finance_Lab` /
  `Research_Master`) was the chat's backlog; it was not read in this pass.
