# Reintroducing RL into CynolycusBot — staged plan

2026-09-23. Source: user's ChatGPT thread "Simplify RL Trading"
(`https://chatgpt.com/share/6ab46686-1f4c-83ea-b686-65c57e5e999e`), full text recovered
from the share page's embedded payload.

Status: **plan only, nothing implemented.** Every stage below has an explicit kill
criterion and a pre-registration requirement. No stage changes live trading.

---

## 0. What the thread actually proposes

Three user turns, three assistant turns. The substantive proposals, in order:

1. **Invert the problem.** Stop asking RL "what trade should I make". Ask "given an
   opportunity the architecture already likes, when/if should I act?" and later "is
   continued exposure still worth it?" Small discrete action spaces, compact states,
   deterministic risk/execution left alone.
2. **V1 = entry timing.** Episode = one candidate. `A = {WAIT, ENTER, REJECT}`.
   Direction is immutable (inherited from upstream intent). Reward = the trade's own
   risk-normalized outcome, **measured against a deterministic enter-immediately
   baseline**, so REJECT has a real opportunity cost and "never trade" is not a
   winning policy.
3. **V2 = exit.** `A = {HOLD, EXIT}`. `Q(s,HOLD)` vs `Q(s,EXIT)`. Trained separately
   from entry. **V3 = sizing**, last.
4. **Reward must preserve convexity.** R units (`(P_exit − P_entry) / initial risk`),
   unclipped, so a +6.8R monster is worth more than a +0.3R winner. Explicitly *not*
   hit rate, and explicitly not a pile of hand-authored bonuses/penalties.
5. **Prefer value-based methods** (DQN-style `Q(s,a)` per action) over PPO, for
   interpretability and because the problem is small, discrete and offline.
6. **Two experiments BEFORE any RL.**
   - **Exp A — retrieval:** what fraction of the historical fat right tail does the
     candidate layer surface *before* the move? If recall is poor, fix discovery, not
     policy. Demote XGBoost from gatekeeper to feature; make Stage 1 intentionally
     high-recall / low-precision.
   - **Exp B — decidability:** on identical candidates and state, bake off
     XGBoost / LightGBM / MLP / contextual bandit / DQN / PPO. RL must earn its
     complexity.

That sequencing instinct is right, and it matches this repo's own standards. The
specifics need five corrections before we spend any compute.

---

## 1. Audit — what CynolycusBot already has

### 1.1 A V1 entry-timing RL module already exists, and it is the design in the thread

`strategies/spy_intraday/Policy/Execution_Agent/` is, almost line for line, §1–§7 of
the thread:

| Thread proposal | Existing implementation |
|---|---|
| Episode = one candidate, upstream intent frozen | `Execution_Agent/README.md`: HTF command inputs `htf_dir/htf_conf/time_since_flip_min/htf_atr_pct` from a frozen 15m agent trace |
| Direction immutable | Actions are direction-gated; `ACTION_ENTER/SCALE_IN/SCALE_OUT/EXIT` all alias `ACTION_EXECUTE` (`env.py:10-18`) |
| `A = {WAIT, ENTER}` | `ACTION_WAIT = 0`, `ACTION_EXECUTE = 1`, one-shot per pending event window (`env.py:10-11`) |
| Reward vs deterministic enter-immediately baseline | `env.py:59-61`: `reward = (agent_event_net − baseline_event_net) − MAE_penalty`, zero on non-execute steps |
| Value/oracle first, RL as fine-tune | README "Stage A/B": oracle event labels + supervised XGBoost, *then* PPO fine-tune |
| Leak-free intent | `build_intent_oof.py` builds walk-forward OOF HTF commands from fold manifests |

`Execution_Agent/oracle.py:143` `build_oracle_event_labels` already computes the
hindsight-optimal execution timing that Stage A imitates.

**Scope limits that matter:** SPY 1-minute only; its parent module (`spy_daytrader`) is
dormant — 6 closed trades, −$106 — and AGENTS.md forbids letting a non-production
module anchor a live conclusion. So this is a **reference implementation to port**, not
evidence that the approach works.

### 1.2 The old PPO failure mode is still visible in the code

`strategies/spy_intraday/Policy/Agent/env_config.py` carries ~25 reward knobs:
`flip_penalty_ret`, `trade_penalty_ret`, `hold_penalty_ret`, `saturation_penalty_ret`,
`magnitude_decay_lambda`, `flat_position_penalty_ret`, `dir_switch_penalty_ret`,
`size_change_penalty_ret`, `convex_k1/k2/theta`, `convex_mfe_thresholds/bonuses`, …
and the action is a **continuous float** (`Agent/env.py:291 step(action: float)`), i.e.
direction *and* size in one head.

That is exactly the failure the thread diagnoses: too many hand-authored incentives,
too large an action space, direction and sizing learned jointly from a noisy reward.
Worth noting `convex_mfe_thresholds/bonuses` shows a right-tail convex reward was
*already* attempted there — so "add a convex reward" is not a new idea in this repo and
needs a reason why it would work now.

`strategies/spy_intraday/Models/off_policy_ppo_agent/off_policy_ppo_agent.py` is **not**
a trading agent — it is a 3-class news-sentiment classifier with class-weighted rewards
and a dilated CNN. Unwired. Ignore it.

`torch` and `gymnasium` are already in `requirements.txt` (lines 8, 14). No new deps.

### 1.3 Exp A's and Exp B's substrate already exists

`scripts/horizon_thesis/build_decision_panel.py` builds the shared decision panel. The v1
artifact on disk (`research/execution_quality/data/decision_panel.parquet`) is
**1,567,029 rows x 57 cols** of (4H decision bar x ticker), carrying

- walk-forward **OOF** `mom_score` and `htf_score` (21-day embargo) back to 2022-11,
- `fwdret_/mfe_/mae_` at holds 5/10/15/20/30 measured from the **first executable price**
  (next session's open — the `23_...md` §0 correction),
- the PIT regime panel (10 columns), past-return/vol/liquidity/beta factors,
- a day-5 checkpoint block and `rem_5_{h}` continuation targets,
- corporate-action flags, marked not dropped.

The builder now targets `decision_panel_v2.parquet` (PANEL_VERSION 2), which is **not on
disk**, and `load_panel` raises on a version mismatch — so a rebuild is step one of Stage 0,
and v2's schema must be re-inspected then rather than assumed equal to v1.
`oof_rank_depth_mom.parquet` / `_htf.parquet` (1,567,029 rows each) are present.

### 1.4 The right-tail label already exists and is already validated

`strategies/momentum_expansion/labels/expansion_labels.py` computes `fwd_max_return`,
`fwd_max_alpha` (MFE minus SPY), `fwd_atr_adj_return`, `fwd_max_drawdown`,
`trend_persistence` over a 25-bar (~10 trading day) forward window, with an explicit
no-leakage guarantee in the module docstring.

`_cross_sectional_expansion_survival_score` (`:139`) composites them by **within-bar
percentile rank** with weights `.40/.25/.20/.15` (`momentum_config.py:151+`).
LIVING_SUMMARY (2026-09-22) records corr 0.678 with a triple-barrier outcome, top decile
97.2% target-first, bottom decile 99.4% stopped-first.

> **Corrected 2026-09-23 by Stage 0.** This matrix holds **1,081 tickers**, not the 3,089
> present as per-ticker files under `data/processed/features_4h/`. The momentum pool and the
> meta universe are effectively the same names, so there is no broad pre-gate population
> inside it and Stage 1 was restructured into two layers (universe, then gate) against the
> daily bar cache. See `research/rl_reintroduction_2026-09-23/00_preregistration.md` §1.2.

### 1.5 The gate the thread wants to loosen is one function with seven thresholds

`strategies/momentum_expansion/inference/candidate_filter.py:14` `momentum_candidate_mask`,
driven by `MOMENTUM_CANDIDATE_FILTER_CONFIG` (`momentum_config.py:126-135`):
low-price exclusion, `min_dollar_vol_pctile_252 0.20`, `max_dist_to_52w_high_atr 8.0`,
`min_xsec_near_high_rank 0.60`, `min_rs_spy_20 0.0`, `min_xsec_ret_20_rank 0.60`,
`min_range_pos_20 0.45`. Exp A is a sweep over this dict. Note
`TRAINING_MATRIX_CONFIG["apply_momentum_candidate_filter"] = False` — the ranker already
trains on broad rows and the filter is applied *after* scoring, which is half of the
"demote XGBoost from gatekeeper" change already in place.

### 1.6 The integration seam already exists

`core/nervous_system/contracts/intent.py:48` `TradeIntent` already carries
`score_components`, `entry_window`, `selected_bar`, `preferred_entry`, `feature_timestamp`
vs `created_at` (validated ordering), `snapshot_id` lineage, `model_version`,
`feature_version`, `config_version`, `reason_codes`. An entry policy slots between intent
production and `core/nervous_system/policy/engine.py` without inventing a contract.

Decision logging: `core/live_signal_audit.py` (`signal_audit_v1`, `order_audit_v2` with
decision-time IV and greeks as of 2026-09-22), and
`Data/inference/intraday_structure/decision_events.jsonl` at 285,744 rows.

### 1.7 What does not exist

- No contextual bandit, no DQN, no fitted-Q anywhere (one mention of "contextual bandit"
  in `docs/superpowers/plans/2026-07-25-options-instrument-routing-experiment.md`, which
  is inside the retracted series).
- No high-recall opportunity generator *separate from* the gatekeeper.
- No sequential HOLD/EXIT policy — every exit in the 4H modules is a deterministic rule.
- No forward-RV label wired anywhere, despite `research/regime_coverage_2026-09-21/forward_rv_predictability.py`
  showing forward 20d RV is predictable at OOS R² = 0.742 (vs 0.003 for forward return).

---

## 2. Five corrections — where the thread collides with measured repo evidence

These reorder the plan. Each is a repo result, not an opinion.

### C1. The 35% options hurdle dominates the entry-timing decision by ~5x

`research/execution_quality/23_rank_depth_and_options.md` §2: momentum's **ordering is
real** — within-bar rank permutation, 20,000 draws, significant in **9 of 9** cells
(holds 8/10/15 × k 1/2/3), edge **+4 to +8pp** of underlying return, most at p ≤ 0.003.
Meta is 5 of 9. So there *is* a decidable selection margin, and the thread's guess
("decent recall, terrible selection") is roughly half right.

Same doc §5, on 944 contract-hold observations over 205 contracts and three expiry cycles,
validation passing (corr(option ret, underlying ret) = +0.63…+0.88): the option round trip
costs **~35% of premium** (12.8pp premium decay on a flat underlying + 23.6pp measured
spread). Every module, every hold, every cycle negative after cost.
`research/regime_coverage_2026-09-21/README.md` reproduces it independently:
option −$953/trade vs equity −$289/trade, t = −2.55, p = 0.011; within momentum, same
signals both routes, **−36.93pp gap**, p = 0.006.

> **A 4–8pp entry-timing improvement cannot pay a 35% round-trip hurdle.**
> Entry-timing RL applied to the option route is arithmetically incapable of fixing P&L.

Consequence: the **instrument-route decision is a larger free variable than entry
timing**, and it belongs in the pre-RL bake-off (Stage 2) as a first-class arm, not as a
downstream detail. It is also the one lever already known to be worth tens of percent.

### C2. "Loosen retrieval for recall" is the worst case for our bar cache

`Data/shared/bars/1d` is **unadjusted and survivor-shaped** — stated in
`build_decision_panel.py`'s own docstring and recorded in memory (a WOLF +2,189% artifact
inflated a prior study 3–6x). A fat-right-tail study is precisely where that bites:
reverse splits manufacture fake monsters, and delisted names are simply absent, so
measured recall is biased **upward** on survivors while the "monster" population is
contaminated.

The guard exists (`research/execution_quality/data/corporate_action_flags.parquet`,
`scripts/rank_depth/apply_ca_guard.py`, `core/corporate_actions.py:160`) and
`25_...md` §B.3 records that applying it **halved** a depth gradient: flagged rows are
186x over-represented at rank 1. Exp A is invalid without it, and it must additionally
state its survivorship exposure explicitly rather than assume it away.

### C3. A recall study of this exact shape has already returned a hard null

`research/regime_coverage_2026-09-21/recall_precision.py` + README: 2,811 tickers,
429k stock-days, event = +15% max gain within 20 sessions. **Base rate 25.87%.**

| signal | fires | recall | precision | lift |
|---|---|---|---|---|
| rising EMA100 touch ±1 ATR | 12.52% | 12.46% | 25.75% | **1.00x** |
| union of EMA50/100/SMA200 | 27.71% | 27.32% | 25.51% | **0.99x** |
| [ref] extended >3 ATR | 23.14% | 20.58% | 23.01% | 0.89x |

**Recall equals the firing rate to within 0.5pp on every row** — the exact signature of
zero information. Two implications for Exp A:

1. **Recall alone is free and must never be the headline.** The repo's own script says
   so in its docstring. The number that decides anything is **precision vs base rate
   (lift)**, or equivalently recall-at-fixed-firing-rate.
2. **"+15% within 20 days" is not a fat right tail in this universe** — it happens on
   26% of stock-days. Exp A's event definition must be far more extreme (top 1%, or
   ≥3R risk-normalized, or +50% MFE) or it measures nothing. The thread's suggested
   thresholds (+8% in 3 days, +4R in 5 days) need calibrating against this base rate
   *before* the experiment runs, not after.

### C4. Exploration is not our problem — so V1 and V2 are not RL

The market is **exogenous** to us at our size, and history is replayable. Therefore we
have **full counterfactual feedback**: for any candidate at any timestep we can compute
the realized outcome of WAIT, of ENTER, and of REJECT. Likewise for HOLD vs EXIT at
every bar of an open position.

That changes the algorithm class materially:

- It is **not a contextual bandit.** Bandits exist because feedback is *partial* — you
  only see the arm you pulled. We see every arm. Importing bandit machinery (Thompson
  sampling, UCB, IPS/DR estimators) would be solving a problem we do not have.
- **V1 (entry) reduces to cost-sensitive supervised learning on the advantage.** The
  label is `R_action − R_baseline`, which we can compute exactly. No policy gradient, no
  value bootstrapping, no exploration schedule.
- **V2 (exit) is optimal stopping with full information.** On a replay we can compute the
  hindsight-optimal stopping time per trade by backward induction, exactly. The genuine
  difficulty is *generalization* — learning a stopping rule from features that works on
  unseen paths — not exploration.

So model-free RL (DQN/PPO) is warranted only as a **function approximator / fine-tune on
top of an oracle-imitation baseline, and must beat that baseline to justify itself.**
This is precisely the Stage A/B structure already in `Execution_Agent`. It also makes the
RL step small, cheap and falsifiable — which is what the user asked for.

Practical upshot: **the first genuinely new artifact we should build is an exit oracle,
not an agent.** It tells us the *headroom* of a perfect exit policy over today's
deterministic rules, in R units, on 1.57M rows. If that headroom is small, every
sequential-RL idea downstream is dead for free.

### C5. The reward must be R units, and `expansion_survival_score` cannot supply it

The thread is right that clipping the tail is fatal. But
`_cross_sectional_expansion_survival_score` is a **within-bar percentile rank composite** —
a +50% MFE and a +5% MFE both map toward 1.0. It is the correct target for a *ranker* and
the wrong one for a convexity-preserving reward.

So the reward for every stage below is
`R = (P_exit − P_entry) / (k · ATR_entry) − costs`, unclipped, with `k` fixed and stated,
and adverse excursion charged separately (`− λ · MAE/risk`) rather than folded in. Report
the **full return distribution** (p1/p25/median/p75/p99, share > +3R) alongside the mean,
because the whole thesis is that the mean hides the tail.

Two further constraints from the repo's standards:

- **Permuted-LABEL null is mandatory.** `25_...md` §B.1 / `scripts/horizon_thesis/topk_depth_null.py`:
  a model trained on within-bar permuted labels shows **+2.1 to +7.6pp** fake top-k
  excess on raw returns via a vol/beta tilt (null top-3 carries 1.41x ATR, 1.44x beta,
  1.56x realized vol). A shuffled-*score* null is ~0 and proves only that the harness is
  clean. Any stage reporting top-k excess carries both arms or it is not reportable.
- **MDE with every null** (AGENTS.md). 88 momentum / 132 HTF / 42 meta closed trades means
  anything computed on realized trades is underpowered by construction. Every stage runs
  on the 1.57M-row panel with pseudo-trades and reports its minimum detectable effect.

---

## 3. The staged plan

Ordered so that each stage can kill the ones after it, cheapest first. Stages 0–3 contain
**no RL at all**. Nothing here touches live trading.

### Stage 0 — pre-registration and substrate (no modelling)

1. Rebuild the panel: `./.venv/bin/python scripts/horizon_thesis/build_decision_panel.py`
   → `decision_panel_v2.parquet`. Verify row count, date span, null rates, and that
   corporate-action flags are populated.
2. Write `research/rl_reintroduction_2026-09-23/00_preregistration.md` fixing, in advance:
   - **Splits.** Train ≤ 2025-06-30, validation 2025-07-01…2026-03-31, **test 2026-04-01
     onward, opened once per stage at most.** Memory records the test block is already
     partly burned by `confluence_discovery`; state exactly which comparisons have touched
     it and treat the remainder as a budget.
   - Embargo (21 days, matching the OOF construction), universe, cost model, `k` in the
     risk denominator, λ on MAE, and the exact metric table every stage must emit.
   - The kill criteria below, verbatim, so they cannot be renegotiated after seeing results.
3. Calibrate the right-tail event definition against base rates (C3) and freeze it.

**Deliverable:** one preregistration doc + a rebuilt panel. **Cost:** low.

### Stage 1 — Exp A: right-tail retrieval audit (no RL)

Question: of the extreme forward moves in the tradeable universe, what share does the
candidate layer surface *before* they happen, and does loosening the gate buy recall
without destroying lift?

- Event: frozen in Stage 0. Report **at least two** definitions (a cross-sectional one —
  top 1% of forward MFE within bar — and an absolute risk-normalized one — MFE ≥ 3R).
- Sweep `MOMENTUM_CANDIDATE_FILTER_CONFIG` from current settings toward fully open, on a
  grid over the seven thresholds; at each point report firing rate, recall, precision,
  **lift over base rate**, and recall-at-matched-firing-rate.
- CA guard applied (C2); report the answer with and without it, per `25_...md` §B.3.
- State the survivorship exposure numerically: how many tickers in the 1d cache have a
  final bar before the study end, and what the study cannot see.

**Kill / branch criteria**
- If lift ≈ 1.00x at every gate setting (the C3 outcome), then the gate carries no
  information and *loosening it is free but pointless* — retrieval is not the binding
  constraint, and Stage 2 proceeds on the current candidate set. Do not build a
  high-recall engine on a null.
- If recall at matched firing rate rises materially as the gate opens, retrieval *is*
  leaving monsters on the table, and the Stage 2 candidate set is the loosened one.
- Either way this stage produces a number we do not currently have, and it is cheap.

### Stage 2 — Exp B: is the ENTER/PASS decision decidable, and on which instrument? (no RL)

Question: on identical candidates and identical state, how much risk-adjusted advantage
is available from a *selection* decision, and does any of it survive the route hurdle?

Arms, all on the same rows, same splits, same cost model:
1. Current deployed OOF score, top-k (the incumbent baseline).
2. Cost-sensitive regression/classification on the **advantage in R units** (C5).
3. LightGBM and a small MLP on the same target.
4. **Permuted-label null** for each of the above (C5, mandatory).
5. **Route arm:** equity vs option, the latter charged the measured ~35% round-trip
   hurdle from `23_...md` §5. This arm answers C1 and is the highest-value cell in the
   table.

Report: per-arm R-unit distribution (not just mean), top-k excess vs the equal-weight
within-bar control, both nulls, MDE, and performance by regime.

**Kill / branch criteria**
- If no arm's advantage exceeds its permuted-label null: there is no decidable selection
  margin at this state representation, and **RL is not the missing ingredient** — stop and
  report that, per AGENTS.md's "next testable variant" rule (the next variant would be
  state enrichment, not a new optimizer).
- If the margin survives the null but is < the option hurdle, then the conclusion is
  **route equity, and RL's entry-timing ceiling is ~4–8pp** — worth having, not worth
  building first.
- If the margin survives on the equity route with room to spare, Stage 5 (entry RL) is
  justified.

### Stage 3 — exit headroom: the oracle stopping study (no RL)

**This is the stage I would prioritise**, because it is cheap, it is new, and it gates
everything sequential.

- On the panel, for each pseudo-entry, compute by **backward induction** the
  hindsight-optimal exit bar under the real cost model, in R units — the full-information
  optimum (C4).
- Compare three policies on identical entries: today's deterministic rule
  (`max_holding_4h_bars = 30`, the premium stop), a fixed-horizon hold, and the oracle.
- **Headroom = oracle − deterministic**, reported as an R-unit distribution with MDE.
- Then the decisive sub-question: fit a *feature-based* imitation of the oracle stopping
  rule (Stage A of the `Execution_Agent` pattern) and measure how much of the headroom it
  recovers out-of-sample. Oracle headroom is an upper bound; imitation recovery is the
  realistic one.

**Kill criterion:** if out-of-sample imitation recovers a negligible share of the headroom,
the exit decision is not learnable from current features and **V2 exit RL is dead** — the
next variant is features (forward-RV is the obvious candidate given R² = 0.742), not an
optimizer.

### Stage 4 — first RL: fitted-Q on HOLD/EXIT

Only if Stage 3 shows recoverable headroom.

- `A = {HOLD, EXIT}`, direction immutable, position sizing fixed, entries taken from the
  deployed score so the comparison is clean.
- **Fitted Q-Iteration** (batch, offline, on the replay) before any deep net — it is the
  right algorithm for full-information offline optimal stopping, and `Q(s,HOLD)` vs
  `Q(s,EXIT)` gives the per-decision diagnostic the thread wants.
- Benchmarks it must beat, in order: deterministic rule → oracle imitation (Stage 3) →
  itself with permuted labels. A DQN or PPO arm is added **only** if FQI beats imitation,
  as a function-approximation comparison.
- Reward per C5. Report the R distribution and the share of trades whose exit moved.

**Kill criterion:** does not beat Stage 3's oracle imitation out-of-sample → RL adds
nothing over supervised imitation here; report and stop.

### Stage 5 — entry timing RL (WAIT / ENTER / REJECT)

Only if Stage 2 cleared. Port the existing `Execution_Agent` design to a 4H module
(momentum first — it has the only 9/9 ordering result) rather than writing a new
environment:

- Reuse the baseline-relative reward shape from `Execution_Agent/env.py:59-61` verbatim in
  spirit: `reward = (R_agent − R_enter_immediately) − λ·MAE`, zero on WAIT steps.
- Reuse the Stage A/B structure: `oracle.py`-style hindsight timing labels → supervised
  head → RL fine-tune, each measured against the previous.
- State: 10–20 **interpreted** inputs (module scores, regime scalar, theme strength,
  catalyst confidence, distance-from-trigger, relative volume, ATR-normalized extension,
  minutes-since-signal), not raw indicators — the thread's representation point, which
  this repo's own evidence supports (`project_themes_grouping_not_ranking`: raw theme
  features *hurt* Meta at −0.0167).
- Entry window from the producing strategy, not a global constant.

**Kill criterion:** cannot beat enter-immediately out-of-sample after costs → stop, per
the thread's own V1 gate.

### Stage 6 — sizing

`A = {0, 0.5, 1.0}` multipliers on the risk budget. Last. Only after Stage 4 or 5 has
produced a validated improvement. Sizing interacts with portfolio construction and risk
limits, so it needs its own preregistration and cannot be bolted onto an entry policy.

---

## 4. Where the code goes

Respecting the existing separation of concerns:

**ALL SIX STAGES ARE RUN.** Read
`research/rl_reintroduction_2026-09-23/README.md` for the synthesis; per-stage findings are the
`0N_*_findings.md` files beside it. Headline: every learned layer failed against its own
control, the largest measured loss is the exit *configuration* (~0.57R/trade) rather than a
missing model, and **this plan's correction C1 is retracted** — the options hurdle is ~6% of
underlying move, not ~35%, because C1 omitted leverage. The sections below are kept as the
original plan of record; where a stage's result contradicts them, the findings docs win.

| Artifact | Location | Why |
|---|---|---|
| Preregistration + findings docs | `research/rl_reintroduction_2026-09-23/` | Matches `regime_coverage_2026-09-21/`, `execution_quality/` |
| Stage 1–3 experiment scripts | same dir, one script per question | Matches `exp_a_stop_sweep.py` / `exp_b_dte_reprice.py` naming |
| Panel rebuild | reuse `scripts/horizon_thesis/build_decision_panel.py` | Do not fork it |
| CA guard | reuse `scripts/rank_depth/apply_ca_guard.py`, `core/corporate_actions.py` | Do not reimplement |
| Permuted-label null | reuse the protocol in `scripts/horizon_thesis/topk_depth_null.py` | Same harness → comparable numbers |
| Reusable policy code (Stage 4+) | `signals/` or a new `policies/entry_timing/` module with its own `tests/` | Keep research scripts and production policy separate |
| Live integration (much later) | between `TradeIntent` production and `core/nervous_system/policy/engine.py` | `contracts/intent.py:48` already carries the lineage fields needed |

Training: CPU-first (FQI, XGBoost, small MLP all fit). If any stage needs a GPU I will say
so explicitly before running it.

---

## 5. Risks and honest expectations

1. **The most likely outcome of Stage 2 is a null**, given C1 and the repo's track record
   (confluence: zero certified interactions; trust gate: AUROC 0.429; theme features into
   the ranker: −0.0167). The plan is built so a null costs days, not months, and so each
   null names its successor variant.
2. **Test-set budget is the scarce resource**, not compute. Five stages × one test look
   each is already a lot of the remaining block. Stages 1 and 3 can run entirely on
   train+validation.
3. **C1 is the uncomfortable finding.** The measured selection edge (4–8pp) is an order of
   magnitude smaller than the measured options hurdle (~35%). If that holds, no amount of
   RL sophistication on entry timing fixes the P&L, and the honest highest-value work is
   the instrument-route decision — which is also the open roadmap item in LIVING_SUMMARY
   (the (a)/(b) DTE-vs-hold decision per module). Stage 2's route arm is there to force
   that comparison into the same table rather than leaving it implicit.
4. **Survivorship and unadjusted prices** are a real limit on any right-tail claim from
   this cache (C2). Stage 1 must report what it cannot see; a paid point-in-time vendor is
   the only full fix and is out of scope here.
5. **Roadmap alignment.** Stages 0–3 support the current roadmap (they are measurement on
   existing modules and directly inform the open options-structure decision). Stages 4–6
   are new capability and should be treated as **conditional** on Stages 1–3, not as
   committed work.
