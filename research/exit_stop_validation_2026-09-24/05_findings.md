# Steps 2-6 — results, and three of my own hypotheses killed

2026-09-24. Scripts: `02_stop_width_sweep.py`, `03_gate_coverage.py`, `04_route_vs_rank.py`,
`05_sizing_and_triggers.py`, plus the re-run of
`research/rl_reintroduction_2026-09-23/03_exit_oracle.py` carrying both exit arms.
**Nothing live was changed.**

---

## Step 2 — the real exit policy, measured. The correction is confirmed.

Re-ran the exit study with `live_approx` replaced by `core.live_4h_exec.ExitPolicy` as wired
(no trail, 1.5-ATR stop, +30% trim of 16%, 53 bars ≈ 27 sessions), keeping the superseded
config as a named baseline. Momentum top-3, 2,430 entries, R units:

| policy | mean R | median R | share ≥ +3R | share ≤ −1R | median exit day |
|---|---|---|---|---|---|
| `old_config` (superseded) | 0.091 | −0.148 | 1.4% | 0.0% | 12 |
| **`live_policy` (actual)** | **0.416** | −0.382 | **7.3%** | **0.0%** | 27 |
| `fixed_30` (just hold) | 0.662 | +0.039 | 10.2% | 16.6% | 30 |
| `oracle` | 2.243 | 1.087 | 21.4% | 0.2% | — |

**The change the repo shipped was worth ~+0.33R per trade** (0.091 → 0.416), which settles the
correction: my "0.57R prize" was mostly already banked. HTF is the same shape (0.161 → 0.393).
A remaining **−0.25R** separates the live policy from simply holding 30 sessions.

### Step 2b — sweeping the two live knobs

Stop width × horizon, everything else at live values. Momentum:

| stop (ATR) | h=10 | h=20 | **h=27 (live)** | h=30 | h=40 |
|---|---|---|---|---|---|
| 1.0 | 0.149 | 0.253 | 0.348 | 0.395 | 0.401 |
| **1.5 (live)** | 0.185 | 0.301 | **0.416** | 0.465 | **0.480** |
| 2.0 | 0.189 | 0.319 | 0.440 | 0.496 | 0.516 |
| 3.0 | 0.205 | 0.360 | 0.495 | 0.556 | 0.603 |
| off | 0.202 | 0.376 | 0.526 | 0.595 | **0.648** |

Mean R rises monotonically in both directions. But the `≤ −1R` share is **0.0% at stops ≤1.5
ATR** and jumps to 17-44% above it — that is mechanical, not magic: a 1.5-ATR stop caps the
loss at −0.75R, and any stop wider than 2.0 ATR admits losses worse than −1R by construction.

**The one free improvement: keep the stop, lengthen the horizon.** `1.5 ATR / 40 sessions`
dominates the live `1.5 ATR / 27` on every column — mean **0.480 vs 0.416**, share ≥+3R
**9.2% vs 7.3%**, share ≤−1R **0.0% in both**. That is +0.064R/trade with no additional tail
risk, i.e. `horizon_bars` 53 → ~80. The costs it does carry are capital turnover and a longer
hold, which this study does not price. HTF shows the same ordering (0.393 → 0.434).

Also worth knowing: **the 1.5-ATR stop fires on 58.5% of momentum positions.** Going wider
buys mean return by accepting losses beyond −1R; that is a risk-preference call, not a bug.

Scope: equity path. On an option the stop also prevents riding a contract toward zero, which a
share path cannot represent, and no historical option premium path exists (2026-07 retraction).

---

## Step 3 — the coverage hole is 9.2%, not 20%. My earlier figure was wrong.

My Stage 1 number conflated two different things. The momentum matrix's ticker set is **not a
constant pool** — names enter and leave (TM appears only from 2025-08; NNDM only
2021-01..2022-03). A session outside a ticker's own span means "not in the universe then",
which is correct behaviour.

| | rows | |
|---|---|---|
| pool rows | 1,222,332 | |
| outside the ticker's own universe span | 133,322 | expected, not a hole |
| effective denominator | 1,119,483 | |
| covered by the gate | 986,161 | **88.09%** |
| **genuine in-span gap** | **102,849** | **9.19%**, 2,594 tail events at 2.52% = **2.29x** covered |

Causes of the missing rows:

| cause | rows | tail rate | vs covered |
|---|---|---|---|
| `not_in_universe_yet` | 129,719 | 3.55% | **3.22x** |
| `feature_nan_in_span` | 102,408 | 2.50% | **2.27x** |
| `left_universe` | 3,603 | 2.11% | 1.92x |
| `missing_bar` | 441 | 6.80% | 6.17x |

Two readings:

1. **The fixable gap is `feature_nan_in_span`** — 102,408 rows across 1,043 tickers where 4H
   bars exist and span the session but the matrix has no row, so features computed to NaN and
   were dropped. Tail 2.27x denser there. That is the actionable 9.2%.
2. **The bigger effect is not a bug at all.** `not_in_universe_yet` carries a **3.22x** tail
   rate: names are admitted to the momentum pool *after* their big moves. That is
   universe-admission latency, and it independently corroborates Stage 1's L1 finding from a
   different direction. It is also the harder problem — the fix is earlier discovery, not a
   pipeline patch.

Note the training matrix was last written **2026-06-14** while the feature files were
refreshed **2026-09-22**, so the matrix is ~3 months stale relative to its own inputs. Any
rebuild should start there.

---

## Step 4 — my routing hypothesis is NOT supported

I predicted the option book would skew to deeper, more liquid ranks. Measured from the
`contract_selection` records (1,722 entries, 4 modules, 491 tickers, 2026-07-14..09-22):

| | top-3 | rank 4+ | delta |
|---|---|---|---|
| option-routed, LIVE modules | 11.3% | 12.9% | **+1.7pp** |
| option-routed, ALL modules | 19.9% | 18.8% | **−1.1pp** |

**The sign flips depending on whether the deprecated module is included, and the magnitude is
1-2pp either way. That is noise. The hypothesis is rejected** — the option book is not
preferentially deep.

What the data shows instead is more useful:

* **Options are rare at every depth** — 11-13% of routed entries become options, flat across
  rank buckets.
* **The dominant blocker is chain liquidity, and it is severe.** `illiquid_option` is the
  largest equity-routing reason after "already held" (340 live entries), and the recorded open
  interest on blocked names has a **median of 0** at ranks 1, 2, 4-5 and 6-10.
  **194 of 386 blocked entries had oi = 0** — literally no open interest. `price_floor` blocks
  another 211.
* Concretely: at rank 1, of 33 entries routed to equity, **19 were blocked as illiquid and 9 by
  the price floor**.

So the routing logic is already doing the right thing — it sends unhedgeable names to equity,
and equity is the better-performing route. The options problem is not mis-selection by rank;
it is that the ~12% which do get options lose heavily.

---

## Step 5 — `top_n = 3` is live and binding, confirmed from the ledger

The pooled audit looked alarming (median `rank_pct` 0.60; 66% of triggered entries at rank ≥4),
but splitting by date settles it: **the last rank>3 triggered entry was 2026-09-08** — the exact
date the rank-depth study cut `top_n` from 10 to 3. After that, only ranks 1-3 appear.

| month | n | median rank | ranks 1-3 | ranks 4-10 |
|---|---|---|---|---|
| 2026-07 | 415 | 4.0 | 150 | 265 |
| 2026-08 | 476 | 5.0 | 139 | 337 |
| 2026-09 | 127 | 3.0 | 66 | **61** (all before 09-08) |

**No bug — and this is positive evidence that the config change actually shipped and binds.**
`top_n_at` applies `n_eff = min(top_n, top_pct·N)`; the `.head(10)` at `runner.py:294` is a
logging cap on an already-≤3 frame.

So Stage 6's "stop trading the bottom decile" is **already satisfied for momentum**. Whether
`meta_ranker` and `multi_ticker_swing_htf` have applied the same depth evidence is **not
measurable from the current logs** — only `dealer_ranker` (deprecated, n=21) produced
`contract_selection` records after 2026-09-08. Live activity also fell sharply in September
(127 momentum triggers vs 476 in August), which is consistent with the recorded WSL/live-ops
interruptions and means the post-change sample is simply too thin.

---

## Step 6 — the pullback trigger cannot be scored live yet

Momentum runs two triggers, roughly evenly: **`break_body_prev_high` 931** audit rows and
**`pullback_continuation` 864**. So ~48% of momentum's triggers are pullback-based.

Stage 5 measured a pullback entry rule at **−0.5877R** against enter-immediately on 434
backtested candidates, and every fixed delay at ≈0. That is a strong prior against the
pullback trigger.

**But it cannot be confirmed on the live ledger: 0 of 96 momentum closed trades could be joined
to a `trigger_rule`.** The rule is logged on the *decision* and never carried onto the
resulting order or closed trade, so outcomes cannot be attributed to triggers at all.

**That is the finding, and it has a one-line fix:** put `trigger_rule` into
`build_equity_order_audit` / `build_option_order_audit` (and hence the closed-trade record), the
same way `iv`/`greeks` were added on 2026-09-22. Until then the Stage 5 result stays a backtest
prior rather than a live measurement — and given it concerns ~half of momentum's entries, making
it measurable is worth more than acting on it blind.
