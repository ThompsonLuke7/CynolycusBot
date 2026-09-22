# The connective layer: exposure, peers, and the decision packet

`WHAT_WE_BUILT.md` closed with two honest limitations:

> "It does not connect modules to each other yet. The exposure engine can see
> across strategies, but only one strategy is on the spine, so it sees one."

This document records what changed, and what still has not.

---

## 1. The two blockers that were actually there

An audit found the gap was sharper than "only Meta is migrated".

**The exposure engine was never called.** `portfolio/exposure.py::calculate_exposure`
was 353 lines, fully tested, and had **no caller outside its own tests**.
`policy/engine.py` never imported it, there were no reason codes for
sector/theme/factor concentration, and `portfolio_limit_vetoes` enforced only
`max_daily_loss` and `max_gross_notional`. So `max_symbol_notional`,
`max_sector_notional`, `max_theme_notional` and `max_factor_notional` were
configured at real dollar figures and enforced by nothing.

**Theme exposure was structurally uncomputable.** `_allocate_theme` bucketed
every position whose underlying was not the snapshot's ticker into
`UNALLOCATED`, because `ContextSnapshot` is single-ticker by construction:
`_validate_ticker_scope` *raises* on any ticker-scoped state that does not match.
A theme limit could therefore only ever see one position — the one being
decided. Sector and factor never had this problem; they come from portfolio-wide
config maps.

The single-ticker invariant is load-bearing (it is what makes a snapshot a
statement about one decision, and Meta's replay parity depends on its hash), so
it was **not** relaxed.

---

## 2. Peers: an additive second field

`ContextSnapshot.peer_theme_memberships` carries memberships for underlyings
*other* than the decision's ticker.

* Peers get **every** causal gate the context ticker's own membership gets —
  `available_at <= decision_time`, validity, and the effective window — via
  `_validate_peer_causality`. Eligibility reuses `_candidate_rejection_reason`
  unchanged, so a peer is held to exactly the same bar.
* Peers are **hashed** as real embedded states (they appear in `state_ids` and
  `state_hashes`), so a changed peer set changes the snapshot hash.
* Peers are **excluded** from the decision's data-quality and version
  accounting. A stale membership on an unrelated holding must not raise
  `DATA_QUALITY_BLOCKING` and veto this ticker.
* Peers are **advisory**: a missing or stale peer is simply not selected and
  never marks the snapshot invalid. Exposure reports the position as
  `UNALLOCATED`, which is the honest answer.
* With no peers requested, the snapshot — including its content hash — is
  **byte-identical** to what the same inputs produced before. That is a pinned
  test, because Meta's replay parity depends on it.

The peer set is resolved from the registry's own PORTFOLIO state
(`SnapshotBuilder.held_underlyings`), not from the broker, so the same answer
replays later.

---

## 3. The concentration veto

`policy.rule.concentration` vetoes an entry that would breach a symbol, sector,
theme, or factor limit, counting every open position.

* **Off by default.** It runs only when `PolicyConfig.portfolio_exposure` is
  set, which `build_router` populates only when
  `CYNOLYCUS_CONCENTRATION_LIMITS=true`. Turning it on changes live behaviour:
  names that used to fill will be refused. It is a deliberate switch, after a
  shadow session — not a side effect of deploying this code.
* **Never blocks an exit.** `applies_to_risk_reducing=False`. Trapping an open
  position is worse than the concentration it would avoid.
* **`UNALLOCATED` never vetoes.** An unmapped position means "we do not know
  this position's theme". A limit on the unknown bucket would block entries for
  a mapping gap rather than for real concentration.
* It calls `evaluate_proposed_concentration`, a pure function that adds the
  proposal as *notional* to the buckets its underlying maps to rather than
  fabricating a `PortfolioPosition` — a synthetic position would need an
  invented quantity and price, and the limits are stated in notional anyway.

---

## 4. Peer groups: the lateral axis

`PeerGroupState` (`StateType.PEER_GROUP`) records which tickers move together
and **how that grouping was derived**. It is produced by
`signals/peer_structure/correlation_groups.py` from trailing correlation, not
from the LLM theme taxonomy.

That choice is measured, not aesthetic. From
`research/execution_quality/24_horizon_thesis_experiments.md` §7, forward
20-session mean pairwise correlation:

| grouping | forward cohesion |
|---|---|
| theme groups | 0.388 |
| **trailing-correlation clusters** | **0.343** |
| sector (size-matched) | 0.233 |
| size-matched random | 0.217 |

Correlation clustering buys ~73% of the taxonomy's edge over random with no
news, no embeddings and no LLM — and without the taxonomy's **83–88% weekly
membership churn** (co-membership Jaccard 0.12–0.14, §4). For a *risk bucket*,
stability is the property that matters: a limit computed on a grouping that
reshuffles weekly is not a limit.

Build discipline, all of it load-bearing:

* bars are cut to `<= as_of_session` **before** anything is computed;
* the universe comes from `load_universe_as_of`, which raises rather than
  falling back to today's list (that fallback is the survivorship bias it
  exists to remove);
* `suspect_sessions` blanks corporate-action gaps — the 1d cache is unadjusted,
  and one 4x split print dominates every correlation the name appears in;
* a $10M/day and $5 floor is applied at build time, because §8 found 87% of the
  leader/follower events were in names that fail it and those names carried the
  entire measured effect.

### Are the groups just a beta tilt?

Partly, and that had to be measured rather than assumed. Forward-20d mean
pairwise correlation for each group against size-matched controls drawn from
the same screened pool (`scripts/peer_group_controls.py`, 3 as-of dates, 60
groups each):

| | forward-20d cohesion |
|---|---|
| **correlation groups** | **0.3212** |
| beta-decile matched | 0.1583 |
| random | 0.0612 |

So of the +0.260 total excess over random, **roughly 37% is a beta tilt**
(beta-matching alone lifts 0.061 → 0.158) and **~63% is the grouping itself**
(0.158 → 0.321). The groups beat both controls at all three dates, same sign
every time.

**The sector arm is retracted.** An earlier run reported a sector control of
0.0804. It was meaningless: the shared universe snapshot's `sector` column is
68% `NaN` and 32% the literal string `"Unknown"` — two non-values — so the
"sector-matched" draw was just an arbitrary bucket. The canonical
`SECTOR_MAP` covers 95/2903 names (3.3%) and the empirical assignments parquet
does not exist, so **no sector control is available on this universe today**.
The script now refuses to emit the arm rather than print a number that looks
like evidence.

Caveats that apply to all of it: membership is the 2026-09-11 snapshot
(survivorship — both arms draw from the same pool so the *contrast* is fair,
the *levels* are not); three as-of dates; no confidence intervals. Directional,
not a significance test.

**What it is not for: ranking.** The theme block fed to the Meta ranker measured
**−0.0167 rho** [−0.0222, −0.0111] (§4). Nothing here should reach a scoring
feature matrix without its own experiment.

---

## 5. The decision packet

`core/nervous_system/context/packet.py` renders one snapshot as structured,
nested evidence (and as flat text), served read-only at
`GET /audit/snapshots/packet?id=<snapshot_id>`.

It is derived, never persisted — computed from the stored snapshot every time,
so it cannot drift from the evidence it describes. It has no IO, no clock and
no model, so an archived snapshot renders identically years later.

It is deliberately **not** a feature source. Nothing in the scoring path should
import it.

---

## 6. Momentum on the spine

Momentum now routes through the same governed path, behind `--governed`
(**off by default**).

The shared bookkeeping was *extracted*, not copied, into
`core/governed_plan_execution.py`. Every branch in it was paid for by a real
incident, and a duplicated copy is how two exit paths drift — a drifted exit
path loses positions. Meta's own `_submit_via_gateway` still has its original
copy; folding it onto the shared helper is a follow-up, and deliberately not
done in the same change that put a second module on the path.

What is genuinely momentum-specific:

* **The score vocabulary.** `MetaIntentConfig` gained `primary_score` and
  `score_fields`; momentum uses `expansion_score` / `rank`, Meta keeps
  `s_combo` / `s_upside` / `s_quality` byte for byte.
* **A name collision worth knowing.** `expansion_score` is *also* the name of a
  forward label in momentum's training matrix — the one that, with
  `expansion_target` and `trend_persistence`, gave the parabolic filter a fake
  AUC of 0.951. In the live ranking frame it is the ranker's *output* at the
  decision bar. Same name, opposite direction in time.
* **Its own snapshot profile** (`momentum_4h_1420@1`), same rules as Meta's so
  the state store is described once, different id so snapshots stay
  attributable and one module's profile can be tightened alone.
* **Its own TICKER state publication.** Meta publishes only its top-K plus what
  it manages, so a momentum name outside that set has no state and the
  governed path refuses a decision it cannot evidence. Two producers for one
  entity is intended: the state describes the ticker, not the strategy, and
  `_stable_state_id` keys on ticker/bar/lineage/content, so identical content
  converges on one row.

**No direct-broker fallback.** An unreachable governed path queues the plan
through `defer_entries_if_market_closed` and `defer_exits_if_opg_unavailable`,
both forced — during market hours the calendar check would otherwise leave an
exit in a plan that is never submitted while the position has already been
dropped from managed state. Falling back to a direct call would reintroduce the
exact bypass the cutover removes.

### Tracking both modules

With both on the spine, every decision lands in the same tables keyed by
`strategy_id`, so the two become comparable in one place rather than through
two JSONL dialects:

```
GET /audit/decisions?strategy_id=momentum_expansion
GET /audit/decisions?strategy_id=meta_ranker
GET /audit/decisions/detail?id=<decision_record_id>
GET /audit/snapshots/packet?id=<snapshot_id>
```

Run momentum governed with:

```
.venv/bin/python -m strategies.momentum_expansion.live.runner --submit --governed
```

`--governed --live` is refused before a client is built; the router refuses
`PRODUCTION_LIVE` in its constructor regardless.

---

## 7. What has NOT changed
* **No signal changed.** No model, label, feature or score was touched.
* **Momentum's default path is still direct.** `--governed` is opt-in per
  process, because two paths submitting the same signal is the worst outcome
  available.
* **Concentration gating is off.** Nothing refuses an order today that did not
  refuse one before.
* **Peer groups are not consumed by any live decision.** They are published
  state and a nightly artifact.
* **Nothing here is proven to improve trading results.** The claim is that
  cross-position exposure is now computable and enforceable, and that a
  decision's context is now readable. Not that either makes money.
