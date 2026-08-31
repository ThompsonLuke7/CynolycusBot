# Stage 0 — feasibility. Result: **GO**, with one live-system finding

Run 2026-08-28. Read-only. Scripts: `scripts/execution_quality/{pull_order_history,stage0_feasibility,stage0_bar_coverage}.py`

All four Stage-0 blockers from the design are resolved. One of them turned into a
finding about the live system rather than about the study.

---

## A. Entry fill timestamps — RECOVERABLE (95.1%)

Alpaca retains full order history with `submitted_at`, `filled_at`,
`filled_avg_price`, `filled_qty`. Pulled 2,067 orders (958 filled, 1,072 canceled
— the canceled ones are unfilled ladder rungs, as designed).

Matching each closed-trade ledger row to its broker entry order(s), grouping
ladder rungs inside a 30-minute window and comparing the rung-weighted VWAP to
the ledger's `entry_avg_price`:

| module | rows | entry fill found | price agrees | no fill |
|---|---|---|---|---|
| momentum_expansion | 34 | 34 (100.0%) | 34 | 0 |
| multi_ticker_swing_htf | 61 | 58 (95.1%) | 58 | 3 |
| meta_ranker | 48 | 43 (89.6%) | 43 | 5 |
| dealer_ranker | 40 | 39 (97.5%) | 39 | 1 |
| **pooled** | **183** | **174 (95.1%)** | **174 (95.1%)** | 9 |

**Every single matched row agrees on price** (174/174, within 2c or 2%). The
match is not a guess — the independently-recorded ledger price confirms it.

The 9 misses are all pre-`2026-07-13`, which is the broker's retention floor
(asked for June, earliest order returned is 07-13). Meta and HTF ledgers start
07-09, so four sessions of history are simply gone. Accepted, not worked around.

**`T_submit` and `T_fill` now exist for 174 of 183 rows.** The entry-timing arm is live.

## B. Null `entry_bar` — EXPLAINED, benign, but it changes the unit of analysis

45 rows carry `entry_bar: null`. The cause is exact and not corruption:

```
(exit_reason, exit_qty < entry_qty, entry_bar is null) -> n
  ('take',    True,  True )  45     <- every null row
  ('take',    True,  False)  11
  ('stop',    False, False)  61
  ('horizon', False, False)  32
  ... zero nulls anywhere else
```

All 45 are **partial take-profit trims**. In [live_4h_exec.py:1038](core/live_4h_exec.py#L1038)
`es` is read from `exit_context`, which by construction only holds *full* exits
(`is_full_exit = bool(exit_context and sym in exit_context)`), so a trim passes
`entry_state=None` and the entry lineage never reaches the row. Harmless to
trading; it just means the ledger row lacks its own provenance.

**Consequence for the study:** 56 of 183 rows are scale-outs, not closes. A
"trade" must therefore be a *position lifecycle* — one entry group, one or more
exits — not a ledger row. Stage 1 reconstructs lifecycles from the broker stream,
which also gives the trim structure for free.

## C. 1-minute tape — SIP is dense, IEX is not, and **the live system runs on IEX**

20 traded names, 2026-08-18 → 08-26, RTH minutes present per session (390 = complete):

```
ticker   IEX/sess  SIP/sess     ticker   IEX/sess  SIP/sess
NBIS      363.0     390.0       CDE       383.7     390.0
SLS       307.2     389.5       TER       300.2     389.5
CRDO      253.7     387.0       ALAB      211.3     386.3
CRWD      379.3     390.0       OKLO      318.7     390.0
MRNA      357.8     389.2       IONQ      381.2     390.0
FIG       374.3     390.0       ASX       331.3     389.7
PURR      370.0     389.3       APTV      272.8     386.2
P         244.7     388.3       ARMK      228.3     377.2
VIAV      255.5     387.5       GMAB      278.5     377.2
FBRX       93.2     258.7       RBRK      226.3     386.2
```

| feed | median gap | mean gap | worst | names with >20% missing |
|---|---|---|---|---|
| IEX | **22.1%** | 24.0% | 76.1% | **11 of 20** |
| SIP | 0.2% | 2.3% | 33.7% | 1 of 20 (FBRX) |

**SIP historical is entitled and returns a near-complete tape.** The study will
use SIP, and FBRX-class names (<300 SIP bars/session) get flagged rather than
silently averaged in.

**Correction, found in Stage 3 and folded back here.** SIP on this subscription
is *delayed*, not live. Querying a window ending inside the last ~15 minutes
returns `subscription does not permit querying recent SIP data`; measured
boundary: `now-16m` succeeds, `now-5m` fails. So the entitlement is
"SIP historical, 15-minute delay" — fine for this study, and the reason the
live-path note below has to be reworded.

### The finding that is not about the study

[`UI/shared_stream.py:119`](UI/shared_stream.py#L119) hardcodes `feed=DataFeed.IEX`, and
`.env` sets `APCA_API_DATA_FEED=iex`. The shared bar stream feeds the intraday
structure engine, the dashboards, and the live intraday decision path.

So **every live intraday decision is being made on a tape missing a median 22%
of RTH minutes on the names actually traded.** A pullback low, an invalidation
touch, an EMA reclaim, or a pivot hold evaluated on a tape with a fifth of its
minutes absent is not the same event as the one the rules were written to detect
— and the intraday engine's median hold is 2 minutes, which is 1–3 bars. This is
a plausible mechanical cause of the MAE > MFE pattern, and it is testable
directly in Stage 4C by replaying the same setups against the SIP tape.

**This is not a config oversight and cannot be fixed by flipping the feed.**
Real-time SIP is not in this subscription (see the correction above), so
`shared_stream.py` streaming IEX is a constraint, not a mistake — as its own
comment about the IEX one-concurrent-stream limit already hints. The actionable
question is therefore a *plan* question: is real-time consolidated data worth
paying for? Stage 4C answers it by quantifying what the missing minutes cost on
setups we already took, which is the number that decision needs.

---

## Gate decision

| blocker | status |
|---|---|
| entry fill timestamps | **resolved** — 95.1%, price-confirmed |
| null `entry_bar` | **explained** — trims; unit of analysis changes to lifecycle |
| 1m coverage | **resolved** — use SIP, not IEX |
| Meta P&L 23/48 | carried forward — those rows are excluded from P&L, kept for timing |

Full-precision arm proceeds on 174 lifecycles. No degradation to bar resolution
is needed. Stage 1 next.
