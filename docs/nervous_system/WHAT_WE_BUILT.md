# The nervous system: what we built, and why

A walkthrough of roughly four weeks of work across 27 tasks. Written to be read
start to finish by someone who wasn't in the room.

**Status: built, not deployed.** Meta Ranker routes through it; the other five
strategies do not. Nothing has run in production. Whether it improves trading
results is unknown and is not the claim being made here.

---

## 1. The problem it solves

Before this, each strategy talked to the broker directly. Meta decided
something, called `submit_order`, and that was the whole story. Five other
modules did the same thing, each in its own way.

That worked, and it hid five specific problems:

**Nobody could reconstruct a decision.** When a position appeared, the record
was a log line. What the market looked like, which model version scored it, what
the ranking was, why *that* strike — gone. The 2026-06-26 OOM killed the server
mid-session and the answer to "what did we own and why" took hours to piece back
together from JSONL.

**A retry could double a position.** Order identity came from the broker, so a
crash between "sent" and "recorded" left no safe way to resume. The only safe
option was to do nothing and reconcile by hand.

**Risk limits lived inside strategies.** Each module enforced its own sizing.
Nothing could answer "how much are we exposed to semiconductors right now,
across everything."

**Research and live drifted.** A backtest computed features one way and the live
runner another. Nothing checked, so nothing caught it.

**Bad data reached conclusions before anyone tested the data.** In July an entire
options-routing study was retracted: the option "prices" were stale trade prints,
and the correlation between option returns and the underlying was **+0.09** where
a long call should be near +0.9. Two error corrections along the way changed the
magnitudes but not the signs, which made the wrong answer look robust.

That last one shaped more of this design than anything else.

---

## 2. The shape of the answer

One path from signal to broker, with each step recorded:

```
strategy ranking
      ↓
context snapshot     what was knowable at the decision time
      ↓
trade intent         what the strategy wants, in shared vocabulary
      ↓
policy decision      what risk permits, with every modifier recorded
      ↓
order request        content-addressed, idempotent
      ↓
journal              written BEFORE the broker call
      ↓
broker adapter       the only thing that may POST
      ↓
execution events     what actually happened
      ↓
outcome              append-only, revised but never rewritten
```

Every arrow is a contract with a content hash. Any decision can be replayed
from its record, and the replay is checked against the live path by hash rather
than by eye.

---

## 3. What was built, and the reason for each piece

### Contracts and state (Tasks 1–8)

Frozen, validated types for market/theme/ticker/dealer state, intents, policy
decisions, orders, and executions. All timestamps timezone-aware; all money in
`Decimal`, never float.

**Why:** a float dollar amount is wrong eventually and silently. A naive
timestamp has no defined instant, so it cannot be ordered against a decision.

The rule that does the most work: **`as_of` (event time) is never
`available_at` (when we could know it).** A bar stamped 16:00 that landed at
16:07 was not knowable at 16:03. Selecting on event time makes a backtest
quietly outperform reality.

### Historical import (Tasks 9–14)

Reads existing JSONL/parquet artifacts into the state registry, by reference and
hash rather than by copying rows. Missing causal availability is quarantined,
never inferred from file mtime — mtime records when bytes were written, not when
the information became available.

### Policy engine (Task 15)

A pure function: intent + snapshot + config → decision. No clock, no IO, no
environment. Enforced by tests that parse the source and fail if it imports a
network library or reads a clock.

**Why pure:** a policy that reads a clock cannot be replayed. Same inputs must
give the same answer tomorrow, or the audit record proves nothing.

Every modifier is recorded — what rule, what it did to the budget, and why.
Multipliers can only *reduce* risk; that's validated at config load, so a typo
cannot increase position size.

### Portfolio and exposure (Task 16)

Sector and factor exposure with explicit unknowns. An unmapped sector is
`UNALLOCATED`, never silently bucketed.

### Options (Tasks 17–18)

Payoff analysis with an **analytic** upper-tail slope — unbounded loss is proven,
not sampled. Fourteen structure families, and a selector that respects the
strategy's preference order rather than silently upgrading equity to options.

Caught here: covered calls were being rejected as unbounded because the option
leg was analysed without its stock. And `build_structure` multiplied ratio by
quantity, overstating max profit by 3.86×.

### Execution (Tasks 19–21)

The gateway writes a journal entry **before** the broker call, never
auto-retries a POST, and resolves ambiguity by looking up the client order ID.
Order identity is content-derived, so a retried pass converges on the same order
instead of placing a second.

**A real bug this caught:** order timestamps initially came from the wall clock,
so a retried 4H pass produced a different content hash, a different client order
ID, and therefore a **duplicate order**. Timestamps are now anchored to the
decision bar.

Risk-reducing exits are deliberately **fail-operational**: a close is never
blocked by a missing quote, and an unreadable market calendar defers entries but
never blocks an exit. Trapping a position is worse than the problem being
avoided.

### Orchestration (Task 22)

Durable job leases with fencing tokens, a transactional outbox, and a
coordinator that commits snapshot + intent + policy + order + event in one
transaction — with the broker never called inside it. A broker call holding a
transaction open would block every other writer, and a rollback would erase the
record that the call happened.

### The Meta cutover (Task 23)

Meta's live runner now has **zero** direct broker calls.

The shared 4H engine is used by four modules, so its direct calls were replaced
by **injection** rather than rewritten: Meta passes a governed submitter, and
everyone else keeps today's behaviour. One module's migration cannot break
another's.

Also fixed here, both pre-existing: an unreadable market calendar was
**submitting** entries rather than deferring them, and the pending-entry queue
was deleted unconditionally after every flush, losing decisions that were merely
blocked by a readiness gate.

### Replay and attribution (Task 24)

The source-fitness gate runs **before** any P&L and defaults to unfit. Trade
prints, last prices, synthetic, forward-filled, interpolated — and midpoints —
are never fit marks. Thresholds are enforced per option side, because averaging
a broken put series against a healthy call series hides the break.

The +0.09 correlation from the retracted study is a test case. If that data
comes back, this refuses it.

The evidence provider is the anti-leakage boundary. Causal rules are advisory as
long as strategy code can reach past them — a function you have to remember to
call is not a guarantee. So the corpus is private, there is no accessor that
returns all of it, and each evidence type has its own scoped method.

Attribution splits a result into underlying movement, reference-to-fill
slippage, fees, and instrument transformation. The parts sum to the whole
exactly; transformation is the residual and is labelled as one rather than
dressed up as an independent measurement.

### Audit surface (Task 25)

A GET-only HTTP handler. No mutating verb is served, no route can reach a
gateway or broker, reading alerts does not acknowledge them, and redaction is
recursive because secrets do not stay at the top level.

Alerts are a **deduplicated projection** over an immutable event log — a hundred
rows for one stuck order is noise, not information — with the history kept
underneath, because when each occurrence happened is what an incident
reconstruction needs.

### Cloud runtime and acceptance (Tasks 26–27)

Settings for Cloud SQL, GCS journal, and Secret Manager, with **two DSN shapes**
because Cloud Run uses a socket and the VM will use TCP. A non-destructive
database CLI with no downgrade, drop, or reset — a destructive command that
exists eventually gets run against the wrong target at the wrong hour.

The acceptance suite proves what holds and the acceptance document leads with
what does not.

---

## 4. How it was built

Every task followed the same loop: write failing tests first, watch them fail,
implement, then **mutation-test** — deliberately break each safety rule and
confirm a test catches it.

Roughly **190 mutations** were run. Most were caught. The ones that survived were
the valuable part, because a surviving mutation means a rule nobody was actually
testing:

- A 500-share "exit" against a 100-share position reached the broker while every
  test passed — the guard existed but was never wired into `submit`.
- Option slippage was untested against the contract multiplier, so every option
  execution cost would have read **a hundred times too small**.
- Two builder guards passed for the wrong reason: the contract raised a
  similar-looking message, so the test never exercised the guard it named.
- The acceptance document's "NOT PROVEN" section could be deleted entirely and
  the test still passed, because the phrase also appeared in the introduction.

Several near-misses were caught mid-change and reverted: renaming an endpoint
would have severed Momentum and Dealer Ranker, and a global string ban would
have changed four dashboards nobody asked me to touch.

---

## 5. What this does not do

- **It does not make the strategies smarter.** No signal changed. This is
  plumbing, governance, and record-keeping.
- **It does not connect modules to each other yet.** The exposure engine can see
  across strategies, but only one strategy is on the spine, so it sees one.
- **It is not proven to help.** No soak has run. The claim is that decisions are
  now recorded, reproducible, and bounded — not that they are better.
- **It is not repository-wide.** Five strategies still submit directly. Their
  bypasses are inventoried with pinned counts so a new one cannot appear
  unnoticed.

---

## 6. Numbers

| | |
|---|---|
| Tasks | 27 |
| Nervous-system tests | ~1,290 |
| Database tables | 29, across 4 migrations |
| Mutations run | ~190 |
| Meta direct broker calls | 0 (was 8) |
| Legacy direct calls remaining | 21, inventoried |
| Real-money code paths | 0, by construction |

---

## 7. What comes next

1. Merge, and configure the environment before the next Meta session.
2. Run locally against Docker PostgreSQL — no GCP required for this.
3. Watch a few sessions with submission off, then turn on paper submission.
4. Provision Cloud SQL and the journal bucket for durability.
5. Move the server to a Compute Engine VM (the WebSocket cannot live on Cloud
   Run), and **turn the local order path off** — two servers submitting the same
   signal is the worst available outcome.
6. Serve the read-only dashboard from Cloud Run.
7. Migrate the other five modules, one at a time. The expensive part is shared
   and already built.
