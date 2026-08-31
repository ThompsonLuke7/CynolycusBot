# 2026-08-27 Callout and Architecture Audit

## Decision

Do **not** add another setup detector, trade-plan object, level engine, or lifecycle
state machine. CynolycusBot already has those in `strategies/intraday_structure`.
The actionable gap is earlier **candidate discovery**: opening gap/RVOL leaders and
validated catalysts do not currently feed that engine. As a result, the engine can
confirm a name that reaches it, but names such as CRM, OKTA, and PTEN were absent or
arrived after the move.

Keep Intraday Structure paper-only until the frozen forward evaluation has enough
data. Its 1,159 historical confirmed setups were approximately flat before costs,
so it is not yet evidence-backed as an execution gate. The option-expression seam
and live tape/OPRA inputs should remain gated behind that validation and fit-for-
purpose NBBO data.

## Evidence boundary

The source screenshots do not expose reliable post times in the pasted text; their
dates were inferred. Therefore this report answers two narrower, defensible
questions:

1. Did a module rank, confirm, or trade each name in the available live artifacts?
2. What happened in the local 2026-08-27 session if that was the intended date?

It does **not** validate the claim that every callout hit. Several inferred 08-27
contracts are inconsistent with that day's underlying range (for example AMZN
never approached 270, NBIS never approached 245, and MRVL fell 4.7%). A proper
hit-rate study needs every callout, including losers and edits/deletions, with the
original posting time, entry, stop, targets, expiry, and contract. Options P&L also
needs contemporaneous bid/ask marks; stale trade prints are not acceptable.

## Current flow and the missing seam

The 4-hour Momentum, HTF, and Meta modules and the 30-minute Swing module each rank
and execute independently. Their audit outputs, dealer ranks, liquidity list, and a
manual list separately feed Intraday Structure. Intraday Structure then computes
levels and progresses a deterministic lifecycle:

`WATCHING -> SETUP_DETECTED -> ARMED -> CONFIRMED -> RUNNING -> TARGET_REACHED ->
EXTENDED -> EXHAUSTED / INVALIDATED -> CLOSED`

That branch is observational and paper-only. It neither gates nor triggers the
orders submitted by the upstream modules. News is polled into the catalyst ledger,
but Intraday Structure does not consume it; the 30-minute scanner uses news only as
an expected-value tilt after its base model has already emitted a signal.

What already exists from the proposed analysis:

- setup objects and explicit lifecycle states;
- prior-day, premarket, opening-range, VWAP, pivot, and dealer-derived levels;
- trigger, invalidation, target ladder, risk/reward, and premarket AVOID plans;
- RVOL, range expansion, relative strength versus SPY/QQQ/sector, and retest/hold
  confirmation;
- closed-setup and abstention ledgers plus forward 1-minute archive code;
- deterministic option selection infrastructure, though not driven by setup
  geometry;
- a live-flow provider abstraction for options tape metrics, though no validated
  OPRA/NBBO source is wired into the runner.

What is missing:

- a market-wide premarket/opening **gap + RVOL + displacement + relative-strength
  acceleration** candidate feed;
- a relevance-validated catalyst candidate feed (the current ledger has false
  positives such as ordinary uses of the word “now” being mapped to NOW);
- an explicit candidate-capacity policy so new event/opening candidates reserve
  slots rather than silently displacing the current 130-name set;
- a common intent/execution ledger joining rank, setup transitions, policy advice,
  order, fill, ownership, and fixed-horizon MFE/MAE;
- validated live NBBO/OPRA flow, if tape confirmation is later shown to add value.

## Ticker audit

“Caught” means evidence exists in a ranker, setup transition, or execution ledger;
it does not mean the trade was profitable or matched the callout. Session moves are
open-to-close using the local 2026-08-27 daily bars.

| Symbol | 08-27 move | What CynolycusBot did | Verdict |
|---|---:|---|---|
| AMZN | -1.15% | No 4H/30m signal. Intraday Structure confirmed two opening longs; both invalidated, about -0.19% modeled net each. | Saw it, wrong twice; inferred callout date likely wrong. |
| APA | +2.79% | Eligible in 30m Tier 2, but no signal, 4H rank, or intraday candidate. | Missed at candidate/rank stage. |
| ASST | +2.98% | HTF entered near 12.11 in July; exits on 08-20 and 08-26 realized roughly +$815 total. | Caught early and traded well. |
| CRM | +9.56% after +11.88% gap | No 4H rank. The 30m model emitted a late short; long-only policy blocked it. Intraday dealer candidate arrived after the session. | Clear discovery and direction miss. |
| CRWD | +9.46% after +10.08% gap | Fresh HTF Sep-18 187.5C from 08-25; one contract sold 08-27 at +236%, remainder marked about +261%. Earlier CRWD attempts had large stop-outs. | Current breakout caught very well; path remains volatile. |
| FSLR | -1.33% after +3.40% gap | Tier 3 is excluded from the live 30m universe; no 4H/intraday signal. | Not captured, but 08-27 was a gap fade, not a sustained uptrend. |
| GOOG | +0.42% | 30m long confirmed and bought a Sep-18 342.5C; EOD option mark was about -14.8%. Dealer policy advised veto and option translation advised stock-only, but paper enforcement was disabled. Intraday Structure also logged two losing longs. | Traded, but not correctly; policy-advice/enforcement mismatch. |
| HOOD | -0.85% | No recent top-ranker, 30m signal, or intraday candidate despite +15.4% over five sessions. | Missed the slower multi-session trend; not an 08-27 momentum win. |
| HUT | -2.59% after +5.24% gap | Existing HTF equity from 08-04 was about -13.3% unrealized. | Seen historically; did not capture the inferred session callout. |
| IBRX | +1.94% | Absent from the active rank/audit evidence; headline mention only. | Not captured. |
| MRVL | -4.73% | Prior Meta option closed 08-21 at about +145% / +$6,280. Intraday Structure's 08-27 long invalidated. | Earlier trend captured; inferred 08-27 bullish callout does not fit the tape. |
| MSFT | +2.06% | Intraday Structure armed a 09:37 VWAP reclaim but never confirmed; repeated invalidation-width abstentions. No 4H/30m trade. | Setup seen, trend missed by risk/confirmation. |
| MU | -3.27% after +3.05% gap | Existing Momentum equity was about -6.6%. A 30m long emitted but expired unconfirmed; Intraday Structure also withheld confirmation. | Correctly avoided adding on 08-27, though old position was weak. |
| NBIS | -2.19% | Existing Meta equity had realized a +39.5% trim and retained gains; separate Momentum/HTF options had three stop-outs. No fresh 08-27 confirmation. | Trend found, option expression/re-entry poor. |
| NOW | +6.09% | No 4H/30m rank. Intraday Structure confirmed a VWAP reclaim, reached target, extended, then invalidated; modeled net was only +0.03%. It could not submit an order. | Structure engine found it but failed to monetize/retain MFE. |
| OKTA | +4.07% after +23.61% gap | No 4H/30m long. Intraday candidate appeared near the close/after-hours; no confirmation. Relevant news also arrived after the open. | Clear no-candidate-at-open / candidate-late miss. |
| PTEN | +5.51% | Tier 3 excluded from 30m; no 4H or intraday candidate. | Clear active-universe/discovery miss. |
| SLS | +8.04% | Momentum ranked it #3 on 08-27; a fresh HTF Sep-18 15C was marked +26.5%. Two earlier options stopped for large losses, and an orphan equity position remained at the broker. | Direction caught, but stop/re-entry and ownership controls need work. |
| SMCI | -0.59% | Earlier Dealer option realized about +192%; Meta and HTF positions retained gains. Intraday premarket breakout plan failed confirmation at the open. | Broader trend captured; correct no-add on 08-27. |
| SMR | +2.53% | Prior HTF trade closed +17.8%; re-entry on 08-26 was +4.2% unrealized at 08-27 EOD. | Caught and traded. |
| TEM | +2.97% | Older Meta equity retained about +44% unrealized. New Momentum Sep-18 75C entered 08-27 and ended about -9.3%. | Trend caught early; fresh option entry was late. |
| SPY | — | Daytrader bought an 08-28 770C and exited one minute later for a verified -$10 after setup failure. | Traded direction, not the cited contract, and not successfully. |
| QQQ | — | Used as context; no direct 08-27 trade. | Not captured as an instrument. |
| BTC / VIX | — | Context/proxy inputs only; no direct execution. | Not applicable to current equity/options execution flow. |

## Recommended changes

### 1. Add event/opening candidate discovery — paper-only first

Feed the existing Intraday Structure engine from three additional, timestamp-safe
sources:

- premarket and first-30-minute gap/RVOL/range/displacement leaders;
- relative-strength and acceleration leaders versus market and sector;
- validated high-impact catalysts whose symbol relevance, event type, source, and
  availability time pass strict checks.

Reserve a defined portion of the 130 candidate slots for these sources, deduplicate
by symbol, persist source priority and observation time, and measure discovery
recall separately from setup-confirmation precision. This targets the actual CRM,
OKTA, PTEN, and APA failure class without widening execution eligibility.

### 2. Build a timestamped callout benchmark

Once original callout timestamps are supplied, join each callout to:

- candidate presence/source and rank at callout time;
- setup state, trigger, invalidation, target, and abstention reason;
- policy advice versus enforced action;
- order intent, broker fill, position owner, and exit;
- underlying MFE/MAE at fixed horizons and, only with NBBO, contract MFE/MAE.

Use all callouts, not screenshots selected after the outcome. Benchmark the
underlying setup first so contract liquidity and stale prints cannot masquerade as
signal quality.

### 3. Improve observability and operational correctness

- Restart/verify the newly wired candidate-scoped 1-minute archive and require a
  session coverage manifest; there is no recoverable 08-27 1m archive because the
  code was added after that session/server start.
- Label policy recommendations that are intentionally ignored in paper mode. The
  GOOG option is the clearest example.
- Reconcile duplicated module ownership and broker orphans before attributing P&L;
  SLS shows why module ledgers alone are insufficient.
- Collapse repeated per-bar abstentions into setup episodes for analysis; hundreds
  of MSFT rows do not represent hundreds of independent opportunities.

### 4. Keep these items gated

- Do not make Intraday Structure an order gate until 6–8 weeks of post-fix forward
  ledger data and the preregistered ablation show incremental value.
- Do not broaden the frozen 30m execution universe or top-k solely from these
  examples. Publish a wider discovery shortlist into the paper structure engine
  and measure it first.
- Do not wire options tape/flow until the actual entitlement and feed are validated
  for NBBO accuracy, causal timestamps, and derivative/underlying return sanity.
- After structure earns promotion, pass target distance, expected holding time,
  invalidation, and arrival probability into option selection. Do not copy short-
  dated Discord contracts without an independently validated payoff study.

## Bottom line

The bot already captured ASST, CRWD, MRVL, NBIS, SLS, SMCI, SMR, and TEM in some
form, with ASST, the current CRWD trade, MRVL, SMCI, and SMR providing the strongest
positive evidence. NOW proves the intraday state machine can identify a clean
continuation, but it is not connected to execution and gave back most of the move.
CRM, OKTA, and PTEN are the clearest examples of the real architecture gap: the
name was not promoted into the 1-minute decision loop at the open. Solve candidate
discovery and measurement first; do not add another overlapping signal engine.
