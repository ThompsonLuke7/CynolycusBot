# Stage 5 — entry timing: there is nothing to time at daily granularity

2026-09-23. Script: `05_entry_timing.py`. Output: `data/stage5_entry_timing.json`.
**Validation only; the test block is never read.**

## Setup

The thread's V1, ported from the design that already exists in
`strategies/spy_intraday/Policy/Execution_Agent/`: `A = {WAIT, ENTER, REJECT}` over a
5-session window after the signal, direction immutable (all long, inherited from the
incumbent score), and the reward is the `Execution_Agent/env.py:59-61` shape —
`R_agent − R_enter_immediately`, zero on WAIT — so REJECT carries a real opportunity cost and
"never trade" cannot win.

Candidates: incumbent `mom_score` top-3 per decision bar, deduplicated to **one position per
name per day** (the panel has two 4H bars per day and both map to the same next-session open;
without the dedupe one tradeable position appears twice). 2,430 candidates over 176 tickers,
12,150 entry-decision rows; 1,876 train / **434 validation** candidates.

Post-entry exit is deterministic — hold 30 sessions, Stage 3's best fixed policy — so this
isolates the entry decision alone. 1R is fixed at the **signal-day** ATR for every arm, so a
delayed entry cannot flatter itself with a smaller denominator.

---

## A. Deterministic timing rules

Enter-immediately baseline: **+2.0758R** on 434 candidates.

| rule | mean R | vs baseline | 95% CI |
|---|---|---|---|
| enter day 0 (baseline) | 2.0758 | 0.0000 | — |
| enter day 1 | 2.1299 | +0.0541 | [−0.052, +0.161] |
| enter day 2 | 2.1175 | +0.0417 | [−0.127, +0.204] |
| enter day 3 | 2.1327 | +0.0569 | [−0.159, +0.256] |
| **enter day 4** | 2.1471 | **+0.0713** | [−0.168, +0.299] |
| **first pullback, else reject** | 1.4881 | **−0.5877** | — |

**Every fixed delay is indistinguishable from entering immediately** — all four CIs straddle
zero, and the point estimates drift up by only ~0.07R across four sessions of waiting.

**"Wait for the pullback" is measurably wrong**: −0.588R. On momentum continuation names the
ones that pull back within five sessions are the ones that were not going to work, and the
rule also rejects the candidates that never pull back — which are the winners.

## B. The learned policy

| arm | mean R | vs baseline | 95% CI | reject rate | mean wait |
|---|---|---|---|---|---|
| learned | 2.0212 | **−0.0546** | [−0.170, +0.052] | 9.2% | 0.47 d |
| permuted-state twin | 1.7263 | −0.3495 | [−0.566, −0.154] | 25.6% | 1.05 d |

* **learned net of its permuted-state twin: +0.2949R** — so it is genuinely using the state,
  not noise. This is the one arm in the whole series that beats its own null.
* **learned net of the best deterministic rule: −0.1259R** — and it does not beat entering
  immediately either (−0.055R, CI straddling zero).

Read together those two lines say something specific and worth having: **the policy correctly
learned that the right action is "enter now".** Mean wait 0.47 sessions, reject rate 9.2%; the
noise twin by contrast dithers (wait 1.05, reject 25.6%) and pays −0.35R for it. The learned
agent's small loss comes from the residual times it deviates.

---

## C. Verdict

**Null, and a clean one.** There is no daily-granularity entry-timing edge on these
candidates: not in a learned policy, not in any fixed delay, and the popular pullback
heuristic is actively harmful. The pre-registered Stage 5 kill criterion ("cannot beat
enter-immediately out of sample after costs → stop") fires.

Two things this does establish:

1. **Enter immediately is the correct deterministic rule** for the 4H momentum book, and
   waiting for a pullback should be ruled out rather than left as folklore. That is a usable
   answer, not just an absence.
2. **The state is informative enough to recover the right action** (+0.295R over its noise
   twin), which is a different result from Stage 2's and Stage 3's arms, both of which lost to
   their nulls outright.

### C.1 The limit that names the next variant

**This is a DAILY-granularity test.** Decisions are taken at session opens over a 5-session
window. The thread's design — and the existing `Execution_Agent` — operates on **1-minute**
bars inside a 5–30 minute window, which is a different question this experiment cannot
answer: an intraday timing edge would be invisible at this resolution.

So the next testable variant is **resolution, not algorithm**: the same reward shape and the
same action space on 1m or 5m bars. The repo already has the 1m cache
(`research/execution_quality/data/bars_1m`) and the reference implementation. Note the
prerequisite that killed a neighbouring study: that cache spans 2026-07-08..08-28 only
(`25_matched_control_and_depth_null.md` §A.2 records a stale-bar bug from exactly this), so
the window would have to be extended before the test has any power — and the MDE should be
computed before the run, not after.
