# Study A — Findings

**Run:** 2026-08-26
**Registration:** `docs/superpowers/plans/2026-08-26-gamma-structure-preregistration.md`
**Generated tables:** `STUDY_A_RESULTS.md` (regenerate with `python -m research.gamma_levels.report_study_a`)
**Sample:** 4,445 arrival events, 2,186 resolved, 41 sessions, 5 ETFs (SPY QQQ IWM GLD SLV), 2026-06-12 → 2026-08-25

---

## The result

**Price rejects at the call wall 69.9% of the time versus 60.6% at an ordinary
strike — a gap of +9.3pp, 95% CI [+3.0, +15.3].** That clears the pre-registered
graduation threshold of +8pp with an interval excluding zero.

Removing the magnet entirely (see the confound below), walls versus plain strikes
gives **+8.2pp, CI [+1.5, +14.5]** — also over the line.

So the answer to "does price actually react to these levels" is **yes, at call
walls, and by a margin big enough to matter.**

## What did *not* survive

| Level | Gap | Verdict |
|---|---|---|
| **Call wall** | **+9.3pp [+3.0, +15.3]** | **Real** |
| Put wall | +1.0pp [-4.5, +6.4] | Nothing |
| Magnet | +1.5pp [-3.2, +6.1] | Nothing — and was an artifact before the fix |

The call/put asymmetry is the most interesting part of the result. It is
consistent with the mechanism rather than with a generic "levels matter" story:
call walls accumulate where customers sell upside (covered calls, overwriting),
leaving dealers long gamma there and hedging *against* continuation. Put-side
open interest is more often customer protection, which does not put dealers in
the same position. A generic support/resistance effect would have shown up on
both sides. It didn't.

## The methodological correction that changed the answer

The first run classified each strike using the snapshot *at the moment price
arrived*. That is circular, and the circularity was large:

- Gamma peaks at the money, so **the magnet is simply the strike nearest spot
  31–70% of the time** (SPY 70%).
- Even the call wall — max call gamma among strikes *above* spot — gets pulled
  onto the arrival strike once spot sits a hair below it.

So "price arrived at the magnet" partly encoded "price is where price is."

The fix: classify the strike from a snapshot **ten minutes before arrival**,
which is also the question a trader actually faces — *this strike was a wall
before I got here; does that change what happens next?*

Effect of the fix:

| | classified at arrival | classified 10 min before |
|---|---|---|
| magnet | +7.7pp ** | +1.5pp (gone) |
| call wall | +7.9pp ** | **+9.3pp ** ** |

The magnet effect was entirely an artifact. The call wall effect got *stronger*,
which is what a real effect does when you remove noise from its comparison.

## When does it work best

Exploratory — many comparisons on one sample, so these are hypotheses.

- **Levels that have held their strike ≥ median duration: +6.3pp [+1.2, +11.4]**
- **Levels in the top quartile of gamma concentration: +7.1pp [+1.5, +13.2]**

Both point the same way and both were pre-declared as substitutes for the
registered conditioners: *a level is worth more when it has been there a while
and when a lot of gamma sits on it.* That is directly actionable and it is what
the `level_persistence` and `gex_share` features already compute.

What showed **nothing**: gamma regime (positive vs negative, +4.4 vs +3.6, both
non-significant) and time of day (first 90 minutes actually −0.8pp). The
volatility-regime split not mattering is a mild surprise given the dispersion
thesis, and worth revisiting with more sessions.

## Per-symbol, including one that goes the other way

| Symbol | Gap | |
|---|---|---|
| IWM | **+17.1pp [+6.2, +27.1]** | strongest |
| SPY | +5.2pp [-3.4, +13.1] | positive, not significant alone |
| QQQ | +4.9pp [-3.0, +11.8] | positive, not significant alone |
| GLD | −2.4pp [-11.1, +6.2] | nothing |
| SLV | **−11.2pp [-20.6, −1.8]** | **significantly negative** |

SLV is not a rounding artifact — its interval excludes zero in the wrong
direction. Either its option structure genuinely behaves differently (it is the
least liquid and has a $0.50 strike grid), or this is the one false positive you
would expect from five symbols at 95% confidence. **Do not trade SLV levels on
this study**, and do not quietly drop it from the pooled number either — it is
in there.

## What this does and does not say about the 100–300% callouts

It supports them, with a caveat about magnitude.

- A 66–70% rejection rate at a call wall, on a 20bps underlying move, is a
  genuinely tradeable edge — and on a 0DTE option a 20bps SPY move *is* a 100%+
  premium move. So the observed edge and the observed returns are consistent.
- But the level edge itself is **~9pp over baseline, not a 90% signal.** A
  trader running at 90% is adding selection on top: which wall, which session,
  entry timing, and when to stand aside. This study measures the level, not the
  trader.
- Note the baseline: **price rejects at 61.6% of *ordinary* strikes.** Any
  claim about a level has to clear that, and most of what looks like "the level
  worked" is the base rate.

## Limitations, stated plainly

1. **41 sessions, one volatility regime**, mid-2026. Nothing here says the
   effect survives a different tape.
2. **49% of arrivals never resolve** within 30 minutes. Those are reported as
   `neither`, not dropped, but the result describes resolved arrivals only.
3. **ETFs only.** Says nothing about single names, where dealer inference is
   much weaker.
4. **This measures the underlying, not P&L.** No spread, no slippage, no option
   pricing. It is evidence that the level matters, not that a strategy is
   profitable.
5. **The exploratory section is exploratory.** Six regime splits plus five
   per-symbol rows is eleven comparisons; at 95% confidence you expect roughly
   one false positive, and SLV is a candidate.

## Next step

The confirmatory arm passed, so the registered decision applies: **the call-wall
touch feature graduates into the intraday structure engine as a level-strength
weight**, conditioned on persistence and concentration.

Before wiring it, the honest next move is the cheapest one: this study will
re-run unchanged on sessions after 2026-08-26. Those sessions are genuinely
out-of-sample. If the call wall holds near +9pp on fresh data, it is real; if it
halves, this was a 41-session regime.
