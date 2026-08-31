# Stage 2 — the signal spine. 2,279 ranked signals, traded or not

Run 2026-08-28. Script: `scripts/execution_quality/stage2_signal_spine.py`
Output: `research/execution_quality/data/stage2_signal_spine.jsonl`

One row per (module, ticker, decision bar), whether or not capital followed. This
is the sample for "do the module signals even make sense" — the traded rows alone
are selected *by the order policy*, so any edge measured on them is contaminated
by that selection.

| module | signals | actionable | planned | traded | plan rate | plan→fill |
|---|---|---|---|---|---|---|
| momentum_expansion | 654 | 654 | 159 | 82 | 24.3% | 51.6% |
| multi_ticker_swing_htf | 716 | 676 | 110 | 77 | 16.3% | 70.0% |
| meta_ranker | 718 | 628 | 88 | 47 | 14.0% | 53.4% |
| dealer_ranker | 191 | 171 | 99 | 40 | 57.9% | 40.4% |
| **total** | **2,279** | **2,129** | **456** | **246** | | |

"Actionable" excludes dry runs (`submit: false`) and non-RTH wall-clock runs.
Each row carries score, rank, rank_pct, the bucket labels, and the module's own
`extra` block flattened to scalars (`trigger_rule`, `dollar_vol_pctile_252`,
`mom_score`/`htf_score`/`news_catalyst_score`, `quality_entry_ok`, the dealer
components), plus `planned_entry` and `was_traded` so the policy's selection is
studied rather than silently conditioned on.

A de-duplication detail that mattered: one decision is logged twice — once as
`signal_decision`, once as `order_plan` — and only the second carries `plan`. The
planned-entry set is therefore collected per (module, bar) *before* de-duplication,
or the flag reads zero for three of the four modules.

## Two funnel facts worth carrying into Stage 4

**Only 54% of planned entries become positions** (246 of 456). The gap is the
order policy — unfilled ladders, gates, capacity — and it is large enough that
"what the module said" and "what we owned" are materially different populations.
Momentum and Meta convert about half; dealer_ranker only 40%.

**The policy systematically does not buy the top of the ranking.** Median
`rank_pct` among planned entries vs among signals that were not planned:

| module | planned | not planned |
|---|---|---|
| meta_ranker | 0.400 | 0.600 |
| momentum_expansion | 0.500 | 0.600 |
| dealer_ranker | 0.500 | 0.600 |
| multi_ticker_swing_htf | 0.997 | 0.997 |

Three of four modules allocate capital to *lower*-ranked names than the ones they
skip. The likely mechanical cause is benign — the highest-ranked names are often
already held, so they cannot generate a new entry, and others fail liquidity or
option-availability gates. But "likely benign" is a hypothesis, and if the model's
ranking carries real information (Analysis A), then routing capital away from the
top of it has a measurable cost. Stage 4A tests both halves: whether rank predicts
forward move at all, and what the observed rank displacement is worth.

HTF is the exception and looks different for a structural reason — it plans from a
tight band at the very top of its ranking (0.997), so there is no displacement to
measure there.

Next: Stage 3, price paths.
