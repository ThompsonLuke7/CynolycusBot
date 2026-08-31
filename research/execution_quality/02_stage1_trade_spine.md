# Stage 1 — the trade spine. 513 lifecycles, six modules, every clock

Run 2026-08-28. Script: `scripts/execution_quality/stage1_trade_spine.py`
Output: `research/execution_quality/data/stage1_trade_spine.jsonl`

## What a row is

Stage 0 established that a ledger row is not a trade — 56 of 183 are partial
scale-outs. A row here is a **position lifecycle** reconstructed from the broker
fill stream: flat → entry rungs → any adds → any trims → flat. That is the
authoritative record and it is module-agnostic, so it also picks up the two
modules that keep no `closed_trades.jsonl` at all.

| module | lifecycles | closed | plan-joined | median hold (min) | option | equity |
|---|---|---|---|---|---|---|
| spy_daytrader | 101 | 101 | – | 8.8 | 101 | 0 |
| multi_ticker_swing (30m) | 126 | 117 | – | 1,692 | 126 | 0 |
| multi_ticker_swing_htf | 101 | 39 | 78 | 10,078 | 23 | 78 |
| momentum_expansion | 84 | 23 | 82 | 5,759 | 21 | 63 |
| meta_ranker | 61 | 19 | 47 | 14,756 | 10 | 51 |
| dealer_ranker | 40 | 27 | 40 | 2,879 | 40 | 0 |
| **total** | **513** | **326** | **247** | | 321 | 192 |

513 of 572 lifecycles attributed (89.7%). Attribution is exact, never fuzzy:
broker order id from the 30m swing's session audit, ledger `order_id` for the 4H
modules, the module's own planned-entry record for positions still open, and
ticker for SPY. The 59 unattributed are left unattributed rather than guessed.

This is **2.8x the sample** the design assumed (183 ledger rows), and it adds the
30m swing and the SPY daytrader, which the ledger-only view could not see at all.

## A correction that mattered

The first join attached each lifecycle to the nearest preceding 4H bar. That was
wrong and it inflated latency by hours. The 4H modules run **twice a day**, and
Alpaca 4H bars are **left-labelled**: the bar stamped 14:00Z spans 14:00–18:00Z
and is not complete until 18:00Z. A run at 18:20Z has decided on the 14:00Z bar,
but "nearest preceding bar" hands it the 18:00Z one, which had existed for 20
minutes. Event time is not availability time.

The join now keys on **the order plan that actually contains this symbol as an
entry**, and latency is measured from `bar_close` (14:00Z bar → 18:00Z; 18:00Z
bar → 20:00Z market close), not from the bar label.

## The decision clock, corrected

Minutes from **bar completion** to order submission:

| module | bar | n | avail→submit (median) | p90 | avail→fill |
|---|---|---|---|---|---|
| dealer_ranker | wall clock | 40 | **0.1** | 1.2 | 0.2 |
| meta_ranker | 14:00Z | 45 | **20.1** | 20.1 | 20.1 |
| multi_ticker_swing_htf | 14:00Z | 78 | **25.1** | 25.2 | 25.1 |
| momentum_expansion | 14:00Z | 64 | **31.4** | 32.7 | 31.5 |
| meta_ranker | 18:00Z | 2 | **1,055** | 1,055 | 1,055 |
| momentum_expansion | 18:00Z | 18 | **1,064** | 1,065 | 1,064 |

Two clean regimes, and the p90 sitting on top of the median says both are
mechanical rather than noisy:

1. **Morning bar → same-session fill, 20–31 minutes after the bar closes.** Tight
   and predictable. Whether 20–31 minutes is *too slow* is Stage 4's question,
   not this one — but it is not erratic, and it is not hours.
2. **Afternoon bar → next-morning fill, ~17.7 hours later.** The 18:00Z bar only
   completes at the 20:00Z market close, so there is no session left to trade it
   in and the entry defers to the next open. **21 of 247 joined entries (8.5%)
   take this path, including 18 of momentum's 82.**

The second regime is the structurally interesting one. Those entries cross an
overnight gap: the fill price is set by the next morning's open, not by anything
the model saw. Stage 3 will price exactly what that gap costs, and whether the
afternoon bar's signals are worth acting on at all under that constraint.

Also captured per lifecycle and used downstream: entry/exit rung structure and
ladder duration (median ~0.1–1.6 s, mean 1.00 rungs — the ladder is filling on
its first rung, so ladder mechanics are not a latency source), trim schedule,
`decision_gain`/`stop_overshoot` from the ledger, option strike/expiry/DTE, and
the order audit's `underlying_price`, `delta`, `mid_price`, `breakeven_move_pct`.

Next: Stage 2, the signal spine — every ranked target, traded or not.
