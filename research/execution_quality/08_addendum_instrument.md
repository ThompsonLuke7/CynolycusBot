# Addendum — the loss is in the instrument, not the model

Added 2026-08-29, prompted by the question "if it isn't execution, must it be the models?"

The Stage 1–5 study measured everything on the **underlying**, deliberately: the
2026-07 retraction happened because option "prices" were stale trade prints
(corr with underlying +0.09), so the standing rule is that timing metrics never
touch option P&L. That rule is correct — and it is exactly what hid this result
until someone asked the follow-up question.

## The gap

| module | route | n | median underlying move | median instrument return | total |
|---|---|---|---|---|---|
| dealer_ranker | option | 27 | −2.8% | **−61.8%** | −$69,916 |
| momentum_expansion | option | 17 | −2.0% | **−48.7%** | −$37,882 |
| multi_ticker_swing_htf | option | 17 | −3.5% | **−49.3%** | −$11,647 |
| meta_ranker | option | 10 | −0.8% | **−40.5%** | −$9,541 |
| multi_ticker_swing (30m) | option | 117 | −0.2% | **−21.6%** | −$57,149 |
| spy_daytrader | option | 95 | −0.0% | −5.7% | −$360 |
| momentum_expansion | equity | 6 | +8.9% | +4.6% | −$1,865 |
| multi_ticker_swing_htf | equity | 22 | +4.2% | **+4.1%** | **+$7,879** |
| meta_ranker | equity | 9 | −38.8% | −34.7% | −$53,719 |

**Equity routes track the underlying. Option routes do not.** Across all 289
closed option lifecycles the median underlying move is **−0.03%** — flat — and
the median option return is **−17.07%**.

## The counterfactual

| | |
|---|---|
| premium deployed | **$816,712** |
| actual option P&L | **−$186,503** (−22.8% of premium) |
| the same dollars held as shares | **−$5,267** (−0.6%) |
| **cost of the option wrapper** | **−$181,236** |

That is ~97% of the entire realized loss. The signal is roughly break-even on the
underlying, and the option overlay converts break-even into −22.8%.

`corr(underlying return, option return) = +0.591` across these lifecycles — well
clear of the +0.09 that invalidated the earlier study, so these are real marks
and this is genuine premium decay, not a data artifact.

## Why this is consistent with Stage 4A, not a contradiction

Stage 4A found the ranking does not order forward moves — the signal is roughly
zero-edge. A zero-edge signal expressed in **shares** costs approximately nothing
(−0.6%). The same zero-edge signal expressed in **21-DTE calls** costs 22.8% of
deployed premium, because theta and spread are charged whether or not the thesis
works. Leverage does not create edge; it multiplies whatever edge exists, and
here that is ~0, so it only multiplies the carry.

The two findings stack:
* **No edge** → nothing to harvest.
* **Option wrapper** → an unconditional fee charged on top of nothing.

Fixing the second does not create profit. It stops a ~$181k bleed while the first
is being worked on, and it is a one-line policy default rather than a research
programme.

## Where routing is decided

`route_option_or_shares` in `signals/meta_context/meta_ranker/options_exec.py:411`.
Options are the **default**; shares are the fallback taken only when the
underlying is under the price floor, no contract is listed, or the contract is
illiquid. So the system's standard expression of a signal is a long call, and
shares happen by accident of filtering.

## Honest limits

* The share counterfactual is **premium-matched, not delta-matched**. $816k of
  premium controls far more than $816k of stock, so a delta-matched share book
  would be larger and would scale its result up. The comparison answers "what did
  the wrapper cost on the same capital", which is the deployment question.
* Equity samples are small per module (6–22), and Meta's equity leg lost $53,719
  on 9 trades. "Shares are safe" is not the claim; "options charge a large
  unconditional fee that a zero-edge signal cannot pay" is.
* Options remain the correct instrument **once an edge is demonstrated** — the
  take-profit exits show 3.03 ATR reached and 2.62 ATR kept, and leverage is what
  makes those pay. This is an argument about sequencing, not about options.
