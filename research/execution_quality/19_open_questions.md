# Why long calls don't work here — and the open-questions list

2026-09-04.

## The long-call question, answered: it's the underlying, not the selection

Measured option spreads (live snapshots, median % of mid, strikes with mid ≥ $0.20):

| underlying | median spread | | underlying | median spread |
|---|---|---|---|---|
| **SPY** | **1.9%** | | OKLO | 11.1% |
| IWM | 2.2% | | RFIL | 32.0% |
| AMD | 2.8% | | SLS | 37.4% |
| MSTR | 2.9% | | TNXP | 76.2% |
| NVDA | 4.4% | | | |
| QQQ | 4.9% | | | |
| AAPL | 5.1% | | | |

**Break-even round-trip cost for the best option policy was 11.9% of premium.**

* On SPY/IWM/AMD/MSTR, a round trip costs roughly **2–5%** — clears break-even by
  3–6x.
* On the names the 4H modules actually trade, it costs **32–76%** — impossible by
  a factor of 3–6 in the other direction.

The people you are watching trade long calls on SPY, QQQ, NVDA, AAPL. **They are
not making better selections against the same cost structure; they are paying
1.9% where we pay 37%.** Same strategy, different instrument market.

Two supporting facts from our own fills:

* Spread scales inversely with premium: $0–0.50 options quote a **50%** median
  spread, $10+ options quote **14.4%**. Our median premium *at exit* is **$1.20**,
  and **46% of our exit quotes are under $1.00** — we systematically exit in the
  worst bucket.
* `intraday_structure` already trades the right names (IWM, SPY, GOOG, AVGO, QQQ;
  median price $114). The 4H modules do not.

**Implication:** the option expression is not dead — it is being applied to the
wrong universe. The fix is a **liquidity filter on the option market**, not a
model change: only express in options when the contract's quoted spread is under
some threshold, else take shares. That is a routing rule, and `route_option_or_shares`
is already the place it belongs.

## On meta-labelling: what I was arguing, and how to test the premise

The claim was **not** "rules are better than ML". It is a division of labour:

* the **primary model** (a rule) needs **recall** — it must fire on most of the
  moves worth taking, even if it also fires on a lot of junk
* the **ML model** supplies **precision** — given a rule-triggered setup, predict
  whether it works, and size accordingly

Your objection was that rules-based has lost money every time. That is consistent
with this design rather than an argument against it: a rule that loses money *by
firing too often on bad setups* is exactly the case meta-labelling fixes. The
design fails in the opposite case — when the rules **miss** the moves — because
ML cannot rescue a trade that was never proposed.

**So the premise is testable before committing to it**, and we now have the data:
`intraday_structure` has written 9,931 `setup_abstention` records and 731
`candidate_fixed_horizon_outcome` records. The question is whether the setups it
declines would have worked. If abstained candidates perform *no better* than
confirmed ones, recall is fine and meta-labelling has something to work with. If
the abstained ones are where the moves were, recall is the problem and no
meta-label will help.

---

# Outstanding questions and investigations

## A. Ready to run now (no blockers)

1. **Option-market liquidity filter.** Measure, per traded name, the quoted
   option spread at entry; re-run the thesis grid restricted to names quoting
   under 5% / 10%. Decides whether options are viable on *any* subset of the
   current universe. **Highest expected value of anything on this list.**
2. **Meta retrain** on `meta_rank_quality` + regime panel. Bundle is built at
   `signals/meta_context/meta_ranker/meta_ranker_colab_bundle.tgz`. Not yet run.
3. **Meta-labelling premise test.** Do `intraday_structure`'s abstentions contain
   the moves? (recall check, above) — decides whether the whole meta-labelling
   direction is worth pursuing.
4. **Exit-execution study.** Break-even is 11.9% round trip; measured is 16.2%,
   of which exit is 12.2% and 97% of exits fill worse than mid. Test limit
   ladders vs the current market-order exit path.
5. **Momentum / HTF matrix refresh.** Data ends 2026-05-14, ~3.5 months stale.
   A data refresh, not a label change — the bake-off said leave those labels alone.

## B. Blocked on data collection

6. **Switch on option mark capture.** `option_mark_capture.py` exists and has
   **never written a file** (`Data/inference/spy/option_marks/` is empty). Until
   it runs, every option-cost number is borrowed from the 30m module.
7. **Intraday structure real fills.** Armed and enabled since 2026-08-31; needs
   ~30 fills for a cost check and ~100 for expectancy. Zero so far.
8. **`momentum_scalper`.** Architecturally complete, blocked on point-in-time
   float, catalyst availability and bid/ask quotes. `Data/raw/momentum_scalper/`
   does not exist.
9. **Afternoon 4H bar deferral.** The 18:00Z bar completes at the close, so its
   entries cross an overnight gap (~17.7h, 8.5% of joined entries). 21 entries,
   only 7 closed. **Revisit at n ≥ 20 closed.**

## C. Known defects, not yet fixed

10. **Dealer Ranker option costs.** Worst entry slippage of the four modules
    (+7.9% vs +1.5–3.3%), and it trades options exclusively. Should it route to
    shares, or be excluded from options entirely?
11. **Two declared-but-missing features.** `FEATURE_COLUMNS_4H` declares
    `xsec_dollar_vol_surge_20_rank` and `xsec_rs_spy_20_rank`, absent from the
    built matrix (108 declared vs 106 present).
12. **HTF label weights are fiction.** Nominal 35/25/25/15, effective 4/92/1/3.
    The bake-off said *do not fix it* (fixing it halves performance), but the
    config now lies about what it does — worth a comment at minimum.
13. **Nervous system covers Meta only.** Momentum, HTF and intraday_structure
    submit directly, inventoried in `test_mvp_acceptance.LEGACY_DIRECT_SUBMIT`.

## D. Open research questions

14. **Multi-leg option strategies** (you are running this). The spread finding
    above applies with force: a vertical charges the spread on **both** legs, so
    on a 37%-spread name a spread trade starts ~74% behind. On SPY-class names at
    1.9% it is a different proposition entirely.
15. **Does the ranking work at all on a tight-spread universe?** Every IC number
    we have is measured on the current universe. If we restrict to optionable
    liquid names, does rank still order forward moves?
16. **Shares at 15–25 days** is the only post-cost positive expression measured
    (+$2,340 at 20d per $1k). It has never been run as an actual strategy.
