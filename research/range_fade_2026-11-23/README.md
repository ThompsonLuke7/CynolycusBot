# Range fade on QQQ — 2026-11-23

**Question (user, daily review 2026-11-23):** QQQ ranged ~648–653 from about 11:00 ET to the close after
a +7% gap. Why didn't intraday structure (or any intraday module) trade the bounces, and what would
enable it? The user's idea: price bounces off fuzzy support/resistance zones; after repeated bounces, fade each edge
and exit near the other side.

**Status:** research only. Nothing here is wired into live or paper trading.

## 1. What the live module did on QQQ (paper ledger, `Data/inference/intraday_structure/`)

* One modeled trade: `vwap_reclaim_continuation` **short**, entry 652.13 at 09:33, exit 647.64 at 09:38
  (extension failed), **+2.43 R after costs** (+0.69% gross). That was the opening flush, not the range.
* The last QQQ setup transition came at **09:52 ET**. After that, QQQ produced 304 abstentions,
  **all short**, all `invalidation_wider_than_max_atr`.
* **Three defects kept the range out of reach:**
  1. **A stale short setup.** `QQQ:short:gamma_structural_level_rejection` has been **ARMED since 2026-11-12
     11:24 ET** with its pivot at ~693.9 (invalidation 694.11). Price never crossed back above 694, and every
     confirmation is refused because the proposed risk is 26–185 ATR, so the setup can neither invalidate nor
     enter. It has logged 1,746 abstentions so far. `QQQ:short:trend_pullback_continuation` has the same problem
     (invalidation 741.07). Setup ids are unique per ticker/direction/detector, so no fresh short could arm at
     today's 652–653 resistance. ARMED has no age limit; `max_setup_bars` is only checked at CONFIRMED.
  2. **Each setup gets one try per day.** A CLOSED setup is revived only inside `_ensure_setups`, which runs only when
     the candidate is re-registered. The QQQ long candidate was last refreshed at 04:00 ET, so the long
     rejection, pullback and VWAP-reclaim setups that closed between 09:48 and 09:52 never came back. Across the
     whole module since 2026-11-02, 4,251 setups closed during RTH and only 264 (6.2%) were detected again later
     the same day.
  3. **A wrong gap reference in discovery.** `candidate_sources.load_discovery_baselines` uses the
     `px` column of `shared_universe.csv` as `prior_close`. That column is stale: across the top 750 eligible
     names it is a median 19.7% off the last daily close, and 84% of names are more than 5% off. For QQQ, px is 742.60
     against an actual prior close of 608.45, so QQQ was promoted as a **−11.7% gap-down short** on a +7% gap up.
  4. **The VWAP exit kills range fades by construction.** The exhaustion check (`engine.py:531`) runs on every
     RUNNING setup and closes a long when `distance_to_vwap_atr < −0.25` (mirrored for shorts). In a balance
     range, session VWAP sits in the middle, so a long bought at the floor or a short sold at the ceiling is already
     on the "lost VWAP" side and exits on its first managed bar. By the same check, 366 of the 757
     structural-rejection setups to date exited "extension conditions deteriorated".
* There is no range or balance fade detector. `gamma_structural_level_rejection` is the closest thing, but its
  stops scale to the 14 × 1-minute ATR (about 0.5 pt on QQQ today, against a range about 5 pt wide), and its levels are
  chain-clustered (see `project_intraday_level_chain_clustering`).

Other modules: Amethyst's QQQ point simulation took 4 longs (put_wall_bounce +2.52 pt; two bullish_magnet stops,
−0.80 and −1.76 pt; one put_wall_bounce still open at the close). `call_wall_rejection` and `bearish_magnet` exist
in `strategies/dealer_positioning/signals.py` but are **disabled** by `enabled_signal_types`, so nothing in the
stack could short the top of the range.

## 2. Does the idea have an edge? (scripts 00–03)

**Rule (01).** Every input is causal, and the entry fills at the next bar's open.
* **Envelope:** the high and low of the prior 60 one-minute bars.
* **Balance gate:** Kaufman efficiency below 0.30, width between 0.30% and 1.5%, and both edges set at least 10 bars ago.
* **Signal:** a rejection bar in the outer 20% of the envelope.
* **Stop:** 0.25 × width beyond the tested edge.
* **Target:** the midline (0.5) or the far edge (0.8).
* **Trading window:** entries 10:30–15:30, flat at 15:55.

Two matched controls use the same gate: a breakout mirror (opposite side, mirrored bracket) and random-time,
random-side entries with the same bracket distances. `--min-touch 2` adds the user's condition, "it has already
bounced": at least 2 prior separate visits to the tested edge and at least 1 to the other edge.

All results are gross bps per trade: mean, with standard error (SE) and the minimum detectable effect at 80% power (MDE80).

| Data | Variant | n | Fade, gross bps/trade | Random | Breakout |
|---|---|---|---|---|---|
| QQQ 1m, 2024-02-01..2026-11-23 (706 sessions) | tgt 0.5 | 2,774 | **−0.06** (SE 0.42, MDE80 1.2) | −0.14 | +0.05 |
| | tgt 0.8 | 1,897 | −0.05 (SE 0.75) | −1.64 | +0.03 |
| | tgt 0.5, min-touch 2 | 2,049 | −0.04 (SE 0.44) | −0.14 | +0.04 |
| | tgt 0.8, min-touch 2 | 1,518 | +0.48 (SE 0.76) | −1.64 | −0.48 |
| SPY 1m, 2024-06..2026-11 | tgt 0.5, min-touch 2 | 1,019 | +0.24 (SE 0.56) | +0.08 | −0.24 |

By year (2024 / 2025 / 2026) the fade is flat in every year. The 61% win rate is entirely offset by losses
that are larger than the wins.

**Result: no measurable edge.** At the 1-minute scale, where price sits inside a balance envelope does not
predict which side gets hit first, whether or not the edge has already held. The MDE80 is about 1.2 bps per trade,
so any true edge is smaller than the round-trip cost (module default 12 bps; QQQ shares realistically 1–2 bps).
That is a rejection at this power, not an underpowered null.

**Today in isolation:** the same rule made +141 bps gross on 11-23 at tgt 0.8 (7 trades, 5 targets hit) and
+30 bps at tgt 0.5. The first trade, at 11:17, was a −62 bps stop: the 650–657 envelope broke down before the
648–653 range formed. Today is the kind of day the eye remembers. The rule cannot know in advance which
balance will hold.

**Naive pivot clustering (00), today only:** 7 trades, −0.9 bps gross in total. Clustering pivots finds zones in the
middle of the chop (650–651.7), not the two outer edges the eye picks out.

## 3. Next testable variant: dealer gamma regime (02, 03)

The variant: a fade should work when dealers are long gamma (they hedge against moves, which pins price) and
fail when they are short gamma. Today fits the variant: QQQ total GEX was positive all session (+95..+108M), and the
gamma flip held at 648.4–648.7 all day, which is the lower edge of the range. 02 rebuilds per-minute
total GEX and the gamma flip from the 26,414 archived Schwab-chain ladders (80 sessions, 2026-06-17..11-23).
03 joins each trade to the latest snapshot at or before the signal bar's close (maximum age 10 minutes).

| Variant (QQQ, sessions with snapshots) | Fade GEX+ | Fade GEX− | Difference |
|---|---|---|---|
| tgt 0.5 | +0.87 (SE 1.90, n 117) | −2.15 (SE 1.84, n 175) | +3.0 |
| tgt 0.5, min-touch 2 | +2.26 (SE 2.14, n 76) | −2.54 (SE 2.06, n 129) | +4.8 (t ≈ 1.6) |
| tgt 0.8, min-touch 2 | +5.41 (SE 3.80, n 59) | −3.01 (SE 3.52, n 88) | +8.4 |

The sign matches the hypothesis in all four variants. The variants share trades, so they are not independent
evidence, and none is significant: MDE80 is 5–10 bps per cell with 72–74 days. **Not measurable here yet.**
The GEX sign comes from the standard open-interest convention (`dealer_inventory_estimated`), not observed
dealer inventory, and the random control's own GEX± split swings ±4–6 bps, which shows how noisy cells this size are.

**What would settle it:**
1. Pool the SPY, IWM and SPX snapshot history with QQQ (same window) to roughly 3–4× the trade count, and pre-register
   min-touch 2 with tgt 0.8.
2. Pre-register the difference between the GEX+ and GEX− fade as the test statistic. Keep the random and breakout controls.
3. If it holds (difference > 0 at p < 0.05, with GEX+ fades net positive at 2 bps), shadow it in paper as a separate
   detector before any live wiring.

## Reproduce

```bash
D=research/range_fade_2026-11-23
.venv/bin/python $D/01_balance_fade_vs_controls.py --extra-jsonl Data/archive/intraday_1m/bars_20261123.jsonl --tgt 0.5 --min-touch 2 --out $D/out/qqq_fade_tgt0.5_touch2.csv
.venv/bin/python $D/01_balance_fade_vs_controls.py --parquet Data/raw/spy/spy_intraday_1min_runtime_rth_cache.parquet --symbol SPY --tgt 0.5 --min-touch 2
PYTHONPATH=. .venv/bin/python $D/02_qqq_gex_series.py $D/out/qqq_gex_series.csv
.venv/bin/python $D/03_fade_by_gex_regime.py $D/out/qqq_gex_series.csv $D/out/qqq_fade_tgt0.5_touch2.csv
```

Data: Alpaca 1-minute bars, IEX volume (volume is not used by these rules). QQQ history is
`Data/raw/qqq/qqq_intraday_1min.parquet`, with 11-23 taken from the live stream archive. These are
model outcomes at bar prices, not fills. The random control uses seed 7.

## 4. Follow-up (same day, later): the four blockers

- **Fixed in code, 2026-11-23 evening. Takes effect at the next server restart.**
  1. **Stale and stuck setups.**
     - Pre-entry setups close at the first bar of a new session.
     - They also close after `max_setup_bars`, now counted from detection, not from creation of the empty shell.
     - They close when a confirmation is refused at ≥ 2 × `max_invalidation_atr` ("pivot out of reach"),
       followed by the failure cooldown.
  2. **One try per day.** Revival now runs on every bar, not only on candidate registration.
  3. **Gap reference.** The prior close comes from the 1d bar cache, and a name whose cache is behind is skipped.
- **Replay check.** Replaying 11-23 QQQ through the engine (null options provider), the HEAD code made
  0 entries and the patched code 30 (28 after 10:00, 16 long and 14 short).
- **Those 30 trades lost.** They came to −36 bps gross in total (11 winners of the 40 replayed across
  11-20 and 11-23). Freeing the module lets it trade the day, but its entries still have no measured edge.
- **Blocker 4 (the VWAP-lost exit) is unchanged.** See regime_switch §8.4: holding longer did not help
  after 10:00.
- **The dealer levels were in the engine's hands.** On 11-23 the live QQQ snapshot gave call wall 652,
  put wall 650, gamma flip 648.45 and GEX positive. That is the range the user drew.
