# Study A results — does price react at gamma levels?

Generated from `study_a_events.parquet`. **4,445 arrival events**, of which **2,186 resolved** (rejection or penetration) across **41 sessions** and **5 symbols**.

An event is price arriving within 10bps of an option strike. Every event is the same kind of event; the strikes differ only in how much gamma sits on them. Outcome thresholds are symmetric (20bps each way) so neither outcome is favoured by the measurement.

## 1. Base rates — what happens at an ordinary strike

Read this first. It is the number every result below has to beat.

| symbol | events | resolved | rejection % | sessions |
|---|---:|---:|---:|---:|
| GLD | 833 | 414 | 65.2 | 39 |
| IWM | 649 | 319 | 69.3 | 39 |
| QQQ | 1424 | 816 | 56.7 | 39 |
| SLV | 347 | 286 | 60.1 | 40 |
| SPY | 1192 | 351 | 63.0 | 40 |

Pooled resolution rate: **49%** of arrivals resolve within 30 minutes; the rest stay inside the band and are reported as `neither`, never dropped.

Pooled rejection rate at **any** strike: **61.6%**.

## 2. Confirmatory — the pre-declared comparison

Session-clustered bootstrap, 2,000 draws. `**` marks an interval excluding zero. The registered decision threshold is **>= 8pp with an interval excluding zero**.

| comparison | n treated | n control | rej% treated | rej% control | gap (pp) | 95% CI | sessions |
|---|---:|---:|---:|---:|---:|---|---:|
| any gamma level vs plain strike ** | 791 | 1395 | 64.3 | 60.1 | +4.3 | [+0.2, +8.2] | 190 |
| call wall vs everything else ** | 239 | 1947 | 69.9 | 60.6 | +9.3 | [+3.0, +15.3] | 190 |
| put wall vs everything else | 243 | 1943 | 62.6 | 61.5 | +1.0 | [-4.5, +6.4] | 190 |
| magnet vs everything else | 558 | 1628 | 62.7 | 61.2 | +1.5 | [-3.2, +6.1] | 190 |

### 2b. Walls only — removing the magnet's circularity

Gamma peaks at the money, so the magnet is the strike nearest spot 31-70% of the time (SPY 70%). "Price arrived at the magnet" is therefore partly the statement "price is where price is", and any arm containing magnets inherits that circularity.

The walls do not have this problem: the call wall sits within 10bps of spot only 14% of the time and the put wall 12%, because they are defined on the far side of spot. This is the clean test.

| comparison | n treated | n control | rej% treated | rej% control | gap (pp) | 95% CI | sessions |
|---|---:|---:|---:|---:|---:|---|---:|
| wall (call or put) vs plain strike, magnets excluded ** | 233 | 1395 | 68.2 | 60.1 | +8.2 | [+1.5, +14.5] | 179 |

## 3. Exploratory — regime splits

> **These are exploratory.** Many comparisons on one sample. They generate hypotheses; they do not confirm them. Nothing here should be wired into a strategy without confirmation on later, unseen sessions.

| comparison | n treated | n control | rej% treated | rej% control | gap (pp) | 95% CI | sessions |
|---|---:|---:|---:|---:|---:|---|---:|
| gamma level, positive-gamma regime | 405 | 1781 | 65.2 | 60.8 | +4.4 | [-1.0, +9.6] | 190 |
| gamma level, negative-gamma regime | 334 | 1852 | 64.7 | 61.1 | +3.6 | [-1.7, +8.7] | 190 |
| gamma level, first 90 min | 220 | 1966 | 60.9 | 61.7 | -0.8 | [-8.0, +6.1] | 190 |
| gamma level, last hour | 116 | 2070 | 66.4 | 61.4 | +5.0 | [-4.1, +14.9] | 190 |
| gamma level that has held >= median duration ** | 368 | 1818 | 66.8 | 60.6 | +6.3 | [+1.2, +11.4] | 190 |
| gamma level in top-quartile concentration ** | 240 | 1946 | 67.9 | 60.8 | +7.1 | [+1.5, +13.2] | 190 |

### Per-symbol, gamma level vs plain strike

| comparison | n treated | n control | rej% treated | rej% control | gap (pp) | 95% CI | sessions |
|---|---:|---:|---:|---:|---:|---|---:|
| GLD | 143 | 271 | 63.6 | 66.1 | -2.4 | [-11.1, +6.2] | 39 |
| IWM ** | 155 | 164 | 78.1 | 61.0 | +17.1 | [+6.2, +27.1] | 39 |
| QQQ | 224 | 592 | 60.3 | 55.4 | +4.9 | [-3.0, +11.8] | 39 |
| SLV ** | 133 | 153 | 54.1 | 65.4 | -11.2 | [-20.6, -1.8] | 40 |
| SPY | 136 | 215 | 66.2 | 60.9 | +5.2 | [-3.4, +13.1] | 33 |

