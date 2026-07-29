# Option Cost & Sizing Reference (rough)

**Source:** 575 real closed option round-trips from live trading (`paired_option_trades.csv`) — actual fills, not modeled marks.

**Status:** rough estimate from one module's live history. NOT a validated model. Everything derived from historical option bars was retracted (see `10_RETRACTION...`), so this is the only option data in the project that can be trusted.

## 1. What was actually being traded

| metric | value |
|---|---:|
| trades | 575 |
| calls / puts | 274 / 301 |
| **median DTE at entry** | **2 days** |
| DTE p25 / p75 | 1 / 4 |
| share entered with DTE <= 2 | 58% |
| median premium | $0.95 |
| premium p25 / p75 | $0.40 / $2.31 |
| median holding time | 20.3 hours |

> **The single biggest risk factor here is DTE, not spread.** A median 2-day option is almost pure theta and gamma; a small adverse move is unrecoverable. Any sizing rule that ignores DTE is mis-specified.

## 2. Round-trip cost

Using the 245 trades with live-recorded marks:

| premium bucket | n | median premium | est. round-trip cost | as % of premium |
|---|---:|---:|---:|---:|
| <$0.50 | 180 | $0.28 | ~$0.16/share | **58%** |
| $0.50-1 | 122 | $0.80 | ~$0.16/share | **20%** |
| $1-2 | 109 | $1.40 | ~$0.16/share | **11%** |
| $2-5 | 100 | $2.92 | ~$0.16/share | **5%** |
| >$5 | 64 | $9.52 | ~$0.16/share | **2%** |

Assumes a **~8-cent half-spread**, the median observed against real fills in Gate G1. Spread is roughly a fixed number of CENTS, so it is punitive on cheap contracts and mild on expensive ones — the same 8 cents is 32% round-trip on a $0.50 option and 3% on a $5.00 option.

## 3. Realized outcomes (real fills)

### by DTE at entry

| DTE | n | win rate | median return | mean return | total P&L |
|---|---:|---:|---:|---:|---:|
| 0-1d | 252 | 40% | -17% | +4% | $-3,286 |
| 2-3d | 164 | 41% | -17% | -0% | $-7,469 |
| 4-7d | 129 | 31% | -32% | -20% | $-3,339 |
| >21d | 30 | 17% | -21% | -23% | $-13,295 |

### by premium paid

| premium | n | win rate | median return | total P&L |
|---|---:|---:|---:|---:|
| <$0.50 | 180 | 40% | -17% | $-129 |
| $0.50-1 | 122 | 35% | -15% | $7 |
| $1-2 | 109 | 36% | -32% | $881 |
| $2-5 | 100 | 34% | -32% | $-6,448 |
| >$5 | 64 | 39% | -10% | $-21,700 |

### calls vs puts

| side | n | win rate | median return | total P&L |
|---|---:|---:|---:|---:|
| calls | 274 | 43% | -12% | $5,523 |
| puts | 301 | 32% | -28% | $-32,912 |

## 4. Sizing implications

- **Assume total loss is the base case, not the tail.** 11% of these trades lost 90%+ of premium. Position size must be survivable at -100%.
- Worst single trade **$-5,070**; median win **$43** — one worst-case loss offsets **118** median wins.
- **Cheap contracts are expensive.** Below $0.50 the spread alone is ~30% round trip. Prefer fewer, more expensive contracts over many cheap ones for the same notional.
- **DTE floor.** 58% of these were entered at <=2 DTE, and that bucket is where the losses concentrate. A minimum-DTE rule is likely the highest-value single change.
- Size from **premium at risk**, not underlying notional: an option position's max loss is the premium, so `contracts = risk_budget / (premium * 100)`.

## 5. What this cannot tell you

- Nothing about multi-leg structures — no spread trades exist in this history.
- Nothing about strategy selection by regime; that needs option marks captured going forward, which do not exist historically for this universe.
- It is one module's book (multi_ticker_swing), median underlying $83, so it may not generalize to the 4H modules.
