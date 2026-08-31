# Label diagnosis — the benchmark adjustment is a no-op, and the biggest weight ranks volatility

2026-08-29. Scripts: `scripts/execution_quality/stage7_label_diagnosis.py`

## The headline defect

`momentum_expansion`'s label composite puts its **largest weight (0.40) on
`fwd_max_alpha`**, defined as:

```python
fwd_max_alpha = fwd_max_return - bench_fwd_max_return      # bench = SPY
```

and the composite is then ranked **within each decision timestamp**:

```python
ts_level = df.index.names[0] ...
df.groupby(level=ts_level)[col].rank(pct=True, ascending=ascending)
```

`bench_fwd_max_return` is **the same number for every ticker on a given bar** —
it is SPY's forward max, and SPY does not vary by ticker. Subtracting a constant
cannot change a cross-sectional rank.

Measured on 2,129 live signals:

```
fwd_max_alpha vs fwd_max_return, within bar:  rho = +1.000
```

**The SPY subtraction is a complete no-op.** The 0.40-weight component is exactly
raw forward max return, ranked. This directly answers "does the ranking take the
index into account?" — it was built to, and it does not.

## What that component actually ranks

Within-bar rank correlation against each ticker's own ATR% (volatility):

| horizon | `fwd_max_alpha` | `fwd_max_return` | `fwd_atr_adj_return` |
|---|---|---|---|
| 5d | +0.216 | +0.216 | +0.002 |
| 10d | +0.230 | +0.230 | +0.008 |
| 15d | +0.260 | +0.260 | +0.045 |
| 20d | +0.306 | +0.306 | +0.083 |

The alpha component tracks volatility (identically to raw return, as expected
once the subtraction is shown to cancel). `fwd_atr_adj_return` — weight 0.25 — is
the one component that *is* properly volatility-neutral, at ~0.00.

So the effective target is roughly **40% "which name is most volatile", 25%
volatility-adjusted excursion, 20% persistence, 15% drawdown.**

That is consistent with three things already measured elsewhere in this study:
the ranker's favourites cluster in $5–10 names ranked ~2x more often than pricier
ones (which then fail the option gate); the edge appearing to grow with horizon;
and MAE exceeding MFE at every horizon — all of which is what ranking volatility
produces, because volatility pays out in both directions.

## Is the label wrong, or are the features wrong?

Two separable tests. Within-bar Spearman, so a market-wide up day cannot
masquerade as ranking skill.

**A. Does the model rank its own training target?** (score vs the reproduced composite)

| module | 5d | 10d | 15d | 20d |
|---|---|---|---|---|
| dealer_ranker | +0.139 | +0.136 | +0.132 | +0.166 |
| meta_ranker | +0.070 | +0.096 | +0.074 | +0.094 |
| momentum_expansion | +0.114 | **+0.193** | **+0.198** | +0.186 |
| multi_ticker_swing_htf | +0.005 | +0.030 | +0.008 | −0.012 |

**B. Does the model rank tradeable move?** (score vs forward MFE in ATR)

| module | 5d | 10d | 15d | 20d |
|---|---|---|---|---|
| dealer_ranker | +0.146 | +0.152 | +0.147 | +0.196 |
| meta_ranker | +0.046 | +0.067 | +0.053 | +0.078 |
| momentum_expansion | +0.115 | +0.163 | +0.152 | +0.110 |
| multi_ticker_swing_htf | −0.034 | −0.002 | −0.009 | −0.045 |

**C. Does the composite predict tradeable move?** ρ = **+0.90 to +0.97** everywhere.

**C is largely mechanical and must not be read as "the label is well designed."**
0.65 of the composite's weight (0.40 alpha + 0.25 ATR-adjusted) is a monotone
transform of forward max return within a bar, and forward MFE in ATR is forward
max return scaled by ATR. They are near-duplicates by construction. What C does
establish is the narrower, still-useful point: **the label is not misaligned with
tradeable move, so relabelling for alignment would buy nothing.**

**The diagnosis is that the FEATURES carry almost no signal.** The model reaches
ρ ≈ 0.07–0.20 against a target that is itself ~0.93 correlated with what we want
to capture. The ceiling is not the label; the model simply cannot see the target
from its inputs. HTF is at ~0.00 and is the weakest of the four.

## What this changes

**Relabelling for horizon alignment is now the *second* priority, not the first.**
Stage 6's 3–4x horizon mismatch is real, but a relabelled target on the same
features would inherit the same ρ ≈ 0.15 ceiling.

Ranked by expected value:

1. **Fix `fwd_max_alpha`.** It is a one-line defect with a large weight. Either
   make it genuinely relative — `fwd_max_return - beta * bench_fwd_max_return`,
   or residualise against the ticker's beta — or drop it and re-weight onto
   `fwd_atr_adj_return`, which is already volatility-neutral. Until then the
   model is being taught that "most volatile" is the answer.
2. **Features.** ρ ≈ 0.15 against a near-perfect target is a feature problem.
   This is where new data since the last training belongs.
3. **Horizon alignment** (Stage 6), after 1 and 2.

Retraining still should not happen before 1 — refitting on a target whose largest
component ranks volatility just relearns volatility.

## Caveats

* One ~2-month sample, one regime, and the ranked universe fell over it.
* Within-bar Spearman needs >= 4 ranked names per bar; bars below that are dropped
  (17–64 usable bars per module).
* The composite is reproduced from `LABEL_CONFIG` on daily bars, while training
  used 4H bars. The no-op finding does not depend on bar size — it follows from
  the benchmark being constant per timestamp — but the correlation magnitudes
  would shift slightly on the training grid.
