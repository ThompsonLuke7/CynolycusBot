# Broad underlying-proxy reconciliation — ACE

This pass covers **all 222 ACE entry lifecycles**. It uses a documented underlying-price proxy where the Discord export lacks an option fill. It answers: “did the underlying move in the alerted call/put direction by a reported management time?” It does **not** answer option P&L, broker profitability, or follower fill quality.

## Outcome coverage

| Outcome class | Entries | Meaning |
| --- | ---: | --- |
| Reported-management proxy winner | 41 | Underlying moved more than +0.05% in the call/put direction from alert to a uniquely linked management anchor. |
| Reported-management proxy loser | 10 | Underlying moved more than −0.05% in that direction. |
| Expiry-horizon proxy winner / loser / flat | 44 / 57 / 6 | No reported management anchor; underlying direction from alert to the stated expiry’s 16:00 ET close. **Does not claim the position was held to expiry.** |
| No management anchor | 57 | No literal-symbol, full-contract, canonical, or uniquely close estimated link; no stated expiry horizon. |
| Missing snapshot / unresolved direction | 5 / 2 | Underlying-bar observation unavailable, or right/direction unavailable. |

The observed-management set is **41 favorable / 10 unfavorable** (80.4% favorable) on the underlying proxy. Of its 22 reported full-close anchors, 15 are favorable and 7 unfavorable. The other 29 management anchors are trims/add-related partial management and should not be treated as full trade outcomes. This is a selectively reported Discord subset, not ACE’s population win rate.

For the 41 favorable management anchors, median direction-adjusted underlying movement was **+1.251%** (mean +3.271%). For the 10 unfavorable anchors, median was **−0.165%** (mean −0.986%). A call counts an increase as favorable; a put counts a decrease as favorable. SPX/SPXW use SPY and NDX/NDXP use QQQ only as explicitly labelled ETF proxies.

## How the links are made

Every entry appears once in `underlying_proxy_ledger.jsonl`. The preferred management anchor is an already-canonical lifecycle link or visible full contract. A ticker literally present in management text is the next tier. Four unlabelled same-channel full-close posts are retained only when there is one uniquely closer open candidate within 36 hours; those rows are `very_low` confidence. Thirty-six competing messages were retained in `ambiguous_management_links.jsonl` rather than assigned arbitrarily.

The read-only IEX 1-minute stock-bar snapshot is the latest **completed** minute before both the alert and anchor. Raw session response files and hashes are under `../underlying_proxy_prices_20260922_06/`. They are retrospective trade bars, not bid/ask quotes, option marks, fills, or proof that a subscriber could execute. A positive underlying movement can still be a losing long option after spread, theta, implied-volatility change, and an earlier unobserved exit.

## What distinguishes the observed favorable cohort

`../underlying_proxy_features_20260922_01/summary.json` compares the 51 management-proxy rows at alert time, using only completed bars before the alert. The observed favorable rows show higher median direction-adjusted short-term momentum and volume:

| Pre-alert feature | Favorable median | Unfavorable median | Difference |
| --- | ---: | ---: | ---: |
| Direction-adjusted 5-minute return | +0.0467% | +0.0224% | +0.0243 pp |
| Direction-adjusted 20-minute return | +0.0999% | +0.0384% | +0.0615 pp |
| Last-5-min / prior-20-min volume | 1.196× | 0.868× | +0.327× |
| Direction-adjusted VWAP distance | +0.193% | +0.388% | −0.195 pp |

The practical first hypothesis is **directional momentum with fresh volume, often entered nearer VWAP/retest rather than after maximum extension**. The VWAP result is not a rule yet—the outcome set has only 10 losers and is selected on publicly posted management. It is a candidate gate to test, not a model to deploy.

These inputs already overlap the system:

- `strategies/intraday_structure/features.py` computes VWAP distance and one-/five-minute relative volume.
- `strategies/multi_ticker_swing/features/build_features.py` computes `dist_to_vwap` and relative-volume inputs.
- `scripts/discord_ledger/spy_trigger_pilot.py` already provides point-in-time 5-/20-minute return and VWAP features for SPY alerts.

So the gap is not “invent a new indicator.” It is to test a narrow ACE-style paper gate: option direction aligned with 5-/20-minute momentum, volume above its short reference, and entry sufficiently close to VWAP rather than chasing extension—then validate it on out-of-sample alert times and no-alert controls before integrating it into any live module.

## Reproduce

```bash
./.venv/bin/python -m scripts.discord_ledger.reconcile_underlying_proxy --run research/discord_ledger_2026-09-21/run_curated_20260921_05 --prices research/discord_ledger_2026-09-21/underlying_proxy_prices_20260922_06/snapshots.jsonl --out research/discord_ledger_2026-09-21/<fresh-proxy-directory>
./.venv/bin/python -m scripts.discord_ledger.analyze_underlying_proxy_features --ledger research/discord_ledger_2026-09-21/<fresh-proxy-directory>/underlying_proxy_ledger.jsonl --out research/discord_ledger_2026-09-21/<fresh-feature-directory>
```
