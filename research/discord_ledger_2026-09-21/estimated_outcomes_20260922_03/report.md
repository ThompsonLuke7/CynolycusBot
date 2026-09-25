# ACE managed-trade price reconciliation (2026-09-22)

This is an estimate-based reconciliation of the supplied Discord export, **not** a broker P&L statement. It combines the source-cited ACE entry lifecycles with text exits, image-management posts, and OCR. The per-trade evidence and source IDs are in `estimated_trade_outcomes.jsonl`; unassigned loss posts are in `unlinked_reported_losses.jsonl`. Prices are per option contract, before fees and spreads. A price change is `(later reference / alert reference − 1) × 100`, **not** account return or a follower's fill.

## What the evidence supports

Of 222 ACE entry lifecycles, 11 have a usable close-price reference under the stated rules: 10 positive and one negative. This **10/11 is coverage of selected, resolvable examples, not an overall win rate**. Only WMT has a complete source-claimed quantity/price cycle. Two rows have a reported exit price but unresolved quantity; one combines a reported sale price with a screenshot showing zero new position; two use a filled-sell screenshot's limit as a lower bound; five use the option's displayed price at an explicit full-close post as an approximate mark, not a sale fill.

| Entry date (UTC) | Underlying | Alert → close reference | Price change | Basis | Source message IDs (entry; management) |
| --- | --- | ---: | ---: | --- | --- |
| Feb 2 | WMT | 0.80 → 1.70 / 1.95 | +128.1% | Claimed 2-contract cycle | `1467893759936037009`; `1467929148705083708`, `1467958821623304334` |
| Feb 6 | QQQ | 0.37 → 0.25 | −32.4% | Text exit; quantity unknown | `1469372524998426686`; `1469376239323316235` |
| Mar 3 | SPY | 0.38 → 1.84 | +384.2% | Image mark at reported full close | `1478410779744010362`; `1478462233985159309` |
| Mar 4 | SPY | 2.00 → 6.04 | +202.0% | Image mark at reported full close | `1478846580194021529`; `1479154644243709962` |
| Jul 6 | IWM | 0.70 → ≥1.40 | ≥+100.0% | Filled-sell limit; new position zero | `1523688441626362016`; `1524064332474748929` |
| Jul 7 | SPY | 1.40 → ≥3.38 | ≥+141.4% | Filled-sell limit; new position zero | `1524141642997825546`; `1524407915677483129` |
| Jul 8 | ORCL | 1.45 → 1.70 | +17.2% | Text exit; quantity/add unresolved | `1524462462563258378`; `1524483945901719562` |
| Jul 8 | ORCL | 0.75 → 3.05 | +306.7% | Text sale + screenshot zero position | `1524492234848731278`; `1524771707095748729` |
| Sep 3 | MU | 2.40 → 19.20 | +700.0% | Image mark at reported full close | `1545072423198785576`; `1545427779737358517` |
| Sep 10 | AAPL | 1.50 → 5.13 | +242.0% | Image mark at reported full close | `1547688324146139197`; `1547964212561580074` |
| Sep 16 | SPX | 1.75 → 16.25 | +828.6% | Image mark at reported full close | `1549847505771040839`; `1549862816343658626` |

The WMT 2-contract claims imply 1.70 + 1.95 − 2 × 0.80 = **$2.05 per-share-equivalent, or $205 gross at the standard 100-share multiplier**, before costs, if all source claims describe actual fills. That is the only cycle here with claimed quantities sufficient for such a calculation. Screenshot limit prices are lower bounds on a sell fill *if the order screen's filled status and contract link are correct*; they do not establish the overall trade's weighted return after unpriced adds.

## Positive snapshots that are not closed-trade results

There are 26 ACE entries with uniquely contract-matched image-management candidates; 18 have a displayed option price comparable to the alert reference. Thirteen of those priced entries have **no** accepted full-close reference. All 28 observed price comparisons across these 18 entries are positive, but screenshots are posted selectively, so this is not a win-rate sample. For example, the Sep 11 INTC 108 call alert at 1.50 (`1548055480541384826`) has a Sep 17 “CALLS GONE” screenshot displaying 3.50 (`1550160332549525574`): **+133.3% alert-to-mark**, not a verified exit fill. The screenshot's “+845.95% Today” is a daily change, not this callout's return.

The ORCL Jul 8/9 sale post (`1524771707095748729`) says 3.05, while its order screen labels 3.00 as the *limit* and about 304.96 as estimated credit. These are not collapsed into one supposed broker fill; the discrepancy remains in the row-level evidence. Image contract matching is at visible underlying/strike/right and time; expiry or adjustments not visible in the image remain uncertain. All screenshot readings are source-reported, not independently broker-verified.

## Missing downside and profitability boundary

Four ACE messages with loss/stop wording could not be attached to a unique entry: `1534197033068400640`, `1534559471794393188`, `1534564389963960350`, `1537205782251180113`. The last is a mixed recap that includes a −20% SPX stop as well as winners. These are **messages, not four proven distinct trades**. Many other entries have no resolvable close, and seven VIP exports contain no messages. Consequently, the ledger does not yet support an overall profit total, average trade return, or population win rate; `estimated_trade_pnl` and `estimated_follower_pnl` remain null rather than filling missing trades with invented sizes or prices.

The evidence does show numerous large **reported favorable option-price moves**. Reproducing them requires a separate point-in-time entry/exit opportunity test: exact contract (including expiry), option bid/ask and size around alert availability, realistic follower delay, and the full loss distribution. Historical underlying bars alone cannot establish follower option fills.

## Reproduce

```bash
./.venv/bin/python -m scripts.discord_ledger.reconcile_estimated_outcomes --study research/discord_ledger_2026-09-21 --run research/discord_ledger_2026-09-21/run_curated_20260921_05 --strategy-audit research/discord_ledger_2026-09-21/strategy_audit_20260921_02 --image-recovery research/discord_ledger_2026-09-21/image_management_recovery_20260921_03 --out research/discord_ledger_2026-09-21/<fresh-estimate-directory>
./.venv/bin/python -m pytest scripts/discord_ledger/tests -q
```
