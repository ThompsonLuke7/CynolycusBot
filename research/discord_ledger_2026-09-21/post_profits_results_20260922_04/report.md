# Supplemental `📈-post-ur-profits` results — ACE and FT

The supplemental export contains **838 messages** through 2026-09-22. It overlaps the earlier 664-message export by 662 identical message IDs and adds **174 new messages** (75 by ACE). Two overlapping non-caller message IDs differ between exports; the original source version remains canonical and both conflicts are recorded in `../source_post_profits_delta_20260922_01/same_id_conflicts.jsonl` rather than silently overwriting evidence.

The merged result index has **106 caller-reported claim rows**:

| Caller | Claim rows | Positive / negative / unsigned | Unique preceding-entry candidates |
| --- | ---: | ---: | ---: |
| ACE | 91 | 86 / 2 / 3 | 26 claims |
| FT | 15 | 13 / 0 / 2 | 0 claims |

The 26 uniquely linked ACE claims represent fewer distinct lifecycles because some posts update the same trade. FT claims have no uniquely matching prior free-channel entry in the export; they may refer to unexported VIP/Premium alerts or lack contract/time detail. None of these source-reported claims is broker P&L or a population win rate.

The parser creates one row per visible contract leg in a recap, including word forms such as “TSLA 415 put” as well as “TSLA 415P.” `caller_result_claims.jsonl` is the combined evidence index; `ace_result_claims.jsonl` and `ft_result_claims.jsonl` retain separate caller views.
