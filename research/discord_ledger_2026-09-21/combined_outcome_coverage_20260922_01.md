# Combined ACE and FT outcome coverage

The free-channel ledger now contains **369 entry lifecycles**: 222 ACE and 147 FT. The evidence has several non-interchangeable tiers:

| Evidence tier | ACE | FT | Combined | Meaning |
| --- | ---: | ---: | ---: | --- |
| Reported-management underlying proxies | 51 (41 favorable / 10 unfavorable) | 4 (3 / 1) | 55 (44 / 11) | Alert-to-reported-management underlying direction; closest available outcome proxy. |
| Unverified expiry-horizon proxies | 107 (44 / 57 / 6 flat) | 102 (45 / 54 / 3 flat) | 209 (89 / 111 / 9 flat) | Alert-to-stated-expiry underlying direction; does **not** claim the option was held. |
| Source-reported post-profit claims | 91 | 15 | 106 | Selective channel claims; 26 ACE claims uniquely link to earlier ACE entries; no FT claim uniquely does. |

Thus there are **264 underlying-direction observations** across ACE and FT, plus 106 caller-reported result-claim rows. There are not hundreds of broker-verified fills because the export does not contain them. Pooling these tiers into one win rate or one ML label would be invalid: the observation rules differ and FT management coverage is far thinner than ACE’s.

See `underlying_proxy_20260922_07/` for ACE, `underlying_proxy_ft_20260922_03/` for FT, and `ace_ft_pooling_assessment_20260922_01.md` for the separate-model decision.
