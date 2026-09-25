# Caller underlying-direction experiment

This is an underlying-only, research-only result. It does not estimate option fills, option P&L, or prove a trade was executable.

Controls are same-caller/symbol/session five-minute directional candidates at least 30 minutes from an exported entry alert. Outcome comparisons weight their directions to the source-cited alert mix within each caller/symbol/session. They are not verified losing trades.

## ACE

| Test metric | 20m | 60m |
| --- | ---: | ---: |
| Alert n | 20 | 14 |
| Alert mean directional return (%) | -0.0343 | 0.1123 |
| Alert directional hit rate | 0.5000 | 0.5000 |
| Control mean directional return (%) | 0.0131 | 0.0199 |
| Alert − control mean (%) | -0.0474 | 0.0925 |

Fixed alert-imitation model status: `fit_fixed_logistic_regression`. Test ROC AUC: `0.6473626373626374`; test average precision: `0.10304228838864983`.

## Interpretation boundary

A positive alert-minus-control result is evidence that the exported alerts had more favorable underlying movement than matched quiet windows in this sample. It is not evidence that a learned rule will remain profitable or that historical option pricing would have matched the caller. Any strategy recreation must be validated on a later, untouched time period with realistic execution assumptions.
