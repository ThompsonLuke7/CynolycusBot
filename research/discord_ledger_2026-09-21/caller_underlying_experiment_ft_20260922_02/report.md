# Caller underlying-direction experiment

This is an underlying-only, research-only result. It does not estimate option fills, option P&L, or prove a trade was executable.

Controls are same-caller/symbol/session five-minute directional candidates at least 30 minutes from an exported entry alert. Outcome comparisons weight their directions to the source-cited alert mix within each caller/symbol/session. They are not verified losing trades.

## FT

| Test metric | 20m | 60m |
| --- | ---: | ---: |
| Alert n | 16 | 15 |
| Alert mean directional return (%) | -0.0527 | -0.2193 |
| Alert directional hit rate | 0.5625 | 0.4000 |
| Control mean directional return (%) | -0.0051 | 0.0742 |
| Alert − control mean (%) | -0.0476 | -0.2935 |

Fixed alert-imitation model status: `fit_fixed_logistic_regression`. Test ROC AUC: `0.7347756410256411`; test average precision: `0.13231631255276427`.

## Interpretation boundary

A positive alert-minus-control result is evidence that the exported alerts had more favorable underlying movement than matched quiet windows in this sample. It is not evidence that a learned rule will remain profitable or that historical option pricing would have matched the caller. Any strategy recreation must be validated on a later, untouched time period with realistic execution assumptions.
