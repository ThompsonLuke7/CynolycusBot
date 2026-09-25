# ACE paired Intraday Structure control replay

Each ACE source alert was compared with one deterministic same-symbol, same-direction, same-session non-alert control. The control was assigned without outcomes; both sides were replayed independently for 60 minutes with execution disabled.

| Metric | Source alerts | Controls | Difference (pp) |
| --- | ---: | ---: | ---: |
| Setup detected | 96.7% | 100.0% | -3.3% |
| Explicit confirmation | 67.5% | 66.7% | 0.8% |
| Modeled share entry | 66.7% | 65.8% | 0.8% |

A positive difference would show that the current engine admits ACE-selected candidates more often than matched quiet windows. It remains conditional on ACE's ticker/direction choice and is not an options-P&L result or an independent universe-discovery test.
