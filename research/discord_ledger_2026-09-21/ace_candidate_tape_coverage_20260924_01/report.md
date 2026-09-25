# ACE candidate-tape coverage audit

A valid historical candidate tape requires an immutable universe snapshot at or before each ACE decision and pre-decision intraday bars for every eligible symbol, in both directions. This audit makes no current-universe substitution and emits no fake candidate rows.

| Measure | Value |
| --- | ---: |
| Source alerts | 120 |
| PIT snapshot resolved | 6 |
| Snapshot plus bar file resolved | 6 |
| Full-universe, two-direction tape possible | 0 |
| Mean eligible-symbol bar coverage | 0.138% |
| Maximum eligible-symbol bar coverage | 0.172% |

## Conclusion

`retrospective_full_candidate_tape_not_available_from_current_inputs`. The existing session files may support alert-symbol analysis, but cannot be relabeled as an independent full-universe candidate tape unless the coverage manifest says every point-in-time eligible symbol is present.
