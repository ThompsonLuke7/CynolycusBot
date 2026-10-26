# Existing Intraday Structure: ACE timing benchmark

The current price detectors and target/exit state machine were scanned independently of ACE alert times. Candidates in both directions refresh every 15 minutes on the observed test symbol/session panel. Execution is disabled.

| Tolerance | Source entries | Algorithm entries | Entry matches | Recall |
| --- | ---: | ---: | ---: | ---: |
| ±1 five-minute candles | 25 | 291 | 4 | 16.0% |
| ±2 five-minute candles | 25 | 291 | 5 | 20.0% |
| ±3 five-minute candles | 25 | 291 | 9 | 36.0% |

Entry and management confluence, shifted baseline, source coverage and complete config are in results.json. Bar-close availability is used for matching; archival CLOSED transitions cannot overwrite an earlier invalidation/exhaustion time. Historical comparison only.
