# Morning start — 2026-08-05

## 1. Verify readiness stamped overnight (do this FIRST)

```bash
cd /home/luket/repos/CynolycusBot
PYTHONPATH=. .venv/bin/python -c "
from datetime import datetime; from zoneinfo import ZoneInfo
import core.live_readiness as lr
ok,reason,_=lr.readiness_status(now=datetime.now(ZoneInfo('America/New_York')))
print('OK' if ok else 'BLOCKED', '-', reason)"
```

* **OK** -> skip to step 2.
* **BLOCKED** -> readiness did not finish. Rerun it, capped:

```bash
ALLOW_LIVE_READINESS=1 FORCE_DATA_READINESS=1 systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --quiet bash scripts/nightly_data_readiness.sh
```

`ALLOW_LIVE_READINESS=1` is required after 07:45 ET — the script refuses to run
inside the live window without it. Takes ~45 min (stage 1 bars ~30 min, stage 4
build ~13 min). If it is already past ~08:45, start the server anyway and accept
that the 4H modules will not open new positions today; exits still work.

## 2. Start the server — by 09:00 ET

```bash
scripts/run_live_server.sh
```

No flags needed: the launcher already supplies --start-all,
--intraday-structure, readiness 22:15, nightly 16:45, dealer-ranker 15:45, and
stays on PAPER (LIVE=1 is what would change that).

**Why 09:00 and not 09:25:** swing warms up ~925 tickers (~10 min), and the
pre-open flush loops fire at 09:35 / 09:37 ET. Five entries are queued from
yesterday and will only be submitted if the server is up for those loops:

| Module | Queued |
|---|---|
| Meta ranker | EQPT |
| Momentum | NEXA |
| HTF swing | CLSK260821C00015000, ICHR, HUT |

## 3. Confirm it is healthy

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8764/     # expect 200
tail -F logs/live_server/server_$(date +%Y%m%d).log
```

## Notes

* Branch must be `main` — `run_live_server.sh` runs whatever is checked out, and
  the 4H runners are fresh subprocesses that pick up the tree live. Check with
  `git rev-parse --abbrev-ref HEAD`.
* systemd install (survives terminal close / VS Code exit) is still pending:
  `scripts/systemd/install.sh` then `sudo loginctl enable-linger luket` then
  `systemctl --user start cynolycus-live`. Stop any hand-started server first —
  two supervisors would fight over the same ports and broker account.
* Still awaiting your decision: purge the test-fixture rows from
  `Data/inference/dealer_ranker/closed_trades.jsonl` (40 fake vs 3 real) and
  `Data/inference/meta_ranker/closed_trades.jsonl` (9 fake vs 3 real).
