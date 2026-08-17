# Nervous system — operations runbook

Covers the QA-paper deployment: what it runs on, how to bring it up, and what
to check when something looks wrong. Production-live is out of scope by
construction — it parses for read-only inspection and returns the stable veto
`ENV_PRODUCTION_LIVE_DISABLED_MVP` instead of executing.

## Target shape

| Component | Choice | Why |
|---|---|---|
| Database | Cloud SQL PostgreSQL **16**, Enterprise, `db-f1-micro`, zonal | Matches `compose.nervous-system.yaml` (`postgres:16`). Version parity is the cheapest bug prevention available: the same models and migrations run in both places. |
| Storage | 10 GB SSD | See sizing below. |
| Region | **`us-east5`** | `gs://cynolycusbot-data` is US-EAST5 and `hello-test` deployed there. Same-region reads are free; cross-region reads bill as egress. |
| Journal | **Its own bucket**, us-east5, uniform access, versioning on, Standard | Retention policies are bucket-wide, not per-prefix. |
| Submission | `CYNOLYCUS_SUBMIT_ENABLED` unset | Deployment is not authorisation. |

### Sizing

`UI/swing_audit` is ~351 MB over 99 days — roughly 3.5 MB/day of verbose JSONL.
The ledger is structured rows rather than JSONL, and Meta produces at most ~20
decisions a day, so single-digit GB/year is the right order of magnitude. This
is a small database, and `db-f1-micro` is sized for it.

**Shared-core has no SLA and no committed-use discount.** That is acceptable
only because this build cannot execute against a live account. Re-tiering to
`db-g1-small` or `db-custom-1-3840` is a **gate on enabling production-live**,
not a later optimisation.

If `import-history` crawls on `f1-micro`, scale the tier up for the import and
back down afterwards — a tier change is a restart, not a rebuild.

### Why not Postgres on the Phase 8 VM

It would cost nothing extra, and that is the wrong trade. On 2026-06-26 the
combined server was OOM-killed at 09:36 ET — 13.9 GB anon-RSS against a 16 GB
cap — and nothing restarted it, so every afternoon loop silently did not run.
Colocating the ledger with the process that OOMs means the ledger dies exactly
when it is needed to explain what happened, which is the entire reason it
exists. An independent failure domain is worth ~$8/month. Backups and PITR
would also become yours to own.

## Bring-up

Every command is non-destructive. There is no downgrade, drop, or reset in the
tooling, by design.

```
export CYNOLYCUS_ENVIRONMENT=QA_PAPER
export CYNOLYCUS_NERVOUS_SYSTEM_MODE=SHADOW
export CYNOLYCUS_DATABASE_URL='postgresql+psycopg://USER:PASS@/cynolycus?host=/cloudsql/PROJECT:us-east5:INSTANCE'
export CYNOLYCUS_OPERATIONAL_ROOT=/var/cynolycus
export CYNOLYCUS_EXECUTION_JOURNAL=gcs
export CYNOLYCUS_EXECUTION_JOURNAL_BUCKET=cynolycusbot-execution-journal
export CYNOLYCUS_ACCOUNT_ALIAS=paper
export CYNOLYCUS_GCP_PROJECT=PROJECT
export CYNOLYCUS_CLOUD_SQL_INSTANCE=PROJECT:us-east5:INSTANCE
export CYNOLYCUS_ALPACA_BASE_URL=https://paper-api.alpaca.markets
export CYNOLYCUS_ALPACA_ACCOUNT_ID=PAXXXXXX
export CYNOLYCUS_SECRET_BINDING=projects/PROJECT/secrets/alpaca-paper

PYTHONPATH=. .venv/bin/python -m scripts.cloud.nervous_system_db create-database --dry-run
PYTHONPATH=. .venv/bin/python -m scripts.cloud.nervous_system_db create-database
PYTHONPATH=. .venv/bin/python -m scripts.cloud.nervous_system_db upgrade-schema
PYTHONPATH=. .venv/bin/python -m scripts.cloud.nervous_system_db schema-status
```

The DSN comes from the environment, never a command-line argument: a DSN on a
command line lands in shell history, in `ps` output, and in any CI log that
echoes its commands.

### Two DSN shapes are supported

Cloud Run reaches Cloud SQL over a **Unix socket** (`?host=/cloudsql/...`). The
VM that runs the schedulers and the persistent Alpaca WebSocket cannot lift
into Cloud Run, so it will reach the same instance over **private-IP TCP**.
Both validate. Supporting one now and the other later would be a rewrite.

## Journal bucket

Separate from `cynolycusbot-data`, and this is load-bearing. A retention policy
applies to a whole bucket, so putting one on the data bucket would block
overwrites on all ~108k objects and break the nightly rsync of `features_4h`,
`news_embeddings`, and `Data/shared/bars`.

**Do not lock the retention policy yet.** A locked policy can never be
shortened or removed, and the bucket cannot be deleted until every object ages
out. Set it unlocked, run a full QA-paper cycle, then lock it once the duration
is settled. Standard class with no aging rules — Nearline's 30-day minimum
duration is a trap for small append-only objects.

## Checks

```
PYTHONPATH=. .venv/bin/python -m scripts.cloud.nervous_system_db schema-status
PYTHONPATH=. .venv/bin/python -m scripts.cloud.nervous_system_db verify-counts
PYTHONPATH=. .venv/bin/python -m scripts.cloud.nervous_system_db verify-backup
curl -s localhost:PORT/api/nervous-system/health | jq
```

`verify-backup` reports `UNVERIFIED` unless the Cloud SQL Admin API says
otherwise. It does not infer a backup from anything else: an unverified
"probably backed up" is worse than a clear "unknown", because only one of them
makes somebody go and look.

Health separates **liveness** (the process is up; answers without a database)
from **readiness** (the system can do its job). A reachable-but-unready system
returns 503, so a degraded journal cannot hide inside a 200.

## Open items

- ~~The migration tutorial says `us-central1` in 16 places against 2
  `us-east5`.~~ **Fixed 2026-08-17** on `main`: §1.6 now reads `us-east5`, every
  operational reference and gcloud snippet follows it, and the two remaining
  mentions are the superseded decision-log row and factual notes about which
  regions carry the GCS free tier. `docs/GCP_MIGRATION_TUTORIAL.md` §3B now
  covers Cloud SQL, the journal bucket and both DSN shapes, so this runbook and
  the tutorial no longer disagree.
- The persistent local `cynolycus` database is at `0002_decision_execution`;
  head is `0004_audit_observability` (measured 2026-08-17). Run
  `upgrade-schema` before the first governed pass — the replay, fitness, audit
  and observability tables do not exist at `0002`.
- Not yet run: the shadow soak (0 of ≥20 sessions), the controlled
  paper-submit subset, and option source fitness against a real entitlement.
  See `MVP_ACCEPTANCE.md`.
