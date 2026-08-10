# Nervous system — MVP acceptance

What is proven, what is not, and what has to happen before this is pointed at
money. Read the "NOT PROVEN" section first; it is the part that decides
anything.

Run the evidence with:

```
python -m scripts.validate_nervous_system_mvp
```

## Scope, stated plainly

This is a **Meta Ranker cutover**, not a repository-wide one. Meta's execution
is governed end to end. HTF Swing, Momentum, Swing, Dealer Positioning, and SPY
Intraday still submit directly to the broker and are unchanged. Their bypasses
are inventoried in `test_mvp_acceptance.py` with pinned counts, so a new one
cannot appear unnoticed, and they are post-MVP migrations.

## PROVEN

**Nothing reaches a live account.** Production-live is refused in the settings,
in the policy engine, in the coordinator, in the gateway constructor, and in the
Meta router — five independent layers, each tested. The environment × mode ×
submit matrix is exhaustive over `{OFF, SHADOW, ENFORCE} × {submit, no submit}`
and every production-live cell returns `ENV_PRODUCTION_LIVE_DISABLED_MVP`.

**Meta submits nothing directly.** `live_runner.py`, `options_exec.py`,
`meta_ranker_dashboard.py`, and the adapter modules contain zero broker submit
calls; only the inward Alpaca adapter POSTs. Enforced by AST scan, because a
runtime test only proves the paths it happened to execute.

**Deployment is not authorisation.** Submission requires the exact string
`"true"`; anything fuzzy resolves to off.

**The decision path is reproducible.** The policy engine reads no clock and no
socket. Order identity is content-derived and anchored to the decision bar, so
a retried 4H pass converges instead of placing a second order — a bug this
suite caught rather than assumed.

**Exits fail operational.** A risk-reducing close is never blocked by a missing
quote; it degrades to a market order carrying the reason it is unpriced. An
unreadable market calendar defers entries but never blocks an exit.

**Outcomes are append-only.** A settled outcome is never re-measured, a
maturing horizon is `PENDING` rather than zero, and earlier revisions are left
exactly as they were.

**Unfit option sources cannot produce option P&L.** The gate runs before any
P&L, defaults to unfit, enforces per side, and treats trade prints, last
prices, synthetic, forward-filled, interpolated — and midpoints — as unfit
marks. The +0.09 correlation from the 2026-07 retraction is a test case.

## NOT PROVEN

These are the reasons this is not yet acceptance. None can be discharged by a
test run.

1. **Shadow soak has not been run.** The gate is ≥ 20 sessions and ≥ 100
   eligible Meta intents in QA-paper shadow with `submit=false`. Zero sessions
   have been run. Until this happens there is no evidence about how the policy
   behaves against real market data over time — only that it behaves correctly
   against the cases we thought to write.
2. **No controlled paper-submit subset has been run.** Entry caps and stop
   conditions are undefined.
3. **No QA infrastructure exists.** The Cloud SQL instance and the GCS journal
   bucket are specified in the runbook and not provisioned. Everything to date
   runs against local Docker PostgreSQL.
4. **Option source fitness has never been measured against a real
   entitlement.** The gate is implemented and tested against synthetic metrics.
   Whether Alpaca's option data actually passes it is unknown, and the honest
   prior — given the 2026-07 retraction — is that it may not.
5. **The persistent development database is two revisions behind** (`0002`;
   head is `0004`). Only the disposable test database has been upgraded.
6. **`AuditStore` is not wired into a running process.** The router, store, and
   transport are tested; nothing serves them yet.

## Promotion sequence

Do not skip steps, and do not run two at once.

1. Development/replay against the fake broker. ✅ done
2. QA-paper shadow, `submit=false`. ⬜ blocked on item 3 above
3. Shadow soak: ≥ 20 sessions, ≥ 100 eligible intents. ⬜
4. Controlled paper-submit subset with defined caps and stop conditions. ⬜
5. Enforce hard environment/readiness/idempotency/bounded-loss/liquidity/
   portfolio rules. ⬜
6. Contextual modifiers stay shadow-only. Promote one non-increasing rule at a
   time, each with its config hash and effective timestamp. ⬜

Before step 5, re-tier the Cloud SQL instance off shared-core: `db-f1-micro`
has no SLA, which is acceptable only while nothing can execute.

## Rollback

Disable entries, or move enforce to shadow or off. Risk-reducing exits, the
journal, and reconciliation all keep running — a rollback that also stops exits
would trap positions, which is worse than the problem it is solving.

**Rollback never reactivates a legacy direct-submit path.** The removed Meta
live route stays removed; `--live` and `--meta-ranker-live` continue to exit 2.

## Known limitations

- Repo-wide test failures: 28, all pre-existing and unrelated (options_lab
  fixtures, capstone result locks, an earnings-calendar network test,
  momentum_expansion). Verified byte-identical against a stashed baseline on
  every task.
- `test_live_postprocessing_covers_full_deployed_feature_manifest` fails in this
  worktree because ignored deployed model artifacts are absent. Unchanged
  throughout.
- The GCP migration tutorial says `us-central1` in 16 places while the bucket
  and the deployed service are in `us-east5`. The 2026-08-05 correction was
  applied to §4.5 only. Anyone following the rest will deploy split across two
  regions and pay egress on every bucket read.
