# Task 10 implementation report: durable theme-membership history and theme-state publication

## Scope

Implemented Task 10 on clean base `1f776009913ef29fc90c2f98f481e7ab80d5a7b0` in
`nervous-system-execution`. No Task 8/9 producer/importer behavior or
persistent `cynolycus` database was modified.

Files changed:

- `themes/dynamic_theme/config.py`
- `themes/dynamic_theme/stages/step08_memberships.py`
- `themes/dynamic_theme/pipeline.py`
- `themes/dynamic_theme/nervous_system_adapter.py`
- `themes/dynamic_theme/tests/test_nervous_system_adapter.py`
- `themes/dynamic_theme/tests/test_seed_and_stability.py`
- `core/nervous_system/contracts/enums.py`
- `core/nervous_system/contracts/states.py`
- `core/nervous_system/persistence/repositories/state.py`
- `core/nervous_system/tests/test_state_contracts.py`
- `LIVING_SUMMARY.md`

The nervous-system contract changes are limited to separating
`ThemeMembership` as `StateType.THEME_MEMBERSHIP`, retaining `ThemeState` as
`StateType.THEME`, and carrying sorted membership scores plus optional crowding,
persistence, and numeric feature metrics on `ThemeState`.

## TDD RED

Tests were added before the Task 10 production adapter/history implementation.

Exact command:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_nervous_system_adapter.py themes/dynamic_theme/tests/test_seed_and_stability.py -q
```

Exact result:

```text
==================================== ERRORS ====================================
__ ERROR collecting themes/dynamic_theme/tests/test_nervous_system_adapter.py __
ImportError while importing test module '/home/luket/repos/CynolycusBot/.worktrees/nervous-system-execution/themes/dynamic_theme/tests/test_nervous_system_adapter.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
themes/dynamic_theme/tests/test_nervous_system_adapter.py:14: in <module>
    from themes.dynamic_theme.nervous_system_adapter import (
E   ModuleNotFoundError: No module named 'themes.dynamic_theme.nervous_system_adapter'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
EXIT_CODE=2
```

## Implementation

- Added atomic same-directory Parquet replacement for both the compatibility
  membership view and append-preserving history.
- Added immutable history columns exactly as required:
  `as_of`, `available_at`, `generated_at`, `ticker`, `theme`,
  `membership_score`, `taxonomy_version`, `producer_version`.
- Same-date/same-taxonomy keys converge without changing existing rows;
  revised taxonomy versions append new evidence.
- Added canonical semantic theme IDs and taxonomy hashing over sorted semantic
  IDs, sorted seed members, embedding model, and clustering parameters. Paths,
  timestamps, insertion order, and cluster numbering are excluded; ephemeral
  `cluster_<number>` IDs fail closed.
- `available_at` is captured as a timezone-aware UTC completion timestamp;
  `generated_at` remains a separate supplied timestamp.
- Added `adapt_theme_states` with deterministic UUID5 state IDs, exact lineage
  validation, sorted score preservation, optional metrics, warning-quality
  states for missing inputs, `ThemeRegime.UNKNOWN`, and no probability
  inference.
- Added optional post-Parquet publication through the caller-owned UOW. The
  adapter never commits or rolls back; publication failures propagate and do
  not remove or rewrite completed Parquet artifacts.

## GREEN and verification

Focused Task 10 command:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_nervous_system_adapter.py themes/dynamic_theme/tests/test_seed_and_stability.py -q
```

Result:

```text
22 passed, 2 warnings
EXIT_CODE=0
```

Full theme suite:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests -q
42 passed, 2 warnings
EXIT_CODE=0
```

The two warnings are the existing `FutureWarning` from
`step05_claude_labeling.py` registry concatenation; no warning originates from
the new history or adapter implementation.

Relevant nervous-system contract/repository regression:

```text
./.venv/bin/python -m pytest core/nervous_system/tests/test_state_contracts.py core/nervous_system/tests/test_state_repository.py -q
37 passed, 6 skipped
EXIT_CODE=0
```

Full nervous-system suite:

```text
./.venv/bin/python -m pytest core/nervous_system/tests -q
190 passed, 31 skipped
EXIT_CODE=0
```

The skipped tests require the configured disposable PostgreSQL test URL.

Compilation and diff checks:

```text
./.venv/bin/python -m compileall -q themes/dynamic_theme core/nervous_system
git diff --check
EXIT_CODE=0 for both commands
```

## Deterministic cross-seed evidence

The same semantic registry was run with different insertion order, cluster
numbers, paths, timestamps, and Python hash seeds:

```text
PYTHONHASHSEED=1 ./.venv/bin/python -c '...compute_taxonomy_version(...)...'
taxonomy:bf66c5f53e7b68adb7f5c45a754ebe6223b94f533863273201aff44e24703f5f

PYTHONHASHSEED=2 ./.venv/bin/python -c '...compute_taxonomy_version(...)...'
taxonomy:bf66c5f53e7b68adb7f5c45a754ebe6223b94f533863273201aff44e24703f5f
```

The test suite also proves same-date idempotency, revised-taxonomy history,
atomic replacement, exact timestamps/schema, compatibility-view parity,
missing-input warnings, score/metric preservation, UNKNOWN/no-probability
behavior, exact lineage validation, caller-owned UOW behavior, persistence
failure propagation, deterministic state IDs, and no ephemeral cluster IDs.

## Self-review

- Compatibility output remains four columns (`ticker`, `theme`,
  `membership_score`, `date`) and is written as the latest current view; its
  history is separate.
- Durable identity never uses `cluster_id`, row number, local path, mtime, or
  runtime timestamp. State IDs use semantic theme/date/availability/taxonomy
  and exact lineage material.
- Adapter lineage requires a non-empty source ID, exact 64-character SHA-256
  artifact hash, and original row locator. Pipeline publication hashes the
  completed Parquet artifacts before state adaptation.
- Research execution has no UOW path and does not open a database. An
  orchestrated UOW path requires an explicit validity policy and leaves the
  UOW transaction boundary to its caller.
- No live/paper order path, source artifact, Task 8/9 producer/importer
  behavior, or persistent database was touched.

## Concerns

- PostgreSQL-backed persistence was not exercised because this worktree has no
  configured disposable nervous-system test URL; offline UOW behavior and all
  existing core tests passed.
- The full theme suite retains two pre-existing `step05_claude_labeling.py`
  `FutureWarning`s.
- A deliberate database migration/import decision would be required before
  reading any historical rows that were previously persisted as generic
  `THEME` `ThemeMembership` records; no such migration was run in this task.

## Correction round 1/5

### RED evidence

The new regression tests were added before the correction implementation. The
focused Task 10 run failed on the expected old behavior:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_nervous_system_adapter.py -q
7 failed, 13 passed
EXIT_CODE=1
```

The first required PostgreSQL attempt used the exact disposable URL but the
pre-existing `postgresql://` fixture had no psycopg2 bridge installed:

```text
NERVOUS_SYSTEM_TEST_DATABASE_URL='postgresql://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus_nervous_system_test' ./.venv/bin/python -m pytest core/nervous_system/tests/test_state_repository.py -q
2 passed, 7 errors — ModuleNotFoundError: psycopg2
EXIT_CODE=1
```

After installing the test-only `psycopg2-binary` bridge, the same command
reached the database and reported the unavailable listener (`2 passed, 7
errors`, `OperationalError`). The existing disposable PostgreSQL container
was then verified healthy and the integration run was repeated outside the
sandbox.

### GREEN evidence

The correction implements all six review findings:

- State IDs now use stable semantic theme IDs, `as_of`, taxonomy/version
  material, and lineage content hashes plus path-free row locators; runtime
  completion timestamps and local source paths are excluded. Artifact-hash or
  row-locator revisions produce new IDs, while the Task 9 atomic idempotent
  repository path remains the publication path.
- Pipeline publication captures UTC feature completion immediately after Step
  9 returns, and passes `max(membership_available_at, feature_completion_at)`
  to validity calculation and the adapter. Mtime is not consulted.
- Legacy `THEME` rows with the explicit `(ticker, theme, membership_score)`
  shape are hash/relationally validated before a copied
  `THEME_MEMBERSHIP` contract is returned. THEME_MEMBERSHIP queries and
  snapshots include only that legacy shape; generic THEME queries exclude it.
- Raw theme labels remain in compatibility/history/features joins. Semantic
  IDs carry a stable SHA-256 suffix and collision checks; `AI & ML` and
  `AI/ML` remain distinct, and ephemeral cluster labels are rejected.
- Schema-less empty memberships produce feature-identified warning states
  with empty scores and `MISSING_MEMBERSHIPS`.
- Existing and incoming history timestamps must be timezone-aware and obey
  `as_of <= generated_at <= available_at`; ThemeState/ThemeMembership causal
  validation follows the same direction.

Focused Task 10 suite:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_nervous_system_adapter.py -q
20 passed
EXIT_CODE=0
```

Full theme suite:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests -q
48 passed, 2 warnings
EXIT_CODE=0
```

Focused nervous-system contract/repository suite with the exact disposable
PostgreSQL URL:

```text
NERVOUS_SYSTEM_TEST_DATABASE_URL='postgresql://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus_nervous_system_test' ./.venv/bin/python -m pytest core/nervous_system/tests/test_state_contracts.py core/nervous_system/tests/test_state_repository.py -q
44 passed
EXIT_CODE=0
```

The legacy PostgreSQL round-trip/query/snapshot/tamper fixture is included in
that result. Full relevant nervous-system suite:

```text
NERVOUS_SYSTEM_TEST_DATABASE_URL='postgresql://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus_nervous_system_test' ./.venv/bin/python -m pytest core/nervous_system/tests -q
222 passed
EXIT_CODE=0
```

Cross-seed deterministic taxonomy evidence (same output for seeds 1 and 777):

```text
1 taxonomy:ad5a398a3e021d914a7f9b61f4ce998c3831104a0f81fb9f41266b8e788e7c55
777 taxonomy:ad5a398a3e021d914a7f9b61f4ce998c3831104a0f81fb9f41266b8e788e7c55
EXIT_CODE=0
```

Compilation and diff checks:

```text
./.venv/bin/python -m compileall -q themes/dynamic_theme core/nervous_system
git diff --check
EXIT_CODE=0 for both commands
```

### Correction files

- `themes/dynamic_theme/stages/step08_memberships.py`
- `themes/dynamic_theme/nervous_system_adapter.py`
- `themes/dynamic_theme/pipeline.py`
- `core/nervous_system/contracts/states.py`
- `core/nervous_system/persistence/repositories/state.py`
- `themes/dynamic_theme/tests/test_nervous_system_adapter.py`
- `core/nervous_system/tests/test_state_repository.py`

### Correction self-review and concerns

- Compatibility output remains the four-column latest view; history remains a
  separate exact-schema append-preserving artifact with immutable same-date
  convergence and revised-taxonomy evidence.
- State identity contains no runtime `available_at`, generated timestamp,
  source path, mtime, cluster number, or row-number-only fallback. Exact
  source hashes and original locators remain in lineage payloads.
- No transaction ownership was added: publication still propagates UOW
  persistence errors without commit/rollback, while Parquet artifacts remain
  present after failure. Research runs remain DB-free.
- No Task 8/9 producer/importer files, plan/ledger files, or persistent
  `cynolycus` database were modified. The shared contract/repository changes
  are limited to the required Task 10 theme semantics and legacy read path.
- The full theme suite retains two pre-existing `step05_claude_labeling.py`
  `FutureWarning`s. No new warning or test failure remains.

## Correction round 2/5

### RED evidence

The new theme regressions were added before production changes. The focused
run failed on each reviewed behavior:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_nervous_system_adapter.py -q
5 failed, 20 passed
EXIT_CODE=1
```

The failures showed availability captured before history normalization,
historical publication selecting the global latest date, absent current
taxonomy attrs and duplicate exact evidence being accepted, and weekly
publication omitting `represented_as_of`.

The PostgreSQL regression was run against the exact disposable URL:

```text
NERVOUS_SYSTEM_TEST_DATABASE_URL='postgresql://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus_nervous_system_test' ./.venv/bin/python -m pytest core/nervous_system/tests/test_state_repository.py -q
1 failed, 9 passed
EXIT_CODE=1
```

The explicit `membership_scores: null` fixture was incorrectly selected by a
`THEME_MEMBERSHIP` query and then failed `ThemeState` reconstruction. This
confirmed that JSON value extraction conflated an absent key with JSON null.

### GREEN evidence

- Daily and weekly runs now pass their represented `as_of` into publication.
  Publication preserves the caller's represented calendar date, requires the
  taxonomy version from current membership attrs, and selects history and
  features only for that exact date and taxonomy. Missing evidence and
  duplicate exact keys fail clearly; global latest history is never selected.
- The historical rerun regression preloads a newer date and proves the older
  run publishes only its own date, taxonomy, feature rows, exact Parquet
  hashes, and original row locators.
- Legacy membership SQL now uses PostgreSQL JSONB key-existence predicates for
  all required legacy keys and requires aggregate `membership_scores` to be
  absent. True legacy rows remain readable; explicit-null, malformed, and
  generic aggregate rows are not routed to legacy conversion. Valid generic
  `THEME` query and snapshot behavior is preserved.
- History captures `available_at` only after current-input validation and
  existing-history normalization, immediately before materializing immutable
  rows for serialization/publication. The clock-controlled test proves the
  order and exact timestamp.

Final focused Task 10 suite:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_nervous_system_adapter.py -q
26 passed
EXIT_CODE=0
```

Focused repository suite with PostgreSQL:

```text
NERVOUS_SYSTEM_TEST_DATABASE_URL='postgresql://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus_nervous_system_test' ./.venv/bin/python -m pytest core/nervous_system/tests/test_state_repository.py -q
10 passed
EXIT_CODE=0
```

Full theme suite:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests -q
54 passed, 2 warnings
EXIT_CODE=0
```

Full nervous-system suite with PostgreSQL:

```text
NERVOUS_SYSTEM_TEST_DATABASE_URL='postgresql://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus_nervous_system_test' ./.venv/bin/python -m pytest core/nervous_system/tests -q
223 passed
EXIT_CODE=0
```

Cross-seed/path/order/timestamp taxonomy evidence:

```text
PYTHONHASHSEED=1
taxonomy:fbc6a6fa7ccaadcf2e798b8d19663a9e65c77b9030660b27e75511fd4867265c

PYTHONHASHSEED=777
taxonomy:fbc6a6fa7ccaadcf2e798b8d19663a9e65c77b9030660b27e75511fd4867265c
EXIT_CODE=0 for both commands
```

Compilation and diff checks:

```text
./.venv/bin/python -m compileall -q themes/dynamic_theme core/nervous_system
git diff --check
EXIT_CODE=0 for both commands
```

### Correction files

- `themes/dynamic_theme/pipeline.py`
- `themes/dynamic_theme/stages/step08_memberships.py`
- `core/nervous_system/persistence/repositories/state.py`
- `themes/dynamic_theme/tests/test_nervous_system_adapter.py`
- `core/nervous_system/tests/test_state_repository.py`

### Self-review and concerns

- Structured review found no remaining Critical, Important, or Minor finding
  against the three round-2 requirements or the six prior closed findings. A
  separate reviewer subagent was unavailable in this session.
- The history `available_at` semantic boundary is the instant after required
  current/prior evidence validates and immediately before serialization. A
  Parquet artifact cannot record its own post-rename completion time without a
  destructive second rewrite, so no mtime or post-write rewrite is used.
- Explicit-null or malformed generic `THEME` payloads remain fail-closed under
  generic contract reconstruction; they are never reinterpreted as legacy
  memberships. Valid aggregate `THEME` rows query and snapshot normally.
- The only warnings are the two pre-existing pandas `FutureWarning`s in
  `step05_claude_labeling.py`. No Task 8/9, plan/ledger, persistent database,
  or unrelated theme clustering/model behavior was changed.

## Correction round 3/5

### RED evidence

The Step 9 schema/round-trip and exact feature-taxonomy publication
regressions were added before production changes:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_nervous_system_adapter.py -q
9 failed, 26 passed
EXIT_CODE=1
```

The failures proved that normal and empty Step 9 outputs omitted taxonomy,
same-date taxonomy revisions were discarded, missing/conflicting membership
taxonomy was accepted, and publication combined or accepted wrong, absent,
missing, conflicting, and duplicate feature-taxonomy evidence.

### GREEN evidence

- Step 9 derives one exact taxonomy version from current membership evidence,
  rejects missing/ambiguous/conflicting evidence, and writes an explicit
  `taxonomy_version` column in both normal and empty output schemas.
- Normal and empty Parquet round trips retain the taxonomy-bearing schema.
  Same-date revisions are keyed by `(date, taxonomy_version)`, so rerunning one
  taxonomy replaces only its own feature evidence and preserves the other.
- Publication validates feature taxonomy attrs when present, requires explicit
  taxonomy on every nonempty feature row, selects only the exact run date and
  current membership taxonomy, and rejects an absent exact match or duplicate
  `(date, ticker, taxonomy_version)` evidence.
- A same-date `taxonomy-v2` plus `taxonomy-v1` artifact publishes only the v1
  rows and their original Parquet row locators for v1 memberships. A same-date
  artifact containing only v2 fails closed for v1 memberships.

Focused Task 10 suite:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_nervous_system_adapter.py -q
35 passed
EXIT_CODE=0
```

Full theme suite:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests -q
63 passed, 2 warnings
EXIT_CODE=0
```

Full nervous-system suite with the exact disposable PostgreSQL URL:

```text
NERVOUS_SYSTEM_TEST_DATABASE_URL='postgresql://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus_nervous_system_test' ./.venv/bin/python -m pytest core/nervous_system/tests -q
223 passed
EXIT_CODE=0
```

Compilation and diff checks:

```text
./.venv/bin/python -m compileall -q themes/dynamic_theme core/nervous_system
git diff --check
EXIT_CODE=0 for both commands
```

### Correction files

- `themes/dynamic_theme/stages/step09_meta_features.py`
- `themes/dynamic_theme/pipeline.py`
- `themes/dynamic_theme/tests/test_nervous_system_adapter.py`

### Self-review and concerns

- Review found no remaining Critical, Important, or Minor issue against the
  sole round-3 finding or prior Task 10 closures. A reviewer subagent was not
  available, so the review checklist was applied directly to the full diff.
- Existing nonempty Step 9 Parquet without explicit taxonomy now fails closed
  instead of being silently mixed or overwritten. Regenerating that artifact
  from taxonomy-bearing Step 8 evidence is required; no synthetic migration or
  fallback was introduced.
- Raw labels and joins remain unchanged. Taxonomy is an identity field, not a
  score/probability; state identity and lineage behavior are unchanged.
- Research mode still returns before publication without a UOW, and no commit
  or rollback ownership was added. No Task 8/9, plan/ledger, persistent
  database, or unrelated clustering/model behavior was modified.
- The full theme suite retains only the two pre-existing pandas
  `FutureWarning`s in `step05_claude_labeling.py`.

## Correction round 4/5

### RED evidence

The empty-run artifact and exact-key duplicate regressions were added before
production changes:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_nervous_system_adapter.py -q -k 'step9_empty_run or step9_rejects_conflicting_existing_duplicate or step9_deduplicates_identical_existing or step9_rejects_conflicting_incoming'
4 failed, 1 passed, 35 deselected
EXIT_CODE=1
```

The failures proved that an empty run did not inspect a legacy feature
artifact, conflicting existing and current `(date, ticker, taxonomy_version)`
rows did not fail closed, and an identical existing duplicate survived a
same-date taxonomy-revision append. The valid revisioned-artifact preservation
control passed before implementation.

### GREEN evidence

- Every empty Step 9 run now reads and validates an existing feature artifact
  before returning. Missing, null, blank, non-canonical, or conflicting
  identity evidence raises without writing; a valid revisioned artifact is
  preserved byte-for-byte. With no artifact, Step 9 still writes the exact
  empty taxonomy-bearing schema and returns the current taxonomy in attrs.
- Current and existing feature rows are validated before evidence filtering or
  replacement. Conflicting duplicate exact keys raise with the offending key;
  rows whose every persisted field is identical are deterministically reduced
  to their first occurrence before a nonempty run is serialized.
- Conflict regressions verify no artifact mutation. The current-row conflict
  regression verifies no artifact is created. Existing same-date v1/v2
  revision preservation and exact publication selection remain covered.

Focused Task 10 suite:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_nervous_system_adapter.py -q
40 passed
EXIT_CODE=0
```

Full theme suite:

```text
./.venv/bin/python -m pytest themes/dynamic_theme/tests -q
68 passed, 2 warnings
EXIT_CODE=0
```

The first sandboxed PostgreSQL invocation was unable to open localhost and
ended `191 passed, 1 failed, 31 errors`; every failure/error was an initial
`psycopg2.OperationalError` connection denial. The exact command was rerun
outside that network sandbox against only the requested disposable database:

```text
NERVOUS_SYSTEM_TEST_DATABASE_URL='postgresql://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus_nervous_system_test' ./.venv/bin/python -m pytest core/nervous_system/tests -q
223 passed
EXIT_CODE=0
```

Compilation and diff checks:

```text
./.venv/bin/python -m compileall -q themes/dynamic_theme core/nervous_system
git diff --check
EXIT_CODE=0 for both commands
```

### Correction files

- `themes/dynamic_theme/stages/step09_meta_features.py`
- `themes/dynamic_theme/tests/test_nervous_system_adapter.py`

### Self-review and concerns

- Direct review found no remaining Critical, Important, or Minor issue against
  either round-4 finding or prior Task 10 closures. The requested code-review
  skill's subagent capability was unavailable, so its scope, architecture,
  testing, compatibility, and production-readiness checklist was applied
  directly to the complete diff.
- Empty runs deliberately never rewrite an existing valid artifact. Identical
  duplicates are validated as field-identical but remain byte-preserved on an
  empty run; the next nonempty serialization deterministically removes them.
- The two warnings are the pre-existing pandas `FutureWarning`s in
  `step05_claude_labeling.py`. The initial PostgreSQL failure was isolated to
  sandbox network denial; the unrestricted disposable-database run passed.
- No pipeline publication, state identity, lineage, raw-label joins, UOW,
  Task 8/9, plan/ledger, or persistent `cynolycus` behavior was changed.
