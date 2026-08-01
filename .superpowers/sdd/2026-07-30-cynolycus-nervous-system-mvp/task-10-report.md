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
