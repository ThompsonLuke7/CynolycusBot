# Task 0 implementation report

## Result

Repaired the catalyst fixture dependency leak while preserving production
defaults. `build_catalyst_records()` now exposes injectable
`earnings_result_features_path` and `earnings_result_labels_path` keyword
arguments, defaulting to the existing forward-guidance artifacts. Passing
`None` for either input disables the optional earnings-result source.

## Root cause

The builder accepted injected news, macro-event, and scheduled-earnings paths,
but unconditionally called `build_earnings_result_catalysts()` without passing
paths. That converter therefore read its process-global forward-guidance
features and labels, adding repository `earnings_result` rows to isolated
fixture tests.

## TDD evidence

1. Baseline command before the regression-test adjustment:

   ```text
   ./.venv/bin/python -m pytest -q signals/catalysts/tests/test_robustness.py signals/catalysts/tests/test_smoke.py
   ...                                                                      [100%]
   3 passed in 0.47s
   ```

   The isolated worktree does not contain the ignored forward-guidance parquet
   artifacts, so this local baseline did not reproduce the source-checkout
   data-pollution failures recorded in the SDD preflight.

2. RED after adding `None` inputs to the two fixture tests, before production
   code changed:

   ```text
   ./.venv/bin/python -m pytest -q signals/catalysts/tests/test_robustness.py signals/catalysts/tests/test_smoke.py
   2 failed, 1 passed in 0.38s
   TypeError: build_catalyst_records() got an unexpected keyword argument 'earnings_result_features_path'
   ```

3. GREEN focused suite:

   ```text
   ./.venv/bin/python -m pytest -q signals/catalysts/tests/test_robustness.py signals/catalysts/tests/test_smoke.py
   ...                                                                      [100%]
   3 passed in 0.44s
   ```

4. GREEN full catalyst suite:

   ```text
   ./.venv/bin/python -m pytest -q signals/catalysts/tests
   ...                                                                      [100%]
   3 passed in 0.40s
   ```

5. Default compatibility check verified that the new parameter defaults are
   the existing `FEATURES_PATH` and `LABELS_PATH` constants.

## Files changed

- `signals/catalysts/pipeline.py` — explicit forward-guidance path inputs and
  conditional optional-source inclusion.
- `signals/catalysts/tests/test_robustness.py` — disables global forward-guidance
  inputs in the empty-fixture regression test.
- `signals/catalysts/tests/test_smoke.py` — disables global forward-guidance
  inputs in the isolated records fixture.

## Self-review

- No-argument behavior remains compatible: both new arguments retain the prior
  process-global defaults, and the converter is still invoked when defaults
  are available.
- Explicit `None` prevents any read of the optional source; no fake paths or
  global monkeypatching are required by fixture callers.
- The change is limited to the requested pipeline and two tests; `git diff
  --check` passed and no unrelated files were modified.

## Commit

Implementation commit: `c3af94b` (`fix: isolate optional earnings result catalysts`).

## Remaining risk

The isolated worktree lacks the ignored forward-guidance artifacts, so a real
no-argument build against repository data could not be executed here. The
default-path signature check passed; the existing source-checkout artifacts
remain the production default when present.
