# Dealer Ranker Launcher Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore supervised server startup while preserving the Dealer Ranker's $5,000 target-notional entry sizing.

**Architecture:** Keep dollar sizing authoritative in `UI.combined_server` and align the shell supervisor with that interface. Protect the shell/Python boundary with a static regression test that does not launch dashboards or trading loops.

**Tech Stack:** Bash, Python 3, pytest, argparse

## Global Constraints

- Default Dealer Ranker entry target remains `$5,000`.
- Paper/live account routing is unchanged; the launcher must not be started during verification.
- Fixed-contract compatibility is intentionally not restored.

---

### Task 1: Align the supervised launcher with target-notional sizing

**Files:**
- Create: `UI/tests/test_live_server_launcher.py`
- Modify: `scripts/run_live_server.sh:77-81`

**Interfaces:**
- Consumes: Dealer Ranker CLI flags declared through `parser.add_argument(...)` in `UI/combined_server.py`.
- Produces: `--dealer-ranker-target-notional ${DEALER_RANKER_TARGET_NOTIONAL:-5000}` in the supervisor argument list.

- [x] **Step 1: Write the failing boundary test**

```python
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_live_server_dealer_ranker_flags_are_supported_by_combined_server() -> None:
    launcher = (REPO_ROOT / "scripts/run_live_server.sh").read_text()
    combined = (REPO_ROOT / "UI/combined_server.py").read_text()
    launcher_flags = set(re.findall(r'"(--dealer-ranker-[a-z-]+)"', launcher))
    parser_flags = set(
        re.findall(r'parser\.add_argument\(\s*"(--dealer-ranker-[a-z-]+)"', combined)
    )
    assert launcher_flags <= parser_flags, (
        "unsupported Dealer Ranker launcher flags: "
        + ", ".join(sorted(launcher_flags - parser_flags))
    )
```

- [x] **Step 2: Run the test and verify the current mismatch fails**

Run: `./.venv/bin/python -m pytest UI/tests/test_live_server_launcher.py -q`

Expected: FAIL containing `unsupported Dealer Ranker launcher flags: --dealer-ranker-contracts`.

- [x] **Step 3: Apply the minimal launcher fix**

Replace:

```bash
"--dealer-ranker-contracts" "${DEALER_RANKER_CONTRACTS:-1}"
```

with:

```bash
"--dealer-ranker-target-notional" "${DEALER_RANKER_TARGET_NOTIONAL:-5000}"
```

- [x] **Step 4: Verify the regression and adjacent interfaces**

Run: `./.venv/bin/python -m pytest UI/tests/test_live_server_launcher.py UI/tests/test_nightly_scheduler.py -q`

Expected: all selected tests pass.

Run: `bash -n scripts/run_live_server.sh`

Expected: exit 0 with no output.

Run: `./.venv/bin/python -m py_compile UI/combined_server.py UI/tests/test_live_server_launcher.py`

Expected: exit 0 with no output.

Run: `./.venv/bin/python -m UI.combined_server --help`

Expected: exit 0; output includes `--dealer-ranker-target-notional` and excludes `--dealer-ranker-contracts`.

- [x] **Step 5: Review the focused diff and update the living summary**

Run: `git diff --check` and `git diff -- scripts/run_live_server.sh UI/tests/test_live_server_launcher.py LIVING_SUMMARY.md`

Expected: no whitespace errors; diff contains only the sizing-boundary fix, regression test, and handoff entry.
