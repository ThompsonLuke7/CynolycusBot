# Intraday Structure Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the existing Intraday Structure dashboard at port 8774 in the shared combined-server navigation.

**Architecture:** `UI.ui_chrome.NAV_PORTS` is the single source for the JavaScript-generated navigation on every dashboard. Add the existing dashboard's label/port there; retain the Hub card and combined-server startup behavior unchanged.

**Tech Stack:** Python 3.12, pytest, server-rendered HTML/JavaScript.

## Global Constraints

- Keep Intraday Structure paper-only and make no signal, data, order, or startup behavior changes.
- Preserve unrelated user changes in the dirty worktree; do not commit this narrow change independently.
- Add a behavior-level regression test before production code.

---

### Task 1: Add the shared navigation link

**Files:**

- Modify: `UI/tests/test_hub_dashboard.py`
- Modify: `UI/ui_chrome.py:19-30`

**Interfaces:**

- Consumes: `UI.ui_chrome.NAV_HTML`, which interpolates the static `NAV_PORTS` list into JavaScript.
- Produces: an `Intraday Structure` link to `http://<host>:8774/` on every dashboard using `NAV_HTML`.

- [ ] **Step 1: Write the failing test**

Add this import and test to `UI/tests/test_hub_dashboard.py`:

```python
from UI.ui_chrome import NAV_HTML


def test_shared_navigation_includes_intraday_structure_dashboard():
    assert '"Intraday Structure",8774' in NAV_HTML
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest UI/tests/test_hub_dashboard.py::test_shared_navigation_includes_intraday_structure_dashboard -q
```

Expected: failure because `NAV_HTML` does not yet contain the port-8774 label and link definition.

- [ ] **Step 3: Write the minimal implementation**

Insert this tuple after `("Dealer Ranker", 8773)` in `UI/ui_chrome.NAV_PORTS`:

```python
    ("Intraday Structure", 8774),
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
./.venv/bin/python -m pytest UI/tests/test_hub_dashboard.py::test_shared_navigation_includes_intraday_structure_dashboard -q
```

Expected: one passing test.

- [ ] **Step 5: Run focused regression checks**

Run:

```bash
./.venv/bin/python -m pytest UI/tests/test_hub_dashboard.py UI/tests/test_intraday_structure_dashboard.py -q
./.venv/bin/python -m py_compile UI/ui_chrome.py
git diff --check
```

Expected: all selected tests pass, compilation succeeds, and no whitespace errors appear.
