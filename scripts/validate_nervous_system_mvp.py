"""Run the MVP acceptance evidence and report it in one place.

Deliberately a reporter, not a gate that can be satisfied by being run. It
executes the suites, prints what passed and what did not, and exits non-zero on
failure. It cannot mark anything as accepted on its own — the operational
requirements (a shadow soak, a controlled paper-submit subset) are not
executable here and are printed as outstanding every time.

    python -m scripts.validate_nervous_system_mvp
    python -m scripts.validate_nervous_system_mvp --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

# Each entry is (label, pytest target). Kept explicit rather than globbed: the
# point is to state what evidence is being claimed.
SUITES: tuple[tuple[str, str], ...] = (
    ("contracts", "core/nervous_system/tests/test_trading_contracts.py"),
    ("policy", "core/nervous_system/tests/test_policy_engine.py"),
    ("policy properties", "core/nervous_system/tests/test_policy_properties.py"),
    ("gateway", "core/nervous_system/tests/test_execution_gateway.py"),
    ("gateway recovery", "core/nervous_system/tests/test_gateway_crash_recovery.py"),
    ("journal", "core/nervous_system/tests/test_journal_hash_chain.py"),
    ("journal probe", "core/nervous_system/tests/test_journal_probe.py"),
    ("option payoff", "core/nervous_system/tests/test_option_payoff.py"),
    ("option selector", "core/nervous_system/tests/test_option_selector.py"),
    ("meta no-bypass", "core/nervous_system/tests/test_meta_no_bypass.py"),
    ("meta intents", "core/nervous_system/tests/test_meta_intent_mapping.py"),
    ("meta gateway", "core/nervous_system/tests/test_meta_gateway_router.py"),
    ("source fitness", "core/nervous_system/tests/test_source_fitness.py"),
    ("replay parity", "core/nervous_system/tests/test_replay_parity.py"),
    ("attribution", "core/nervous_system/tests/test_outcome_attribution.py"),
    ("audit http", "core/nervous_system/tests/test_audit_http.py"),
    ("cloud runtime", "core/nervous_system/tests/test_cloud_runtime.py"),
    ("db cli", "core/nervous_system/tests/test_cloud_db_cli.py"),
    ("mvp acceptance", "core/nervous_system/tests/test_mvp_acceptance.py"),
)

# Requirements that no test can discharge. Printed every run so they cannot be
# quietly forgotten between "the suite is green" and "we turned it on".
OUTSTANDING: tuple[str, ...] = (
    "Shadow soak: >= 20 sessions and >= 100 eligible Meta intents, not yet run.",
    "Controlled paper-submit subset with entry caps and stop conditions, not yet run.",
    "QA Cloud SQL instance and GCS journal bucket not yet provisioned.",
    "Option source fitness not yet measured against a real entitlement.",
    "Legacy direct-submit paths (HTF, Momentum, Swing, Dealer, SPY) remain; "
    "this is a Meta cutover, not a repository-wide one.",
)


def _run(target: str) -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "--no-header"],
        capture_output=True,
        text=True,
    )
    tail = (completed.stdout or completed.stderr).strip().splitlines()
    return completed.returncode == 0, tail[-1] if tail else "no output"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    results = []
    for label, target in SUITES:
        ok, summary = _run(target)
        results.append({"suite": label, "ok": ok, "summary": summary})

    failed = [entry for entry in results if not entry["ok"]]

    if args.json:
        print(json.dumps({
            "ok": not failed, "results": results, "outstanding": list(OUTSTANDING),
        }, indent=2))
    else:
        for entry in results:
            mark = "PASS" if entry["ok"] else "FAIL"
            print(f"  [{mark}] {entry['suite']:<20} {entry['summary']}")
        print("\nNOT PROVEN by this script — operational, and still outstanding:")
        for item in OUTSTANDING:
            print(f"  - {item}")
        print(
            "\n"
            + ("ALL SUITES PASSED. This is not acceptance."
               if not failed else f"{len(failed)} SUITE(S) FAILED.")
        )
        if not failed:
            print("Acceptance additionally requires the outstanding items above.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
