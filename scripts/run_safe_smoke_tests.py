from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


SAFE_TEST_SUITES = [
    ("signals/catalysts/tests", ()),
    ("signals/events/tests", ()),
    ("signals/events/forward_guidance/tests", ("sklearn",)),
    ("signals/meta_context/tests", ()),
    ("strategies/momentum_expansion/tests", ("alpaca",)),
    ("strategies/multi_ticker_swing/tests", ()),
    ("signals/news/tests", ()),
    ("signals/social_attention/tests", ()),
    ("UI/tests", ()),
]


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    selected_paths: list[str] = []
    skipped: list[str] = []
    for path, modules in SAFE_TEST_SUITES:
        missing = [name for name in modules if find_spec(name) is None]
        if missing:
            skipped.append(f"{path} (missing {', '.join(missing)})")
        else:
            selected_paths.append(path)
    if skipped:
        print("Skipping unavailable smoke suites:")
        for item in skipped:
            print(f"  - {item}")
    if not selected_paths:
        print("No smoke suites are runnable in this environment.")
        return 1
    args = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "not network and not slow and not live",
        *selected_paths,
        *sys.argv[1:],
    ]
    return subprocess.call(args, cwd=repo)


if __name__ == "__main__":
    raise SystemExit(main())
