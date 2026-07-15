from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_absolute_runner_launch_with_repo_pythonpath_does_not_shadow_signals():
    repo = Path(__file__).resolve().parents[3]
    runner = repo / "strategies/dealer_positioning/live_ranked_options.py"
    env = {**os.environ, "PYTHONPATH": str(repo)}
    result = subprocess.run(
        [sys.executable, str(runner), "--help"],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Run dealer-ranked ATM options pass" in result.stdout
