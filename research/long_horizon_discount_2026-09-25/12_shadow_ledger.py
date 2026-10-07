"""Moved to scripts/shadow/momentum_shadow_ledger.py (2026-10-06) so the weekly refresh can run it and tests
can import it. This shim keeps the old command working; see that module for the schedule and the arms.

    .venv/bin/python research/long_horizon_discount_2026-09-25/12_shadow_ledger.py --auto | --mark | --status
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.shadow.momentum_shadow_ledger import main  # noqa: E402

if __name__ == "__main__":
    main()
