"""
Regression tests for the capstone results lock.

Guards two things:
  1. the fixed artifacts behind every paper-cited number have not silently
     changed (quick-hash fingerprints), and
  2. re-running the reproduction code on those artifacts returns the locked
     numbers.

If a model or backtest is intentionally regenerated, re-run
  PYTHONPATH=. .venv/bin/python scripts/capstone/reproduce_results.py --write-lock
and commit the new lock together with the new artifacts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.capstone import reproduce_results as rr

LOCK_PATH = rr.LOCK_PATH

pytestmark = pytest.mark.safe


@pytest.fixture(scope="module")
def lock() -> dict:
    if not LOCK_PATH.exists():
        pytest.skip(f"results lock missing: {LOCK_PATH} (run reproduce_results.py --write-lock)")
    return json.loads(LOCK_PATH.read_text())


def _lock_metrics(lock: dict) -> dict[tuple[str, str], dict]:
    return {(m["model"], m["metric"]): m for m in lock["metrics"]}


def _assert_rows_match(rows: list[dict], lock: dict) -> None:
    locked = _lock_metrics(lock)
    missing, mismatched = [], []
    for r in rows:
        key = (r["model"], r["metric"])
        if key not in locked:
            missing.append(key)
            continue
        want, got = locked[key]["value"], r["value"]
        if isinstance(want, float) and isinstance(got, (int, float)):
            if not math.isclose(want, got, rel_tol=1e-6, abs_tol=1e-9):
                mismatched.append((key, want, got))
        elif want != got:
            mismatched.append((key, want, got))
    assert not missing, f"metrics absent from lock (re-write lock?): {missing}"
    assert not mismatched, "locked values no longer reproduce: " + \
        "; ".join(f"{k}: lock={w} now={g}" for k, w, g in mismatched)


# ---------------------------------------------------------------------------
# 1. Artifact fingerprints
# ---------------------------------------------------------------------------

def test_artifacts_unchanged(lock):
    current = rr.fingerprint()
    drifted = []
    for name, locked_fp in lock["artifacts"].items():
        cur = current.get(name, {"exists": False})
        if not locked_fp.get("exists"):
            continue  # was already missing at lock time (documented in audit)
        if not cur.get("exists"):
            drifted.append(f"{name}: artifact deleted")
        elif cur["hash"] != locked_fp["hash"]:
            drifted.append(f"{name}: content changed (rows {locked_fp.get('rows')} -> {cur.get('rows')})")
    assert not drifted, (
        "paper-cited artifacts changed since the results lock — regenerate the lock "
        "intentionally if this was a deliberate retrain/rebuild:\n  " + "\n  ".join(drifted)
    )


# ---------------------------------------------------------------------------
# 2. Cheap reproductions (json/csv-backed) — every headline the paper cites
# ---------------------------------------------------------------------------

def test_swing_eval_and_backtest_lock_reproduce(lock):
    rows = rr.swing_model_metrics(recompute_probs=False)
    rows += rr.swing_backtest_lock()
    rows += rr.swing_backtest_clean_lock()
    _assert_rows_match(rows, lock)


def test_swing_backtest_clean_beats_stale_selection_bias(lock):
    """The val-selected/test-frozen swing patch (audit §1.4) must exist and its
    win rates must be plausible PnL rates, not the double-digit artifacts of a
    broken loader silently dropping most tickers."""
    locked = _lock_metrics(lock)
    key = ("swing", "bt_v2_clean_win_rate")
    assert key in locked, "sweep_v2_clean artifact missing — run the clean swing backtest and re-lock"
    assert 0.0 < locked[key]["value"] < 1.0


def test_family_compare_clean_reproduces(lock):
    """Val-selected/test-frozen momentum & HTF order-policy patch (audit §0.2/§2/§3)."""
    rows = rr._family_compare_clean_lock("momentum", "mom_family_clean")
    rows += rr._family_compare_clean_lock("htf_swing", "htf_family_clean")
    assert rows, "family_compare_clean artifacts missing — run scripts/capstone/family_backtest_clean.py --strategy all"
    _assert_rows_match(rows, lock)


def test_family_compare_clean_fixes_selection_bias_magnitude(lock):
    """Regression guard for the audit's worked example: the ORIGINAL
    test-selected momentum policy rails to ret_over_dd=44.6x (loosest grid
    edge, tuned on the same window it reports); the clean val-selected/
    test-frozen number must be far smaller and plausible."""
    locked = _lock_metrics(lock)
    key = ("momentum", "clean_deployed_winner_ret_over_dd")
    assert key in locked, "momentum family_compare_clean missing from lock"
    clean_ret_over_dd = locked[key]["value"]
    assert 0 < clean_ret_over_dd < 15, (
        f"clean momentum ret_over_dd={clean_ret_over_dd} — expected well below the "
        "44.6x test-selection artifact; investigate before citing in the paper"
    )


def test_swing_paper_trading_reproduces(lock):
    _assert_rows_match(rr.swing_paper_trading(), lock)


def test_headline_swing_backtest_values(lock):
    """Pin the exact numbers the advisor doc / paper cite (audit §1.4)."""
    locked = _lock_metrics(lock)
    assert math.isclose(locked[("swing", "bt_v2_long_wr_sector_agg")]["value"], 0.626226, abs_tol=1e-6)
    assert math.isclose(locked[("swing", "bt_v2_short_wr_sector_agg")]["value"], 0.599913, abs_tol=1e-6)
    # the correct combined value — the advisor doc's 62.3% is wrong and must not return
    assert math.isclose(locked[("swing", "bt_v2_combined_wr_sector_agg")]["value"], 0.61322, abs_tol=1e-5)


def test_competition_metrics_reproduce(lock):
    rows = []
    rows += rr._competition_metrics("momentum", "mom_eval_metrics", "mom_seed_results")
    rows += rr._competition_metrics("htf_swing", "htf_eval_metrics", "htf_seed_results")
    rows += rr._competition_metrics("meta_quality", "meta_q_eval", "__none__")
    rows += rr._competition_metrics("meta_upside", "meta_u_eval", "__none__")
    _assert_rows_match(rows, lock)


def test_spy_and_benchmark_reproduce(lock):
    _assert_rows_match(rr.spy_metrics() + rr.benchmark_metrics(), lock)


# ---------------------------------------------------------------------------
# 3. OOF top-K signal-quality stats (parquet-backed, ~1 min total)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_momentum_oof_topk_reproduces(lock):
    oof = rr._read_oof(rr.ARTIFACTS["mom_oof"])
    _assert_rows_match(
        rr._topk_oof_stats(oof, "momentum", extra_cols=("fwd_max_return", "fwd_max_drawdown")), lock)


@pytest.mark.slow
def test_htf_oof_topk_reproduces(lock):
    oof = rr._read_oof(rr.ARTIFACTS["htf_oof"])
    _assert_rows_match(
        rr._topk_oof_stats(oof, "htf_swing", extra_cols=("fwd_best_high_return", "fwd_worst_low_return")), lock)


@pytest.mark.slow
def test_meta_oof_including_combo_reproduces(lock):
    _assert_rows_match(rr.meta_metrics(), lock)


# ---------------------------------------------------------------------------
# 4. Heavy: swing probs x matrix join (~2 min / ~6GB) — run explicitly with -m slow
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_swing_probs_recompute_matches_eval_metrics(lock):
    rows = rr.swing_recompute_test_from_probs()
    _assert_rows_match(rows, lock)
    # cross-check: recomputed test accuracy agrees with eval_metrics.json
    em = json.loads(rr.ARTIFACTS["swing_eval_metrics"].read_text())
    rec = {r["metric"]: r["value"] for r in rows}
    assert math.isclose(rec["test_accuracy_recomputed"], em["test_accuracy"], abs_tol=5e-4)
