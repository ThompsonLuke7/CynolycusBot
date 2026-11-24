"""Experiment manifest for the frozen evaluator (year-two checklist item 0.3).

Every evaluation run names, before it sees a result, what it is evaluating and how:
the data (cutoff, universe, feature schema, target, entry basis), the model (family,
hyperparameters, seed, calibration), the trading assumptions (cost model, execution
policy), the code (git commit) and the pre-registered decision rule. ``evaluator.evaluate``
refuses to run without one, and every read of the sealed test window is appended to
``TEST_READ_LOG`` with the manifest's hash.

A manifest is a JSON file; ``load_manifest`` validates it and fails fast on anything
missing, unknown or contradictory.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
TEST_READ_LOG = REPO / "Data" / "research" / "evaluator" / "test_reads.jsonl"

# Outcomes must start from an entry the strategy could have taken. "prior_close" (the
# decision bar's own close, i.e. before the signal existed) inflated MFE in an earlier
# study and is refused (checklist 0.3 regression list).
EXECUTABLE_ENTRIES = ("next_bar_open", "next_1h_open", "decision_plus_delay")
PRIMARY_METRICS = ("neutral_ic", "ic", "net_ret_at_k", "precision_at_k", "perm_delta_ic")

# Named, versioned cost models (round trip, fraction of notional). Equity: 10 bps covers
# spread + slippage for the liquid top of the universe (an assumption, stated, not fitted).
# Options: the ~35% round-trip hurdle measured on the account's own fills
# (research/execution_quality, rank-depth study 2026-10).
COST_MODELS = {
    "zero_v1": 0.0,
    "equity_10bps_v1": 0.0010,
    "equity_25bps_v1": 0.0025,
    "options_35pct_v1": 0.35,
}

REQUIRED = ("experiment_id", "hypothesis", "dataset_cutoff", "universe_version", "feature_schema_hash",
            "target_version", "label_horizon_sessions", "label_entry", "model_family", "hyperparameters", "seed", "calibration",
            "cost_model", "execution_policy", "git_commit", "primary_metric", "decision_rule")


class ManifestError(ValueError):
    """The manifest is missing, incomplete or inconsistent; nothing is evaluated."""


@dataclass(frozen=True)
class Manifest:
    experiment_id: str
    hypothesis: str
    dataset_cutoff: str            # last decision date the data may contain (YYYY-MM-DD)
    universe_version: str
    feature_schema_hash: str
    target_version: str
    label_horizon_sessions: int    # sets the bootstrap block (overlapping labels are not independent)
    label_entry: str               # one of EXECUTABLE_ENTRIES
    model_family: str
    hyperparameters: dict
    seed: int
    calibration: str               # "none" is a valid, explicit answer
    cost_model: str                # key of COST_MODELS
    execution_policy: str
    git_commit: str
    primary_metric: str            # one of PRIMARY_METRICS
    decision_rule: str             # pre-registered, in words
    top_k: tuple[int, ...] = (1, 3, 5, 10)
    min_names_per_bar: int = 200
    hold_sessions: int | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def round_trip_cost(self) -> float:
        return COST_MODELS[self.cost_model]


def manifest_hash(m: Manifest) -> str:
    """Stable 16-hex hash of the manifest's canonical JSON."""
    blob = json.dumps(asdict(m), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=20)


def git_head() -> tuple[str | None, bool | None]:
    head = _git("rev-parse", "HEAD").stdout.strip() or None
    status = _git("status", "--porcelain", "--untracked-files=no")
    return head, (bool(status.stdout.strip()) if status.returncode == 0 else None)


def validate(m: Manifest, *, check_git: bool = True) -> Manifest:
    """Fail fast on an incomplete or contradictory manifest."""
    for name in REQUIRED:
        v = getattr(m, name)
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ManifestError(f"manifest field {name!r} is empty")
    if m.label_entry not in EXECUTABLE_ENTRIES:
        raise ManifestError(f"label_entry {m.label_entry!r} is not an executable entry {EXECUTABLE_ENTRIES}: "
                            "outcomes measured from the decision bar's own (prior) close are refused")
    if m.cost_model not in COST_MODELS:
        raise ManifestError(f"unknown cost_model {m.cost_model!r}; known: {sorted(COST_MODELS)}")
    if m.primary_metric not in PRIMARY_METRICS:
        raise ManifestError(f"unknown primary_metric {m.primary_metric!r}; known: {PRIMARY_METRICS}")
    try:
        cutoff = pd.Timestamp(m.dataset_cutoff)
    except (ValueError, TypeError) as exc:
        raise ManifestError(f"dataset_cutoff {m.dataset_cutoff!r} is not a date") from exc
    if cutoff.normalize() > pd.Timestamp.now().normalize():
        raise ManifestError(f"dataset_cutoff {m.dataset_cutoff} is in the future")
    if not m.top_k or any(int(k) < 1 for k in m.top_k):
        raise ManifestError(f"top_k must be positive integers, got {m.top_k}")
    if int(m.label_horizon_sessions) < 1:
        raise ManifestError("label_horizon_sessions must be >= 1")
    if int(m.min_names_per_bar) < 1:
        raise ManifestError("min_names_per_bar must be >= 1")
    if check_git:
        r = _git("merge-base", "--is-ancestor", m.git_commit, "HEAD")
        if r.returncode != 0:
            raise ManifestError(f"git_commit {m.git_commit!r} is not in this checkout's history (HEAD must "
                                "contain the code the manifest names)")
    return m


def load_manifest(path: str | Path, *, check_git: bool = True) -> Manifest:
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"manifest not found: {p}")
    raw = json.loads(p.read_text())
    unknown = sorted(set(raw) - set(Manifest.__dataclass_fields__))
    if unknown:
        raise ManifestError(f"unknown manifest fields {unknown} (put free-form detail in 'extra')")
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise ManifestError(f"manifest missing required fields: {missing}")
    if "top_k" in raw:
        raw["top_k"] = tuple(int(k) for k in raw["top_k"])
    return validate(Manifest(**raw), check_git=check_git)


def test_reads(experiment_id: str | None = None, *, log_path: Path | None = None) -> list[dict]:
    log_path = log_path or TEST_READ_LOG
    if not log_path.exists():
        return []
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    return [r for r in rows if experiment_id is None or r.get("experiment_id") == experiment_id]


def log_test_read(m: Manifest, *, reason: str, log_path: Path | None = None) -> int:
    """Append one sealed-test read; returns how many reads this experiment now has."""
    log_path = log_path or TEST_READ_LOG
    if not reason or not reason.strip():
        raise ManifestError("a test-window read needs a stated reason")
    head, dirty = git_head()
    prior = test_reads(m.experiment_id, log_path=log_path)
    row = {"at_utc": datetime.now(timezone.utc).isoformat(), "experiment_id": m.experiment_id,
           "manifest_hash": manifest_hash(m), "git_head": head, "git_dirty": dirty,
           "read_number": len(prior) + 1, "reason": reason.strip()}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    return row["read_number"]
