"""Small filesystem transactions shared by strategy runners and their risk pass."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from pathlib import Path

LOCK_ROOT = Path(__file__).resolve().parents[1] / "Data/runtime/risk_pass"


@contextlib.contextmanager
def module_state_lock(module: str, *, root=None):
    directory = Path(root or LOCK_ROOT)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{module}.lock").open("a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def load_state(path):
    path = Path(path)
    if not path.exists():
        return {"managed": {}, "history": []}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Invalid state object: {path}")
    return value


def save_state(path, state):
    """Durable atomic replacement; caller holds the read-modify-write lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh, indent=2, default=str, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def append_once(path, record, *, identity):
    """Append a durable event once, including after state/ledger crash recovery."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            # Do not silently step past corruption and append a duplicate.
            for line in path.read_text().splitlines():
                if line.strip() and json.loads(line).get("event_id") == identity:
                    return False
        with path.open("a") as fh:
            fh.write(json.dumps({**record, "event_id": identity}, default=str, allow_nan=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return True
