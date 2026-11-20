"""Decision-state archive (``decision_state_v1``): every scored candidate, every decision.

WHY. A live decision cannot be re-derived later. The bar cache gets catch-up
writes, matrices are rebuilt, models are retrained and universes refreshed, so
"what did the model see when it chose X" is lost unless it is stored at the
moment of choice. Until this existed, only the *picks* were logged
(``core.live_signal_audit``), without feature vectors, so neither research/live
feature parity nor "why did we miss this winner" could be measured
(``research/year_two_experiment_checklist_2026-11-04.md`` item 0.2).

WHAT. One parquet file per decision run, one row per *scored candidate* (not
only the picks), under ``Data/research/decision_state/<module>/<ET date>/``.
Files are written once and never rewritten: the writer links a fully written
temp file into place and refuses to replace an existing one. Nothing here
touches the state DB (it was down 2026-10-07..10-24).

Row layout:

* run columns (same on every row): ``schema_version``, ``run_id``, ``module``,
  ``decision_ts`` (wall clock when the decision was made, UTC),
  ``inference_start`` / ``inference_end``, ``model_hash``, ``model_version``,
  ``feature_schema_hash``, ``universe_hash``, ``n_candidates``, ``git_commit``,
  ``git_dirty``, ``inputs_asof_json`` (availability time of each input),
  ``regime_json``, ``run_meta_json`` (pass kind, submit flag, config);
* candidate columns: ``bar_ts`` (the bar the decision is about, UTC; the
  module's own bar label), ``ticker``, ``side``, ``score``, ``score_name``,
  ``rank`` (1 = best among eligible rows), ``eligible``, ``gate_reason`` (first
  gate that failed), ``selected``, ``action``;
* ``g__<gate>``: one boolean per gate, True = passed;
* ``f__<feature>``: the full model feature vector (float32, which is what
  XGBoost sees);
* ``x__<name>``: module extras that are not model inputs (other scores,
  contract or quote details).

JOINS. ``run_id`` is also stamped onto each pick's signal audit
(``stamp_run_id``) as ``decision_run_id``. Order-plan audits and order audits
embed the signal audit, so the contract chosen, the quote at decision and the
order outcome join back to the archived row by ``(decision_run_id, ticker)``.

SAFETY. Archiving must never stop a trading run. ``archive_decision_state``
logs and returns ``None`` on any failure unless ``strict=True`` (tests).
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.live_signal_audit import json_safe

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "decision_state_v1"
REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "Data" / "research" / "decision_state"
ENV_ROOT = "CYNOLYCUS_DECISION_ARCHIVE_ROOT"
ENV_SWITCH = "CYNOLYCUS_DECISION_ARCHIVE"  # "0" / "off" / "false" disables writes
ET = ZoneInfo("America/New_York")

FEATURE_PREFIX = "f__"
GATE_PREFIX = "g__"
EXTRA_PREFIX = "x__"

RUN_COLUMNS = (
    "schema_version", "run_id", "module", "decision_ts", "inference_start", "inference_end",
    "model_hash", "model_version", "feature_schema_hash", "universe_hash", "n_candidates",
    "git_commit", "git_dirty", "inputs_asof_json", "regime_json", "run_meta_json",
)
CANDIDATE_COLUMNS = (
    "bar_ts", "ticker", "side", "score", "score_name", "rank", "eligible", "gate_reason",
    "selected", "action",
)


@dataclass(frozen=True)
class ArchiveResult:
    path: Path
    run_id: str
    rows: int
    features: int


# --------------------------------------------------------------------------- helpers

def new_run_id() -> str:
    return uuid.uuid4().hex


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_utc(value: Any, *, field: str = "timestamp") -> pd.Timestamp:
    """A timezone-aware UTC Timestamp. Naive input is refused: guessing its zone
    is how a 4-5 hour look-ahead gets into a join."""
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"{field} is missing")
    if ts.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware, got naive {ts}")
    return ts.tz_convert("UTC")


def _utc_series(values: pd.Series, *, field: str) -> pd.Series:
    out = pd.to_datetime(values, errors="coerce")
    if getattr(out.dt, "tz", None) is None:
        if out.notna().any():
            raise ValueError(f"{field} must be timezone-aware")
        return out.dt.tz_localize("UTC")
    return out.dt.tz_convert("UTC")


def file_asof(path: str | Path) -> str | None:
    """Last-modified time of an input file (UTC ISO), i.e. when its content
    became available to the process that read it. None if absent."""
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _short_sha(payload: str | bytes) -> str:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(data).hexdigest()[:16]


_FILE_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_digest(path: Path) -> str:
    st = path.stat()
    key = (str(path.resolve()), int(st.st_size), int(st.st_mtime_ns))
    cached = _FILE_DIGESTS.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _FILE_DIGESTS[key] = digest
    return digest


def model_hash(paths: Iterable[str | Path]) -> str | None:
    """Content hash of the model artefacts that produced the score. Directories
    are hashed file by file. A missing file is recorded, not ignored."""
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(q for q in p.rglob("*") if q.is_file()))
        else:
            files.append(p)
    if not files:
        return None
    h = hashlib.sha256()
    for p in sorted(files, key=lambda q: str(q)):
        digest = _file_digest(p) if p.exists() else "missing"
        h.update(f"{p.name}:{digest};".encode("utf-8"))
    return h.hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def git_state() -> tuple[str | None, bool | None]:
    """(commit, dirty) of the checkout, read once per process: that is the code
    this process loaded."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO, capture_output=True, text=True, timeout=20,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def feature_schema_hash(frame: pd.DataFrame, feature_cols: Sequence[str]) -> str:
    spec = [[c, str(frame[c].dtype) if c in frame.columns else "missing"] for c in feature_cols]
    return _short_sha(json.dumps(spec, separators=(",", ":")))


def universe_hash(tickers: Iterable[str]) -> str:
    return _short_sha(",".join(sorted({str(t).upper() for t in tickers})))


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(json_safe(value), sort_keys=True, default=str, separators=(",", ":"))


def stamp_run_id(signal_audits: Mapping[str, dict] | None, run_id: str | None) -> None:
    """Carry the archive run id on every pick's signal audit, so order and fill
    records (which embed the signal audit) join back to the archived row."""
    if not run_id or not signal_audits:
        return
    for audit in signal_audits.values():
        if isinstance(audit, dict):
            audit["decision_run_id"] = run_id


# --------------------------------------------------------------------------- frame

def _tickers(candidates: pd.DataFrame) -> pd.Series:
    if "ticker" in candidates.columns:
        raw = candidates["ticker"]
    elif "symbol" in candidates.columns:
        raw = candidates["symbol"]
    elif candidates.index.name in ("ticker", "symbol"):
        raw = pd.Series(candidates.index, index=candidates.index)
    else:
        raise KeyError("candidates need a ticker/symbol column or index")
    return pd.Series(raw.astype(str).str.upper().to_numpy(), index=candidates.index)


def _positional(value: Any, index: pd.Index, *, name: str) -> pd.Series:
    """Per-candidate values in candidate ROW order (Series indexes are ignored:
    callers build them from the same frame, and the archive resets the index)."""
    arr = value.to_numpy() if isinstance(value, pd.Series) else np.asarray(value, dtype=object)
    if arr.ndim == 0:
        arr = np.full(len(index), arr.item(), dtype=object)
    if len(arr) != len(index):
        raise ValueError(f"{name!r} has {len(arr)} values for {len(index)} candidates")
    return pd.Series(arr, index=index)


def _bool_column(value: Any, index: pd.Index, *, name: str) -> pd.Series:
    s = _positional(value, index, name=name)
    return s.where(s.notna(), False).astype(bool)


def _feature_column(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("float32")
    return s.astype("string")


def _extra_column(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return s
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return s.map(lambda v: v if v is None or isinstance(v, str) else _json(v)).astype("string")


def build_decision_frame(
    *,
    module: str,
    candidates: pd.DataFrame,
    bar_ts: Any,
    score_col: str | None,
    feature_cols: Sequence[str] | None = None,
    gates: Mapping[str, Any] | None = None,
    selected: Iterable[str] | pd.Series | None = None,
    rank_col: str | None = None,
    side: str | pd.Series | None = None,
    action: Mapping[str, str] | pd.Series | None = None,
    extra_cols: Sequence[str] = (),
    model_paths: Sequence[str | Path] = (),
    model_version: str | None = None,
    inputs_asof: Mapping[str, Any] | None = None,
    regime: Any = None,
    run_meta: Mapping[str, Any] | None = None,
    inference_start: Any = None,
    inference_end: Any = None,
    decision_ts: Any = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Pure: build the ``decision_state_v1`` frame for one decision run."""
    if not isinstance(candidates, pd.DataFrame) or candidates.empty:
        raise ValueError("candidates must be a non-empty DataFrame")
    index_is_ticker = (candidates.index.name in ("ticker", "symbol")
                       and candidates.index.name not in candidates.columns)
    cand = candidates.reset_index(drop=not index_is_ticker)
    idx = cand.index
    tickers = _tickers(cand)
    if tickers.duplicated().any():
        dupes = sorted(set(tickers[tickers.duplicated()]))[:5]
        raise ValueError(f"duplicate tickers in one decision run: {dupes}")

    decision = to_utc(decision_ts if decision_ts is not None else utc_now(), field="decision_ts")
    if isinstance(bar_ts, str) and bar_ts in cand.columns:
        bars = _utc_series(cand[bar_ts], field="bar_ts")
    elif isinstance(bar_ts, pd.Series):
        if len(bar_ts) != len(idx):
            raise ValueError(f"bar_ts has {len(bar_ts)} values for {len(idx)} candidates")
        bars = _utc_series(pd.Series(bar_ts.to_numpy(), index=idx), field="bar_ts")  # row order
    else:
        bars = pd.Series(to_utc(bar_ts, field="bar_ts"), index=idx)
    if bars.isna().any():
        raise ValueError(f"bar_ts missing on {int(bars.isna().sum())} candidates")
    if (bars > decision).any():
        raise ValueError("bar_ts after decision_ts: the decision cannot be about a future bar")

    score = (pd.to_numeric(cand[score_col], errors="coerce").astype("float64")
             if score_col else pd.Series(np.nan, index=idx, dtype="float64"))

    gate_frame = pd.DataFrame(index=idx)
    for name, value in (gates or {}).items():
        gate_frame[GATE_PREFIX + name] = _bool_column(value, idx, name=name)
    eligible = gate_frame.all(axis=1) if not gate_frame.empty else pd.Series(True, index=idx)
    gate_reason = pd.Series(pd.NA, index=idx, dtype="string")
    for col in reversed(list(gate_frame.columns)):  # first failing gate wins
        gate_reason = gate_reason.mask(~gate_frame[col], col[len(GATE_PREFIX):])

    if rank_col:
        rank = pd.to_numeric(cand[rank_col], errors="coerce").astype("float64")
    else:
        rank = score.where(eligible).rank(ascending=False, method="first")

    if selected is None:
        chosen = pd.Series(False, index=idx)
    elif isinstance(selected, pd.Series) and selected.dtype == bool:
        chosen = _bool_column(selected, idx, name="selected")
    else:
        picks = {str(t).upper() for t in selected}
        chosen = tickers.isin(picks)

    if isinstance(action, pd.Series):
        act = _positional(action, idx, name="action").astype("string")
    elif isinstance(action, Mapping):
        act = tickers.map({str(k).upper(): v for k, v in action.items()}).astype("string")
    else:
        act = pd.Series(pd.NA, index=idx, dtype="string")
    default_action = np.where(chosen, "selected", np.where(eligible, "eligible", "gated"))
    act = act.fillna(pd.Series(default_action, index=idx, dtype="string"))

    if isinstance(side, pd.Series):
        sides = _positional(side, idx, name="side").astype("string")
    else:
        sides = pd.Series(side, index=idx, dtype="string")

    if feature_cols is None:
        skip = {"ticker", "symbol", "timestamp", score_col, rank_col, *extra_cols}
        feature_cols = [c for c in cand.columns
                        if c not in skip and pd.api.types.is_numeric_dtype(cand[c])]
    feature_cols = list(dict.fromkeys(feature_cols))
    missing = [c for c in feature_cols if c not in cand.columns]

    commit, dirty = git_state()
    run = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or new_run_id(),
        "module": str(module),
        "decision_ts": decision,
        "inference_start": to_utc(inference_start, field="inference_start") if inference_start is not None else pd.NaT,
        "inference_end": to_utc(inference_end, field="inference_end") if inference_end is not None else decision,
        "model_hash": model_hash(model_paths),
        "model_version": model_version,
        "feature_schema_hash": feature_schema_hash(cand, feature_cols),
        "universe_hash": universe_hash(tickers),
        "n_candidates": int(len(cand)),
        "git_commit": commit,
        "git_dirty": dirty,
        "inputs_asof_json": _json(dict(inputs_asof or {})),
        "regime_json": _json(regime),
        "run_meta_json": _json({**dict(run_meta or {}), "missing_features": missing}),
    }
    cols: dict[str, Any] = {k: pd.Series([v] * len(idx), index=idx) for k, v in run.items()}
    cols.update({
        "bar_ts": bars,
        "ticker": tickers.astype("string"),
        "side": sides,
        "score": score,
        "score_name": pd.Series(score_col, index=idx, dtype="string"),
        "rank": rank,
        "eligible": eligible.astype(bool),
        "gate_reason": gate_reason,
        "selected": chosen.astype(bool),
        "action": act,
    })
    for c in gate_frame.columns:
        cols[c] = gate_frame[c]
    for c in feature_cols:
        cols[FEATURE_PREFIX + c] = (_feature_column(cand[c]) if c in cand.columns
                                    else pd.Series(np.nan, index=idx, dtype="float32"))
    for c in extra_cols:
        if c in cand.columns:
            cols[EXTRA_PREFIX + c] = _extra_column(cand[c])
    frame = pd.DataFrame(cols)
    for c in ("inference_start", "inference_end", "decision_ts"):
        frame[c] = pd.to_datetime(frame[c], utc=True)
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- write / read

def archive_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get(ENV_ROOT, "").strip()
    return Path(env) if env else DEFAULT_ROOT


def archiving_enabled() -> bool:
    return os.environ.get(ENV_SWITCH, "1").strip().lower() not in {"0", "off", "false", "no"}


def partition_path(root: Path, module: str, decision_ts: pd.Timestamp, run_id: str) -> Path:
    local = decision_ts.tz_convert(ET)
    return root / module / local.strftime("%Y-%m-%d") / f"{local.strftime('%H%M%S')}_{run_id[:12]}.parquet"


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_immutable(frame: pd.DataFrame, path: Path) -> Path:
    """Write ``frame`` to ``path`` once. Raises FileExistsError rather than
    replace anything already there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_parquet(tmp, index=False, compression="zstd")
        with tmp.open("rb") as fh:
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)  # atomic, and fails if the target exists
        except FileExistsError:
            raise
        except OSError:
            if path.exists():
                raise FileExistsError(path) from None
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    _fsync_dir(path.parent)
    return path


def archive_decision_state(*, module: str, root: str | Path | None = None, strict: bool = False,
                           **kwargs: Any) -> ArchiveResult | None:
    """Build and write one decision run. Never raises unless ``strict``: a
    failed archive is logged as an error and trading carries on."""
    if not archiving_enabled():
        return None
    try:
        frame = build_decision_frame(module=module, **kwargs)
        run_id = str(frame["run_id"].iloc[0])
        path = partition_path(archive_root(root), module, frame["decision_ts"].iloc[0], run_id)
        write_immutable(frame, path)
        n_feat = sum(c.startswith(FEATURE_PREFIX) for c in frame.columns)
        logger.info("decision_state: %s archived %d candidates x %d features -> %s",
                    module, len(frame), n_feat, path)
        return ArchiveResult(path=path, run_id=run_id, rows=len(frame), features=n_feat)
    except Exception:  # noqa: BLE001 - must never break a trading run
        if strict:
            raise
        logger.exception("decision_state: archiving FAILED for module=%s (trading unaffected)", module)
        return None


def list_partitions(root: str | Path | None = None, module: str | None = None,
                    start: str | None = None, end: str | None = None) -> list[Path]:
    base = archive_root(root)
    modules = [base / module] if module else sorted(p for p in base.glob("*") if p.is_dir())
    out: list[Path] = []
    for mdir in modules:
        for ddir in sorted(p for p in mdir.glob("*") if p.is_dir()):
            if (start and ddir.name < start) or (end and ddir.name > end):
                continue
            out.extend(sorted(ddir.glob("*.parquet")))
    return out


def load_decision_state(root: str | Path | None = None, module: str | None = None,
                        start: str | None = None, end: str | None = None,
                        columns: Sequence[str] | None = None) -> pd.DataFrame:
    """All archived rows for ``module`` between ET dates ``start``..``end``
    (inclusive, ``YYYY-MM-DD``). Columns are the union across runs."""
    parts = list_partitions(root, module, start, end)
    if not parts:
        return pd.DataFrame(columns=[*RUN_COLUMNS, *CANDIDATE_COLUMNS])
    frames = []
    for p in parts:
        if columns is None:
            frames.append(pd.read_parquet(p))
        else:
            import pyarrow.parquet as pq
            have = set(pq.ParquetFile(p).schema_arrow.names)
            frames.append(pd.read_parquet(p, columns=[c for c in columns if c in have]))
    return pd.concat(frames, ignore_index=True, sort=False)


# --------------------------------------------------------------------------- order joins

def order_plan_actions(audit_log: str | Path) -> pd.DataFrame:
    """What became of each archived pick, from a module's order-plan audit log.

    One row per (``decision_run_id``, ``ticker``) found in ``order_plan``
    events of the shared 4H engine (momentum, Meta, HTF, dealer): the routing
    action and reason, the contract chosen with its quote at decision time, and
    the planned order's disposition. Join to the archive on
    ``run_id == decision_run_id`` and ``ticker``. Rows written before the
    archive existed have no run id and are skipped. Exits of held names are not
    here (they are not decisions on the scored bar); the closed-trade ledgers
    hold them.
    """
    p = Path(audit_log)
    rows: list[dict[str, Any]] = []
    if not p.exists():
        return pd.DataFrame(columns=["decision_run_id", "ticker"])
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            if '"order_plan"' not in line or "decision_run_id" not in line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") != "order_plan":
                continue
            audits = ev.get("signal_audits") or {}
            planned = {str(x.get("symbol")): x for x in ev.get("planned") or []}
            for ticker, sel in (ev.get("contract_selection") or {}).items():
                run_id = (audits.get(ticker) or {}).get("decision_run_id") \
                    or ((sel or {}).get("signal_audit") or {}).get("decision_run_id")
                if not run_id:
                    continue
                sel = sel or {}
                order_symbol = sel.get("occ") if sel.get("action") == "option" else ticker
                plan = planned.get(str(order_symbol)) or {}
                rows.append({
                    "decision_run_id": run_id, "ticker": str(ticker).upper(), "module": ev.get("module"),
                    "plan_bar": ev.get("bar"), "submit": ev.get("submit"),
                    "route_action": sel.get("action"), "route_reason": sel.get("reason"),
                    "occ": sel.get("occ"), "ref_price": sel.get("ref_price"),
                    "quote_json": _json(sel.get("quote")),
                    "planned_qty": plan.get("qty"), "planned_side": plan.get("side"),
                    "disposition": plan.get("disposition"),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- intraday structure

INTRADAY_EVENTS = REPO / "Data" / "inference" / "intraday_structure" / "decision_events.jsonl"
INTRADAY_ENTRY_STATE = "CONFIRMED"
_INTRADAY_SKIP = {"ticker", "direction", "timestamp", "setup_id", "setup_type", "schema_version",
                  "engine_version", "no_trade_reason", "context_regime", "spot_as_of"}


def intraday_structure_decision_state(path: str | Path = INTRADAY_EVENTS, *,
                                      start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Intraday Structure's decisions in ``decision_state_v1`` columns.

    That module already keeps an event-sourced decision log
    (``decision_events.jsonl``: abstentions with every gate input, and setup
    state transitions), so it is mapped here at read time rather than written a
    second time. Abstentions become ``action="abstain"`` with the refusal as
    ``gate_reason``; transitions become ``action="transition:<STATE>"`` and a
    transition into CONFIRMED is the entry (``selected``). ``start``/``end``
    filter on the ET date of ``bar_ts``.
    """
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=[*RUN_COLUMNS, *CANDIDATE_COLUMNS])
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            if '"setup_abstention"' not in line and '"setup_transition"' not in line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("event_type")
            payload = ev.get("payload") or {}
            if kind not in ("setup_abstention", "setup_transition"):
                continue
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "run_id": ev.get("event_id"),
                "module": "intraday_structure",
                "decision_ts": ev.get("recorded_at") or ev.get("available_at") or ev.get("event_time"),
                "bar_ts": payload.get("timestamp") or ev.get("event_time"),
                "ticker": str(payload.get("ticker") or ev.get("ticker") or "").upper(),
                "x__setup_id": payload.get("setup_id"),
                "x__setup_type": payload.get("setup_type"),
                "x__source_schema": payload.get("schema_version") or ev.get("schema_version"),
            }
            if kind == "setup_abstention":
                row.update({
                    "side": payload.get("direction"),
                    "score": payload.get("confidence"),
                    "score_name": "confidence",
                    "eligible": False,
                    "gate_reason": payload.get("no_trade_reason"),
                    "selected": False,
                    "action": "abstain",
                    "regime_json": _json({"context_regime": payload.get("context_regime"),
                                          "regime_evidence": payload.get("regime_evidence")}),
                })
                for k, v in payload.items():
                    if k in _INTRADAY_SKIP:
                        continue
                    if isinstance(v, bool) or isinstance(v, (int, float)) or v is None:
                        row[FEATURE_PREFIX + k] = np.nan if v is None else float(v)
                    elif isinstance(v, list):
                        row[EXTRA_PREFIX + k] = _json(v)
            else:
                to_state = str(payload.get("to_state") or "")
                row.update({
                    "side": str(payload.get("setup_id") or "::").split(":")[1] or None,
                    "score": np.nan,
                    "score_name": None,
                    "eligible": True,
                    "gate_reason": None,
                    "selected": to_state == INTRADAY_ENTRY_STATE,
                    "action": f"transition:{to_state}",
                    FEATURE_PREFIX + "spot": payload.get("spot"),
                    EXTRA_PREFIX + "from_state": payload.get("from_state"),
                    EXTRA_PREFIX + "reason": payload.get("reason"),
                    EXTRA_PREFIX + "evidence": _json(payload.get("evidence")),
                })
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=[*RUN_COLUMNS, *CANDIDATE_COLUMNS])
    frame["decision_ts"] = pd.to_datetime(frame["decision_ts"], utc=True, errors="coerce")
    frame["bar_ts"] = pd.to_datetime(frame["bar_ts"], utc=True, errors="coerce")
    et_date = frame["bar_ts"].dt.tz_convert(ET).dt.strftime("%Y-%m-%d")
    keep = pd.Series(True, index=frame.index)
    if start:
        keep &= et_date >= start
    if end:
        keep &= et_date <= end
    return frame.loc[keep].reset_index(drop=True)
