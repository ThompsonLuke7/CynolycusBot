"""Model-to-data alignment audit: do the live models see the inputs they were trained on?

``scripts/integrity_audit.py`` checks logic and plumbing (stuck states, stale caches,
fallbacks). This checks the other half: whether each live model's feature vectors match its
training data. Every class below reached live unseen for months (2026-11-23/24):

* the 30m swing model scored stream-built bars and a June-frozen daily context, with
  ``market_cap_bucket`` 0 for every name (169 of 190 features off research);
* the SPY day trader scored every 10m bar one minute in;
* momentum scored partial 13:00-15:00 ET hours on ~75% of sessions;
* the economic calendar file collapsed to the TradingView feed (33 events/day, no impact
  field), and the earnings loader stopped reading the report history.

Checks:

``parity``                the nightly live-vs-research check (``scripts/decision_state/parity_check.py``):
                          its latest report, per module, and its age.
``drift:<module>``        the latest archived session's live features against a profile
                          of that module's training matrix: ``null_jump`` (null rate up
                          >= 30 pts), ``out_of_range`` (>= half the values outside the
                          training 0.1-99.9% range), ``stuck`` (a cross-sectional feature
                          constant across the names of each decision), ``frozen`` (a
                          feature that changes daily in training, unchanged for >= 95% of
                          names since the previous archived session), and, as WARN only,
                          ``tail_shift`` (>= 60% of values in one training decile tail).
``econ_calendar``         the deterministic FOMC/NFP/CPI calendar the 30m model trained
                          on: schema, density, a forward high-impact event.
``earnings_forward``      share of the 30m universe with a known next earnings date.
``swing30m_context``      the nightly 30m daily-context rows: fresh, and coverage per
                          column against the training export.

Read-only apart from the cached training profiles (``Data/research/alignment_profiles/``,
rebuilt when a training file changes). The first run builds them (a few minutes); later
runs take seconds. Exit 1 on any FAIL.

    .venv/bin/python -m scripts.alignment_audit                  # latest archived session
    .venv/bin/python -m scripts.alignment_audit --date 2026-11-24 --json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.calendar.us_market_calendar import next_trading_day

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "Data/research/decision_state"
PARITY_DIR = REPO / "Data/research/decision_state_parity"
PROFILE_DIR = REPO / "Data/research/alignment_profiles"
NY = "America/New_York"
PROFILE_VERSION = 2


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL | INFO | SKIP
    detail: str
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Ref:
    """One training source: parquet path, timestamp column and ticker column (None = one name)."""
    path: Path
    ts: str = "timestamp"
    ticker: str | None = "ticker"
    end: str | None = None          # use rows before this date only


REFERENCES: dict[str, list[Ref]] = {
    "momentum_expansion": [Ref(REPO / "strategies/momentum_expansion/data/processed/training_matrix_4h.parquet")],
    # HTF archives the Meta-matrix row it ranks on (htf_score is computed in the matrix
    # update from the shared 4H panel, whose inputs drift:momentum_expansion covers).
    "multi_ticker_swing_htf": [Ref(REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix_research.parquet"),
                               Ref(REPO / "strategies/multi_ticker_swing_htf/data/processed/training_matrix_4h.parquet")],
    "meta_ranker": [Ref(REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix_research.parquet")],
    "multi_ticker_swing_30m": [
        Ref(REPO / "strategies/multi_ticker_swing/data/processed/training_matrix.parquet"),
        Ref(REPO / "strategies/multi_ticker_swing/data/training_export/daily_context_features.parquet", ts="date"),
    ],
    # Rebuilt nightly from 2024-06 on; rows before the sealed test window are its training span.
    "spy_day_trader": [Ref(REPO / "Data/inference/spy/10min/debug_matrices_warmup/spy/live_meta_matrix_on_trace_ts.parquet",
                           ticker=None, end="2026-05-15")],
}

# Forward labels some archives carry next to the inputs (NaN live by design).
# (build_meta_ranker_matrix: meta_label, trade_quality and meta_rank_quality are built from fwd_*.)
LABEL_COLUMNS = {"trend_persistence", "meta_good", "meta_upside", "meta_label", "trade_quality",
                 "meta_rank_quality"}


def is_label(name: str) -> bool:
    return name.startswith("fwd_") or name in LABEL_COLUMNS


# Drift thresholds (see the module docstring).
MIN_ROWS = 20
NULL_JUMP = 0.30
OUT_OF_RANGE = 0.50
RANGE_MARGIN = 0.05       # of the training 0.1-99.9% width (sampling and float32 rounding)
STUCK_GROUPS = 0.90       # share of live decision groups in which the feature is constant
REF_VARIES = 0.80         # ...for a feature that varies within >= 80% of training groups
FROZEN_UNCHANGED = 0.95
REF_CHANGES = 0.50        # ...for a feature unchanged session-over-session <= 50% in training
TAIL_SHIFT = 0.60


# --------------------------------------------------------------------------- profiles

def _sessions(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True)
    return t.dt.tz_convert(NY).dt.normalize()


def profile_frame(df: pd.DataFrame, features: list[str], *, ts: str, ticker: str | None,
                  sessions: list | None = None) -> dict:
    """Per-feature training statistics from a sample of whole decision groups. Pure.

    A group is one bar (all names) for a multi-name source, one session for a single name.
    ``unchanged`` = share of names whose last value of a session equals the previous
    session's (how "daily" a feature is), over consecutive sessions of ``sessions`` (the
    source's session list; default: the sessions present in ``df``).
    """
    df = df.copy()
    df["_sess"] = _sessions(df[ts])
    group = ts if ticker else "_sess"
    out: dict[str, dict] = {}
    last = (df.sort_values(ts).groupby([ticker, "_sess"] if ticker else ["_sess"]).tail(1)
            .sort_values([*([ticker] if ticker else []), "_sess"]))
    # Session-over-session pairs only where the previous row IS the previous session of
    # the source (the sample also holds random bars from unrelated days).
    order = sessions if sessions is not None else sorted(df["_sess"].unique())
    prev_of = dict(zip(order[1:], order[:-1]))
    prev_sess = last.groupby(ticker)["_sess"].shift(1) if ticker else last["_sess"].shift(1)
    consecutive = (prev_sess == last["_sess"].map(prev_of)).to_numpy()
    for f in features:
        if f not in df.columns:
            continue
        s = pd.to_numeric(df[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        nn = s.dropna()
        stats = {"null": float(s.isna().mean()), "n": int(len(s))}
        if len(nn):
            q = nn.quantile([0.001, 0.1, 0.5, 0.9, 0.999]).to_numpy(float)
            stats.update(q001=q[0], q10=q[1], q50=q[2], q90=q[3], q999=q[4])
            nun = s.groupby(df[group]).nunique()
            stats["group_varies"] = float((nun > 1).mean())
            lv = pd.to_numeric(last[f], errors="coerce")
            prev = lv.groupby(last[ticker]).shift(1) if ticker else lv.shift(1)
            ok = lv.notna() & prev.notna() & consecutive
            stats["unchanged"] = float((lv[ok] == prev[ok]).mean()) if ok.any() else None
        out[f] = stats
    return out


def _sample_reference(ref: Ref, features: list[str], *, n_groups: int, seed: int) -> pd.DataFrame:
    """Rows for ``n_groups`` random training bars (and the next session's same-name rows,
    so session-over-session change is measurable)."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    names = set(pq.read_schema(ref.path).names)
    cols = [c for c in features if c in names]
    keys = [ref.ts] + ([ref.ticker] if ref.ticker else [])
    tcol = pq.read_table(ref.path, columns=[ref.ts]).column(ref.ts).to_pandas()
    t = pd.to_datetime(tcol, utc=True)
    if ref.end:
        t = t[t < pd.Timestamp(ref.end, tz="UTC")]
    uniq = pd.Series(t.unique()).sort_values().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if ref.ticker:
        sess = uniq.dt.tz_convert(NY).dt.normalize()
        last_of_day = uniq.groupby(sess).max().sort_index()
        pick = rng.choice(max(1, len(last_of_day) - 1), size=min(n_groups // 2, len(last_of_day) - 1), replace=False)
        chosen = pd.concat([last_of_day.iloc[pick], last_of_day.iloc[pick + 1]])
        # plus random intraday bars for the cross-sectional statistics
        extra = uniq.iloc[rng.choice(len(uniq), size=min(n_groups // 2, len(uniq)), replace=False)]
        chosen = pd.concat([chosen, extra]).drop_duplicates()
    else:
        chosen = uniq          # one name: every row is small enough
    raw_type = pq.read_schema(ref.path).field(ref.ts).type
    values = pd.DatetimeIndex(chosen)
    if getattr(raw_type, "tz", None) is None:
        values = values.tz_convert(None)
    table = pq.read_table(ref.path, columns=list(dict.fromkeys(keys + cols)),
                          filters=pc.field(ref.ts).isin(values.to_numpy()))
    out = table.to_pandas().reset_index()
    out.attrs["sessions"] = sorted(uniq.dt.tz_convert(NY).dt.normalize().unique())
    return out


def _signature(refs: list[Ref]) -> list:
    return [[str(r.path), r.path.stat().st_mtime if r.path.exists() else None, r.end] for r in refs]


def load_profile(module: str, features: list[str], *, n_groups: int = 400, seed: int = 0,
                 rebuild: bool = False) -> dict | None:
    """Cached training profile for ``module`` (rebuilt when a source file changes)."""
    refs = REFERENCES.get(module)
    if not refs or not all(r.path.exists() for r in refs):
        return None
    path = PROFILE_DIR / f"{module}.json"
    sig = _signature(refs)
    if path.exists() and not rebuild:
        cached = json.loads(path.read_text())
        if cached.get("version") == PROFILE_VERSION and cached.get("sources") == sig \
                and set(features) <= set(cached.get("requested", [])):
            return cached
    feats: dict[str, dict] = {}
    for ref in refs:
        df = _sample_reference(ref, features, n_groups=n_groups, seed=seed)
        want = [f for f in features if f in df.columns and f not in feats]
        feats.update(profile_frame(df, want, ts=ref.ts, ticker=ref.ticker, sessions=df.attrs.get("sessions")))
    prof = {"version": PROFILE_VERSION, "module": module, "sources": sig, "requested": sorted(features),
            "built_at": datetime.now(timezone.utc).isoformat(), "features": feats}
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(prof, indent=1, default=float))
    tmp.replace(path)
    return prof


# --------------------------------------------------------------------------- drift

def drift_flags(live: pd.DataFrame, prev: pd.DataFrame | None, profile: dict, *,
                multi_name: bool) -> dict[str, list[dict]]:
    """Flagged features by class for one module's live session. Pure.

    ``live``/``prev``: archived rows (``ticker``, ``bar_ts``, ``f__<feature>``) of the
    session under audit and of the previous archived session.
    """
    flags: dict[str, list[dict]] = {k: [] for k in ("null_jump", "out_of_range", "stuck", "frozen", "tail_shift")}
    live = live.drop_duplicates(["bar_ts", "ticker"], keep="last")
    group = live["bar_ts"] if multi_name else _sessions(live["bar_ts"])
    for f, ref in profile["features"].items():
        col = f"f__{f}"
        if col not in live.columns or len(live) < MIN_ROWS:
            continue
        s = pd.to_numeric(live[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        nn = s.dropna()
        null = float(s.isna().mean())
        if null - ref["null"] >= NULL_JUMP:
            flags["null_jump"].append({"feature": f, "live": round(null, 3), "train": round(ref["null"], 3)})
        if "q001" not in ref or len(nn) < MIN_ROWS:
            continue
        margin = RANGE_MARGIN * (ref["q999"] - ref["q001"]) + 1e-9 * max(1.0, abs(ref["q999"]))
        outside = float(((nn < ref["q001"] - margin) | (nn > ref["q999"] + margin)).mean())
        if outside >= OUT_OF_RANGE and ref["q001"] != ref["q999"]:
            flags["out_of_range"].append({"feature": f, "share": round(outside, 3), "live_median": float(nn.median()),
                                          "train_range": [ref["q001"], ref["q999"]]})
        varies = ref.get("group_varies") or 0.0
        if multi_name and varies >= REF_VARIES:
            sizes = s.groupby(group).size()
            nun = s.groupby(group).nunique(dropna=False)
            big = sizes[sizes >= 5].index
            if len(big) and float((nun.loc[big] <= 1).mean()) >= STUCK_GROUPS:
                flags["stuck"].append({"feature": f, "value": float(nn.iloc[0]), "train_group_varies": round(varies, 3)})
            if len(nn) >= 50:
                lo, hi = float((nn < ref["q10"]).mean()), float((nn > ref["q90"]).mean())
                if max(lo, hi) >= TAIL_SHIFT:
                    flags["tail_shift"].append({"feature": f, "below_q10": round(lo, 3), "above_q90": round(hi, 3)})
        unchanged = ref.get("unchanged")
        if prev is not None and unchanged is not None and unchanged <= REF_CHANGES and col in prev.columns:
            a = live.sort_values("bar_ts").groupby("ticker")[col].last()
            b = prev.sort_values("bar_ts").groupby("ticker")[col].last()
            both = a.index.intersection(b.index)
            a, b = pd.to_numeric(a.loc[both], errors="coerce"), pd.to_numeric(b.loc[both], errors="coerce")
            ok = a.notna() & b.notna()
            if int(ok.sum()) >= MIN_ROWS and float((a[ok] == b[ok]).mean()) >= FROZEN_UNCHANGED:
                flags["frozen"].append({"feature": f, "names": int(ok.sum()), "train_unchanged": round(unchanged, 3)})
    return flags


def _archive_days(module: str) -> list[date]:
    root = ARCHIVE / module
    if not root.exists():
        return []
    return sorted(date.fromisoformat(p.name) for p in root.iterdir() if p.is_dir() and p.name[:4].isdigit())


def _load_archive(module: str, day: date) -> pd.DataFrame:
    parts = sorted((ARCHIVE / module / day.isoformat()).glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in parts]
    return pd.concat(frames, ignore_index=True, sort=False)


def check_drift(module: str, day: date | None, *, rebuild: bool = False) -> Check:
    name = f"drift:{module}"
    days = _archive_days(module)
    if day is not None:
        days = [d for d in days if d <= day]
    if not days:
        return Check(name, "SKIP", "no archived session")
    live = _load_archive(module, days[-1])
    prev = _load_archive(module, days[-2]) if len(days) > 1 else None
    feats = sorted(c[3:] for c in live.columns if c.startswith("f__") and not is_label(c[3:]))
    if live.empty or not feats:
        return Check(name, "SKIP", f"{days[-1]}: no archived features")
    prof = load_profile(module, feats, rebuild=rebuild)
    if prof is None:
        return Check(name, "SKIP", "no training reference configured or found")
    prof = {**prof, "features": {f: v for f, v in prof["features"].items() if f in set(feats)}}
    multi = live["ticker"].nunique() > 1
    flags = drift_flags(live, prev, prof, multi_name=multi)
    hard = {k: v for k, v in flags.items() if v and k != "tail_shift"}
    covered = len(prof["features"])
    parts = [f"{k} {len(v)} ({', '.join(x['feature'] for x in v[:4])}{'...' if len(v) > 4 else ''})"
             for k, v in flags.items() if v]
    head = (f"{days[-1]}: {len(live)} rows, {covered}/{len(feats)} features have a training profile"
            + (f" (vs {days[-2]})" if prev is not None else ""))
    status = "FAIL" if hard else ("WARN" if flags["tail_shift"] else "PASS")
    return Check(name, status, head + ("; " + "; ".join(parts) if parts else "; aligned"),
                 {"session": days[-1].isoformat(), "flags": flags, "features_profiled": covered})


# --------------------------------------------------------------------------- parity

def check_parity(now: datetime | None = None, *, max_age_hours: float = 36.0) -> Check:
    reports = sorted(PARITY_DIR.glob("parity_*.json"))
    if not reports:
        return Check("parity", "FAIL", f"no parity report in {PARITY_DIR}")
    rep = json.loads(reports[-1].read_text())
    now = now or datetime.now(timezone.utc)
    age = (now - datetime.fromisoformat(rep["checked_at"])).total_seconds() / 3600
    failing, notes = [], []
    for mod, r in sorted(rep.get("modules", {}).items()):
        st = r.get("status")
        if st == "FAIL":
            slots = [f"{s}" for s, v in (r.get("by_slot") or {}).items() if v.get("status") == "FAIL"]
            failing.append(f"{mod} ({len(r.get('failing_features', []))} feats"
                           + (f"; slots {', '.join(slots)}" if slots else "") + ")")
        elif st != "PASS":
            notes.append(f"{mod} {st}")
    status = "FAIL" if failing else ("WARN" if age > max_age_hours else "PASS")
    detail = (f"{reports[-1].name} ({age:.0f} h old): "
              + ("FAIL " + "; ".join(failing) if failing else "no module failing")
              + (f"; {', '.join(notes)}" if notes else ""))
    if age > max_age_hours:
        detail += f"; older than {max_age_hours:.0f} h (did the nightly readiness run?)"
    return Check("parity", status, detail, {"report": str(reports[-1]), "modules": {
        m: r.get("status") for m, r in rep.get("modules", {}).items()}})


# --------------------------------------------------------------------------- sources

def check_econ_calendar(today: date) -> Check:
    from signals.meta_context.build_meta_ranker_matrix import ECONOMIC_CALENDAR_CANDIDATES, economic_events_from_calendar
    path = ECONOMIC_CALENDAR_CANDIDATES[0]
    if not path.exists():
        return Check("econ_calendar", "FAIL", f"missing {path}")
    raw = pd.read_parquet(path)
    if "event_date" in raw.columns or "impact" not in raw.columns or "date" not in raw.columns:
        return Check("econ_calendar", "FAIL",
                     f"{path.name} is not the deterministic FOMC/NFP/CPI schema (columns {list(raw.columns)}); "
                     "rebuild: python -m signals.events.economic_calendar --build")
    ev = economic_events_from_calendar(raw)
    t0 = pd.Timestamp(today)
    nxt = ev[(ev["event_date"] >= t0) & (ev["is_high_impact"] > 0)]["event_date"].min()
    past = ev[(ev["event_date"] < t0) & (ev["event_date"] >= t0 - pd.Timedelta(days=365))]
    per_day = len(past) / 252
    problems = []
    if pd.isna(nxt) or (nxt - t0).days > 45:
        problems.append("no high-impact event in the next 45 days")
    if per_day > 0.5:
        problems.append(f"{per_day:.2f} events/session (training: ~0.13)")
    return Check("econ_calendar", "FAIL" if problems else "PASS",
                 (", ".join(problems) if problems else
                  f"{len(ev)} events, {per_day:.2f}/session over the last year, next high-impact {nxt.date()}"))


def check_earnings_forward(today: date, *, warn_below: float = 0.70) -> Check:
    from signals.meta_context.build_meta_ranker_matrix import EARNINGS_CALENDAR_CANDIDATES, earnings_events_from_calendar
    path = EARNINGS_CALENDAR_CANDIDATES[0]
    if not path.exists():
        return Check("earnings_forward", "FAIL", f"missing {path}")
    uni = set(json.loads((REPO / "strategies/multi_ticker_swing/config/trading_universe.json").read_text()))
    ev = earnings_events_from_calendar(pd.read_parquet(path))
    t0 = pd.Timestamp(today)
    fut = ev[(ev["event_date"] >= t0) & (ev["event_date"] <= t0 + pd.Timedelta(days=120)) & ev["ticker"].isin(uni)]
    share = fut["ticker"].nunique() / max(1, len(uni))
    status = "PASS" if share >= warn_below else "WARN"
    return Check("earnings_forward", status,
                 f"{share:.0%} of the {len(uni)}-name 30m universe has a next earnings date within 120 days "
                 f"(training ~95%). Low right after a reporting season while Yahoo posts next dates.",
                 {"share": share})


def check_swing30m_context(today: date) -> Check:
    from strategies.multi_ticker_swing.live.context_features import LIVE_DAILY_CONTEXT, TRAINING_DAILY_CONTEXT
    if not LIVE_DAILY_CONTEXT.exists():
        return Check("swing30m_context", "FAIL", f"missing {LIVE_DAILY_CONTEXT} (nightly readiness stage)")
    live = pd.read_parquet(LIVE_DAILY_CONTEXT)
    live["date"] = pd.to_datetime(live["date"])
    need = pd.Timestamp(next_trading_day(today))
    problems = []
    if live["date"].max() < need:
        problems.append(f"latest row {live['date'].max().date()} < next session {need.date()}")
    train = pd.read_parquet(TRAINING_DAILY_CONTEXT)
    train["date"] = pd.to_datetime(train["date"])
    train = train[train["date"] >= train["date"].max() - pd.Timedelta(days=180)]
    cur = live[live["date"] == live["date"].max()]
    low = []
    for c in [c for c in train.columns if c not in ("ticker", "date") and c in cur.columns]:
        tr, lv = float(train[c].notna().mean()), float(cur[c].notna().mean())
        if tr - lv >= 0.30:
            low.append(f"{c} {lv:.0%} vs {tr:.0%}")
    if low:
        problems.append("coverage below training: " + ", ".join(low))
    return Check("swing30m_context", "FAIL" if problems and not low else ("WARN" if problems else "PASS"),
                 "; ".join(problems) if problems else f"rows through {live['date'].max().date()}, coverage matches training")


# --------------------------------------------------------------------------- run

def run(day: date | None = None, *, rebuild: bool = False) -> list[Check]:
    today = day or datetime.now(timezone.utc).astimezone().date()
    checks = [check_parity()]
    for module in REFERENCES:
        try:
            checks.append(check_drift(module, day, rebuild=rebuild))
        except Exception as exc:  # noqa: BLE001 - one module's failure must not hide the rest
            checks.append(Check(f"drift:{module}", "FAIL", f"audit error {type(exc).__name__}: {exc}"))
    for fn in (check_econ_calendar, check_earnings_forward, check_swing30m_context):
        try:
            checks.append(fn(today))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check(fn.__name__.removeprefix("check_"), "FAIL", f"audit error {type(exc).__name__}: {exc}"))
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", type=date.fromisoformat, default=None, help="audit the archive up to this ET date")
    ap.add_argument("--rebuild-profiles", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    checks = run(args.date, rebuild=args.rebuild_profiles)
    if args.json:
        print(json.dumps({"date": (args.date or date.today()).isoformat(),
                          "checks": [asdict(c) for c in checks]}, indent=2, default=str))
    else:
        width = max(len(c.name) for c in checks)
        print(f"alignment audit{f' for {args.date}' if args.date else ''}")
        for c in checks:
            print(f"  {c.status:<4}  {c.name:<{width}}  {c.detail}")
    return 1 if any(c.status == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
