"""Unified searchable index over the news / catalyst record stores.

The signal stores are optimised for the model pipelines, not for browsing:
``news_records.parquet`` is a single 425MB row group where ~90% of the bytes are
article text, so *any* filtered read decodes the whole file. This module builds a
compact, ticker-sorted index once per source refresh and answers per-ticker
queries off it with real row-group pruning, so the combined server can browse
327k+ records without carrying them in RSS.

Sources folded into one row set (all time-stamped at *publication* time, UTC):

  * ``signals/news/data/processed/news_records.parquet``          origin="news"
    The canonical nightly library.
  * ``signals/news/data/processed/live_catalyst_records.parquet`` origin="live"
    The intraday RTH poller ledger. Roughly half its rows are already in the
    canonical library once the nightly collection has run, so it is deduped
    against it on ``(ticker, content_hash)`` with the canonical row winning.
  * ``signals/catalysts/data/processed/catalyst_records.parquet`` origin="catalyst"
    Mostly a scored *view* of news rows (7445/7765 record_ids are canonical), so
    its ``catalyst_kind``/``event_type``/score are joined onto the matching news
    rows and only the catalyst-only records are appended as rows of their own.

Everything else is joined on ``record_id`` from stores the NIGHTLY job already
refreshes, so the library tracks the pipelines instead of freezing:

  * ``news_catalyst_per_record.parquet`` — the deployed catalyst model's score
    and trajectory probabilities for each headline (92.9%).
  * ``news_catalyst_signal.parquet`` — the per-(ticker, day) score the live meta
    ranker actually trades on (93.1%).
  * ``news_embeddings.parquet`` — FinBERT tone (100%).
  * ``news_labels.parquet`` — realized forward returns (HINDSIGHT).

Nothing here recomputes, imputes, or re-thresholds a model output; the index is
a read-only projection, and the bull/bear thresholds are imported values from
``build_news_signal.py`` rather than new ones invented here.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

NEWS_RECORDS = REPO / "signals/news/data/processed/news_records.parquet"
LIVE_RECORDS = REPO / "signals/news/data/processed/live_catalyst_records.parquet"
CATALYST_RECORDS = REPO / "signals/catalysts/data/processed/catalyst_records.parquet"
# NOT USED — kept as named constants only to document what was deliberately
# dropped. news_scores.parquet / catalyst_scores.parquet come from
# build_news_similarity_scores(), a superseded offline branch: it froze on
# 2026-05-22, covers ~2% of the library, and NOTHING consumes it — the deployed
# catalyst booster's 468 features contain no similarity term, and
# live_scorer.py's inference path explicitly replaces the cosine-similarity
# score. Re-running it is also impractical (its inner loop re-stacks every
# eligible prior per record: measured 1.0s@1k / 7.6s@2k / 36.3s@4k records,
# clean O(n^2) -> ~69h for 331k, no incremental mode). Read
# NEWS_CATALYST_PER_RECORD instead.
_RETIRED_NEWS_SCORES = REPO / "signals/news/data/processed/news_scores.parquet"
_RETIRED_CATALYST_SCORES = REPO / "signals/catalysts/data/processed/catalyst_scores.parquet"
# Per-record forward returns. Rebuilt nightly by `news.main --stage incremental`
# (label-mature), so unlike news_scores.parquet this one tracks the library.
NEWS_LABELS = REPO / "signals/news/data/processed/news_labels.parquet"
# The per-(ticker, day) score the META RANKER actually trades on, rebuilt nightly
# by `build_news_signal --incremental-cache` and joined into the meta matrix as
# catalyst_score. Showing THIS keeps the library's headline number identical to
# the one the live system sees, instead of a superseded offline artifact.
NEWS_CATALYST_SIGNAL = REPO / "signals/meta_context/data/processed/news_catalyst_signal.parquet"
# PER-RECORD model output behind that daily signal — the deployed catalyst
# classifier's score plus the multiclass trajectory probabilities, cached and
# refreshed nightly by the same build_news_signal pass. This is the model's read
# on THIS headline (not the day), and the trajectory probabilities are a
# genuine decision-time direction call.
NEWS_CATALYST_PER_RECORD = REPO / "signals/meta_context/data/processed/news_catalyst_per_record.parquet"
# FinBERT tone lives with the embeddings (NOT in news_feature_matrix.parquet,
# which belongs to the superseded offline branch and is ~0.6% populated).
NEWS_EMBEDDINGS = REPO / "signals/news/data/processed/news_embeddings.parquet"

# Reused verbatim from signals/meta_context/build_news_signal.py so the library's
# bullish/bearish call is the SAME rule the meta signal aggregates on.
BULL_ALIGNMENT_THRESHOLD = 0.50   # p_bull_steady + p_bull_volatile
BEAR_ALIGNMENT_THRESHOLD = 0.30   # p_crash_stayed

INDEX_PATH = REPO / "Data/runtime/news_library_index.parquet"
STAMP_PATH = REPO / "Data/runtime/news_library_index.json"

BARS_1D_DIR = REPO / "Data/shared/bars/1d"

# Every source file the index is derived from. The stamp records (mtime, size)
# for each so a nightly refresh of any one of them triggers exactly one rebuild.
_SOURCES = {
    "news_records": NEWS_RECORDS,
    "live_catalyst_records": LIVE_RECORDS,
    "catalyst_records": CATALYST_RECORDS,
    "news_labels": NEWS_LABELS,
    "news_catalyst_signal": NEWS_CATALYST_SIGNAL,
    "news_catalyst_per_record": NEWS_CATALYST_PER_RECORD,
    "news_embeddings": NEWS_EMBEDDINGS,
}

# Columns projected out of news_records. Deliberately excludes text/body/summary
# and the earnings_* text blobs — they are ~85% of the file and the library only
# ever displays a headline.
_NEWS_COLS = [
    "record_id", "ticker", "timestamp", "headline", "source", "url",
    "catalyst_family", "catalyst_subtype", "relation_type", "impact_role",
    "source_quality", "is_direct_catalyst", "relation_confidence",
]

# Final index schema, in column order, split into two zones that must never be
# confused for one another:
#
#   KNOWABLE AT THE TIME
#     record_catalyst_score  the deployed model's score for THIS headline
#     day_catalyst_score     the per-(ticker, day) score the meta ranker trades on
#     predicted_direction    bullish/bearish/neutral from the trajectory model
#     p_bullish              p_bull_steady + p_bull_volatile
#     tone                   FinBERT argmax (a top-3 feature of the model above)
#
#   HINDSIGHT (see HINDSIGHT_COLUMNS) — realized only after the record, so it can
#   describe what happened but can never be an input to a decision.
INDEX_COLUMNS = [
    "ticker", "timestamp", "headline", "source", "url", "origin",
    "catalyst_family", "catalyst_subtype", "catalyst_kind", "event_type",
    "relation_type", "impact_role", "source_quality", "is_direct_catalyst",
    "relation_confidence",
    # --- knowable at the time ---
    "record_catalyst_score", "day_catalyst_score",
    "predicted_direction", "p_bullish", "p_crash_stayed", "tone",
    "finbert_positive_score", "finbert_negative_score", "finbert_neutral_score",
    # --- HINDSIGHT: everything below describes what happened AFTER the record ---
    "move_1d_pct", "move_5d_pct", "max_favorable_pct", "max_adverse_pct",
    "move_rank_pct", "outcome",
    "record_id",
]

# Columns that are only knowable after the fact. The UI groups these behind a
# hindsight divider so none of them can be mistaken for a decision-time signal.
HINDSIGHT_COLUMNS = (
    "move_1d_pct", "move_5d_pct", "max_favorable_pct", "max_adverse_pct",
    "move_rank_pct", "outcome",
)

# Band for the good / bad / neutral call on the next session's move. A display
# convention, NOT a model output: it is a fixed threshold on the realized 1-day
# return, chosen to sit outside routine daily noise for a typical name.
OUTCOME_BAND_PCT = 2.0

# Bumped whenever INDEX_COLUMNS or the build logic changes, so an index written
# by an older build is treated as stale even when no source file has moved.
INDEX_VERSION = 4

# Dimensions the chart can colour its news markers by.
COLOR_BY_FIELDS = ("catalyst_family", "source", "origin", "relation_type")

# Bucket for every value beyond the colour-capped top classes.
OTHER_CLASS = "Other"

# Row groups small enough that a single-ticker predicate prunes to a handful.
_ROW_GROUP_SIZE = 4096

MAX_LIMIT = 500

logger = logging.getLogger(__name__)

_build_lock = threading.Lock()


def _source_stamp() -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name, path in _SOURCES.items():
        try:
            st = path.stat()
        except OSError:
            continue
        out[name] = [st.st_mtime, float(st.st_size)]
    return out


def _read_stamp() -> dict[str, Any] | None:
    try:
        return json.loads(STAMP_PATH.read_text())
    except (OSError, ValueError):
        return None


def index_is_current() -> bool:
    if not INDEX_PATH.exists():
        return False
    stamp = _read_stamp()
    if not stamp:
        return False
    if stamp.get("version") != INDEX_VERSION:
        return False
    return stamp.get("sources") == _source_stamp()


def build_index(*, force: bool = False) -> dict[str, Any]:
    """Build (or refresh) the compact index. Returns the stamp payload.

    Held under a process lock so two concurrent dashboard requests cannot both
    pay the multi-hundred-MB build cost, and written temp-then-rename so a reader
    thread never sees a half-written footer (same hazard the HTF dashboard hit).
    """
    with _build_lock:
        if not force and index_is_current():
            return _read_stamp() or {}
        if not NEWS_RECORDS.exists():
            raise FileNotFoundError(f"news records not found at {NEWS_RECORDS}")

        import pandas as pd

        frames: list[pd.DataFrame] = []

        news = pd.read_parquet(NEWS_RECORDS, columns=_NEWS_COLS)
        news["origin"] = "news"
        content_hash = pd.read_parquet(
            NEWS_RECORDS, columns=["ticker", "content_hash"]
        )["content_hash"]
        news_keys = set(zip(news["ticker"].astype(str), content_hash.astype(str)))
        frames.append(news)
        del content_hash

        # Live ledger: keep only what the canonical library has not absorbed yet.
        if LIVE_RECORDS.exists():
            live = pd.read_parquet(LIVE_RECORDS)
            keys = list(zip(live["ticker"].astype(str), live["content_hash"].astype(str)))
            live = live[[k not in news_keys for k in keys]].copy()
            live["origin"] = "live"
            live = live.rename(columns={"catalyst_score": "catalyst_score"})
            frames.append(live)
        del news_keys

        # Catalyst records: join the catalyst-only fields onto news rows, then
        # append the records that exist nowhere else.
        catalyst_only = None
        if CATALYST_RECORDS.exists():
            cat = pd.read_parquet(
                CATALYST_RECORDS,
                columns=["record_id", "ticker", "timestamp", "headline", "source", "url",
                         "catalyst_kind", "event_type", "relation_type", "impact_role",
                         "relation_confidence", "is_direct_catalyst"],
            )
            known = set(news["record_id"].astype(str))
            mask = [r not in known for r in cat["record_id"].astype(str)]
            catalyst_only = cat[mask].copy()
            catalyst_only["origin"] = "catalyst"
            if len(catalyst_only):
                frames.append(catalyst_only)
            cat_join = cat[[not m for m in mask]][["record_id", "catalyst_kind", "event_type"]]
            del cat, known, mask
        else:
            cat_join = None

        df = pd.concat(frames, ignore_index=True, sort=False)
        del frames, news

        for col in ("catalyst_kind", "event_type"):
            if col not in df.columns:
                df[col] = None
        if cat_join is not None and len(cat_join):
            df = df.merge(cat_join, on="record_id", how="left", suffixes=("", "_j"))
            for col in ("catalyst_kind", "event_type"):
                df[col] = df[col].fillna(df.pop(f"{col}_j"))

        # --- Per-record output of the DEPLOYED catalyst model (92.9% coverage,
        # refreshed nightly). This supersedes news_scores.parquet /
        # catalyst_scores.parquet entirely: the deployed booster's 468 features
        # contain no similarity term (live_scorer.py's inference path explicitly
        # "replaces the cosine-similarity score"), so nothing reads those files.
        if NEWS_CATALYST_PER_RECORD.exists():
            pr = pd.read_parquet(
                NEWS_CATALYST_PER_RECORD,
                columns=["record_id", "catalyst_score", "p_bull_steady",
                         "p_bull_volatile", "p_crash_stayed"],
            ).drop_duplicates("record_id").rename(
                columns={"catalyst_score": "record_catalyst_score"})
            df = df.merge(pr, on="record_id", how="left")
            bull = (pd.to_numeric(df.pop("p_bull_steady"), errors="coerce").fillna(0)
                    + pd.to_numeric(df.pop("p_bull_volatile"), errors="coerce").fillna(0))
            crash = pd.to_numeric(df["p_crash_stayed"], errors="coerce")
            has = df["record_catalyst_score"].notna()
            df["p_bullish"] = bull.where(has)
            # Same thresholds build_news_signal.py aggregates on, so the library's
            # call matches the meta signal's bull/bear alignment counts.
            df["predicted_direction"] = pd.Series(pd.NA, index=df.index, dtype="string") \
                .mask(has & (crash.fillna(0) >= BEAR_ALIGNMENT_THRESHOLD), "bearish") \
                .mask(has & (bull >= BULL_ALIGNMENT_THRESHOLD), "bullish") \
                .mask(has & (bull < BULL_ALIGNMENT_THRESHOLD)
                      & (crash.fillna(0) < BEAR_ALIGNMENT_THRESHOLD), "neutral")

        # --- FinBERT tone (100% covered; a top-3 feature of the deployed model).
        if NEWS_EMBEDDINGS.exists():
            fb = pd.read_parquet(
                NEWS_EMBEDDINGS,
                columns=["record_id", "finbert_positive_score", "finbert_negative_score",
                         "finbert_neutral_score"],
            ).drop_duplicates("record_id")
            df = df.merge(fb, on="record_id", how="left")
            tri = df[["finbert_positive_score", "finbert_negative_score",
                      "finbert_neutral_score"]].apply(pd.to_numeric, errors="coerce")
            # idxmax over an all-NA row is deprecated (and will raise), and every
            # record without FinBERT is exactly that — so pick the winner only on
            # scored rows and leave the rest null rather than guessing "neutral".
            scored = tri.notna().any(axis=1)
            tone = pd.Series(pd.NA, index=df.index, dtype="string")
            if scored.any():
                tone.loc[scored] = (
                    tri.loc[scored].idxmax(axis=1)
                    .map({"finbert_positive_score": "positive",
                          "finbert_negative_score": "negative",
                          "finbert_neutral_score": "neutral"})
                    .astype("string")
                )
            df["tone"] = tone

        # --- Forward returns (HINDSIGHT). Per record_id, rebuilt nightly, so this
        # tracks the library instead of freezing like news_scores.parquet.
        if NEWS_LABELS.exists():
            lab = pd.read_parquet(
                NEWS_LABELS,
                columns=["record_id", "forward_1d_return", "forward_5d_return",
                         "max_forward_return", "max_drawdown"],
            ).drop_duplicates("record_id")
            df = df.merge(lab, on="record_id", how="left")
            df["move_1d_pct"] = pd.to_numeric(df.pop("forward_1d_return"), errors="coerce") * 100.0
            df["move_5d_pct"] = pd.to_numeric(df.pop("forward_5d_return"), errors="coerce") * 100.0
            df["max_favorable_pct"] = pd.to_numeric(df.pop("max_forward_return"), errors="coerce") * 100.0
            df["max_adverse_pct"] = pd.to_numeric(df.pop("max_drawdown"), errors="coerce") * 100.0

        # --- The score the live meta ranker actually sees, per (ticker, day).
        # It is a DAY aggregate, so every record on a ticker-day shares it; the
        # UI labels it as the day's score rather than this headline's score.
        if NEWS_CATALYST_SIGNAL.exists():
            sig = pd.read_parquet(
                NEWS_CATALYST_SIGNAL, columns=["timestamp", "ticker", "news_catalyst_score"])
            sig["ticker"] = sig["ticker"].astype(str).str.upper()
            sig["_day"] = pd.to_datetime(sig["timestamp"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
            sig = (sig.dropna(subset=["_day"])
                      .drop_duplicates(["ticker", "_day"])
                      .rename(columns={"news_catalyst_score": "day_catalyst_score"})
                      [["ticker", "_day", "day_catalyst_score"]])
            df["_day"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce") \
                .dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
            df["ticker"] = df["ticker"].astype(str).str.upper()
            df = df.merge(sig, on=["ticker", "_day"], how="left")
            df = df.drop(columns=["_day"])

        for col in INDEX_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[INDEX_COLUMNS]

        df["ticker"] = df["ticker"].astype("string").str.upper()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["ticker", "timestamp"])
        for col in ("record_catalyst_score", "day_catalyst_score", "p_bullish",
                    "p_crash_stayed", "finbert_positive_score", "finbert_negative_score",
                    "finbert_neutral_score", "is_direct_catalyst", "relation_confidence",
                    "move_1d_pct", "move_5d_pct", "max_favorable_pct", "max_adverse_pct"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # --- Derived HINDSIGHT columns ---
        # Rank: where this record's next-session move sits within THIS TICKER's own
        # news history (percentile, 0-100). Ranking within the ticker is what makes
        # it readable — a 3% move is unremarkable for a biotech and huge for a
        # mega-cap, so a cross-ticker rank would mostly encode volatility.
        df["move_rank_pct"] = (
            df.groupby("ticker")["move_1d_pct"].rank(pct=True, na_option="keep") * 100.0
        )
        # Good / bad / neutral on the realized next-session move. Explicitly a
        # display convention over a fixed band (see OUTCOME_BAND_PCT) — it is
        # hindsight, never a prediction, and it stays null where the move is null
        # rather than defaulting to "neutral".
        move = df["move_1d_pct"]
        df["outcome"] = pd.Series(
            pd.NA, index=df.index, dtype="string"
        ).mask(move >= OUTCOME_BAND_PCT, "good") \
         .mask(move <= -OUTCOME_BAND_PCT, "bad") \
         .mask(move.notna() & move.abs().lt(OUTCOME_BAND_PCT), "neutral")

        # Ticker-major, then newest-first inside a ticker: the sort IS the index.
        # Ticker order gives row-group min/max pruning on the ticker predicate;
        # the descending timestamp means the default "most recent" view is
        # already in order after the pruned read.
        df = df.sort_values(["ticker", "timestamp"], ascending=[True, False],
                            kind="mergesort", ignore_index=True)

        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = INDEX_PATH.with_suffix(
            INDEX_PATH.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
        df.to_parquet(tmp, index=False, row_group_size=_ROW_GROUP_SIZE,
                      compression="zstd")
        tmp.replace(INDEX_PATH)

        stamp = {
            "version": INDEX_VERSION,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "rows": int(len(df)),
            "tickers": int(df["ticker"].nunique()),
            "scored": {
                col: int(df[col].notna().sum())
                for col in ("record_catalyst_score", "day_catalyst_score",
                            "predicted_direction", "tone",
                            "move_1d_pct", "max_favorable_pct", "outcome")
            },
            "sources": _source_stamp(),
        }
        STAMP_PATH.write_text(json.dumps(stamp, indent=2))
        logger.info("News library index built: %d rows, %d tickers -> %s",
                    stamp["rows"], stamp["tickers"], INDEX_PATH)
        return stamp


def ensure_index() -> dict[str, Any]:
    """Return the current stamp, rebuilding the index first if it is stale."""
    if index_is_current():
        return _read_stamp() or {}
    return build_index()


def _norm_ts(value: Any) -> Any:
    import pandas as pd

    if value in (None, ""):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def search(
    *,
    ticker: str | None = None,
    query: str | None = None,
    family: str | None = None,
    source: str | None = None,
    origin: str | None = None,
    relation: str | None = None,
    start: Any = None,
    end: Any = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Most-recent-first records matching the filters.

    ``ticker`` is pushed into the parquet read so a single-name query touches
    only its own row groups; the remaining filters are applied to that slice.
    Without a ticker the whole index is read (headline-sized columns only), which
    is what a free-text search across the library costs.
    """
    import pandas as pd

    ensure_index()
    limit = max(1, min(int(limit), MAX_LIMIT))
    offset = max(0, int(offset))
    ticker = (ticker or "").strip().upper() or None

    filters = [("ticker", "==", ticker)] if ticker else None
    df = pd.read_parquet(INDEX_PATH, filters=filters)

    df = _apply_filters(df, query=query, family=family, source=source,
                        origin=origin, relation=relation, start=start, end=end)

    total = int(len(df))
    # A ticker-filtered read is already newest-first from the build sort; a
    # cross-ticker read is not, so sort unconditionally rather than relying on it.
    page = df.sort_values("timestamp", ascending=False).iloc[offset:offset + limit]

    rows = []
    for rec in page.to_dict("records"):
        rec["timestamp"] = (
            None if pd.isna(rec["timestamp"]) else pd.Timestamp(rec["timestamp"]).isoformat()
        )
        rows.append(rec)
    return {"rows": rows, "total": total, "offset": offset, "limit": limit}


def _apply_filters(df, *, query, family, source, origin, relation, start, end):
    """Shared predicate for :func:`search` and :func:`event_days`."""
    if query:
        needle = query.strip().lower()
        if needle:
            df = df[df["headline"].fillna("").str.lower().str.contains(needle, regex=False)]
    for col, val in (("catalyst_family", family), ("source", source),
                     ("origin", origin), ("relation_type", relation)):
        if val:
            df = df[df[col].fillna("") == val]
    start_ts, end_ts = _norm_ts(start), _norm_ts(end)
    if start_ts is not None:
        df = df[df["timestamp"] >= start_ts]
    if end_ts is not None:
        df = df[df["timestamp"] <= end_ts]
    return df


def event_days(
    *,
    ticker: str,
    query: str | None = None,
    family: str | None = None,
    source: str | None = None,
    origin: str | None = None,
    relation: str | None = None,
    start: Any = None,
    end: Any = None,
    preview: int = 6,
    color_by: str = "catalyst_family",
    max_classes: int = 3,
) -> dict[str, Any]:
    """Per-session-day record counts for the chart overlay.

    Aggregates the WHOLE filtered match set rather than a page of it. Paging the
    records and counting the page would silently drop older news days off the
    chart — a 1Y plot of a heavily-covered name would show markers only on the
    most recent weeks and read as "no news before June", which is precisely the
    kind of chart that overstates what the data says.

    Days are bucketed in America/New_York: a 22:44 UTC headline is after-hours
    news on the *previous* ET session, and bucketing it by UTC date would park
    the marker a day late.

    ``color_by`` names the dimension the markers are coloured by. Only the
    ``max_classes`` most common values in the current view get their own colour;
    everything else is reported under the ``"Other"`` class. That cap is not
    cosmetic — the chart's marks are dots, which are compared all-pairs rather
    than only against their neighbours, and no 4th hue in the palette clears the
    colourblind/normal-vision separation floors against this surface. Exact
    values stay available in the tooltip and in the table.
    """
    import pandas as pd

    ensure_index()
    sym = (ticker or "").strip().upper()
    if not sym:
        return {"days": [], "classes": [], "color_by": color_by, "total": 0}
    field = color_by if color_by in COLOR_BY_FIELDS else "catalyst_family"
    cols = ["timestamp", "headline", "source", "catalyst_family", "origin",
            "relation_type"]
    df = pd.read_parquet(INDEX_PATH, columns=cols, filters=[("ticker", "==", sym)])
    df = _apply_filters(df, query=query, family=family, source=source,
                        origin=origin, relation=relation, start=start, end=end)
    if df.empty:
        return {"days": [], "classes": [], "color_by": field, "total": 0}

    df = df.sort_values("timestamp", ascending=False)
    df["day"] = df["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    df["klass"] = df[field].fillna("(none)").replace("", "(none)")

    counts = df["klass"].value_counts()
    named = list(counts.index[: max(0, int(max_classes))])
    classes = [{"name": str(k), "count": int(counts[k])} for k in named]
    other = int(counts.iloc[len(named):].sum())
    if other:
        classes.append({"name": OTHER_CLASS, "count": other})
    named_set = set(named)
    df["klass"] = df["klass"].where(df["klass"].isin(named_set), OTHER_CLASS)

    days: list[dict[str, Any]] = []
    for day, grp in df.groupby("day", sort=True):
        breakdown = grp["klass"].value_counts()
        head = grp.head(max(0, int(preview)))
        days.append({
            "day": day,
            "count": int(len(grp)),
            # The marker takes the day's dominant class; the tooltip shows the
            # full breakdown so a mixed day is never misread as pure.
            "klass": str(breakdown.index[0]),
            "breakdown": {str(k): int(v) for k, v in breakdown.items()},
            "headlines": [
                {"headline": r["headline"], "source": r["source"],
                 "family": r["catalyst_family"], "klass": r["klass"]}
                for r in head.to_dict("records")
            ],
        })
    return {"days": days, "classes": classes, "color_by": field, "total": int(len(df))}


def facets(ticker: str | None = None) -> dict[str, list[str]]:
    """Distinct filter values, scoped to a ticker when one is given."""
    import pandas as pd

    ensure_index()
    ticker = (ticker or "").strip().upper() or None
    filters = [("ticker", "==", ticker)] if ticker else None
    cols = ["catalyst_family", "source", "origin", "relation_type"]
    df = pd.read_parquet(INDEX_PATH, columns=cols, filters=filters)
    return {col: sorted(x for x in df[col].dropna().unique().tolist() if x) for col in cols}


def tickers(prefix: str = "", limit: int = 25) -> list[str]:
    """Ticker suggestions for the search box, most-covered first."""
    import pandas as pd

    ensure_index()
    df = pd.read_parquet(INDEX_PATH, columns=["ticker"])
    counts = df["ticker"].value_counts()
    prefix = (prefix or "").strip().upper()
    if prefix:
        counts = counts[counts.index.str.startswith(prefix)]
    return [str(t) for t in counts.index[: max(1, int(limit))]]


def price_series(ticker: str, *, days: int = 365) -> dict[str, Any]:
    """Daily OHLC bars for the chart, from the shared bar cache.

    Read-only against ``Data/shared/bars/1d`` — the same cache the 4H modules
    read — so the chart cannot drift from what the strategies see. Bars are
    session-stamped in UTC; no resampling, gap-filling, or adjustment happens
    here, so a missing day is a genuinely missing bar (holiday/halt/no coverage).
    """
    import pandas as pd

    sym = (ticker or "").strip().upper()
    path = BARS_1D_DIR / f"{sym}.parquet"
    if not sym or not path.exists():
        return {"ticker": sym, "bars": [], "error": f"no 1d bar cache for {sym or '(none)'}"}
    df = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if days and days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
        df = df[df["timestamp"] >= cutoff]
    bars = [
        {
            "t": pd.Timestamp(r["timestamp"]).isoformat(),
            "o": float(r["open"]), "h": float(r["high"]),
            "l": float(r["low"]), "c": float(r["close"]),
            "v": float(r["volume"]) if pd.notna(r["volume"]) else None,
        }
        for r in df.to_dict("records")
        if pd.notna(r["close"])
    ]
    return {"ticker": sym, "bars": bars}


def status() -> dict[str, Any]:
    stamp = _read_stamp() or {}
    return {
        "index_path": str(INDEX_PATH),
        "current": index_is_current(),
        "built_at": stamp.get("built_at"),
        "rows": stamp.get("rows"),
        "tickers": stamp.get("tickers"),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the news library search index.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if current.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(build_index(force=args.force), indent=2))


if __name__ == "__main__":
    main()
