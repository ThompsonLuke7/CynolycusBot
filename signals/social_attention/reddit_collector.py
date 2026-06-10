"""Historical Reddit collection and normalization."""

from __future__ import annotations

import hashlib
import logging
from datetime import timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from signals.social_attention.config import (
    CHECKPOINT_DIR,
    DEFAULT_SUBREDDITS,
    REDDIT_COMMENTS_PATH,
    REDDIT_POSTS_PATH,
    REDDIT_SUBMISSIONS_PATH,
    ensure_dirs,
)
from signals.social_attention.io import merge_write_table, read_json, read_table, write_json, write_table
from signals.social_attention.pullpush_client import PullPushClient

logger = logging.getLogger(__name__)


def _timestamp_from_utc(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, unit="s", utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(ts)


def _month_chunks(start: str | pd.Timestamp, end: str | pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    cur = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    stop = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while cur < stop:
        nxt = min(cur + pd.DateOffset(months=1), stop)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def _content_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:20]


def normalize_pullpush_item(kind: str, item: dict) -> dict[str, object]:
    reddit_id = str(item.get("id") or "").strip()
    source_id = f"reddit_{kind}:{reddit_id}"
    created_utc = item.get("created_utc")
    timestamp = _timestamp_from_utc(created_utc)
    title = str(item.get("title") or "")
    selftext = str(item.get("selftext") or "")
    body = str(item.get("body") or "")
    text = " ".join(part for part in (title, selftext, body) if part).strip()
    permalink = str(item.get("permalink") or "")
    if permalink and permalink.startswith("/"):
        url = "https://reddit.com" + permalink
    else:
        url = str(item.get("url") or permalink)

    row = {
        "post_id": source_id,
        "reddit_id": reddit_id,
        "source": "reddit",
        "kind": kind,
        "subreddit": str(item.get("subreddit") or "").strip(),
        "timestamp": timestamp,
        "created_utc": int(created_utc) if created_utc is not None else None,
        "title": title,
        "selftext": selftext,
        "body": body,
        "text": text,
        "author": str(item.get("author") or ""),
        "score": float(item.get("score") or 0.0),
        "num_comments": float(item.get("num_comments") or 0.0) if kind == "submission" else 0.0,
        "upvote_ratio": item.get("upvote_ratio"),
        "permalink": permalink,
        "url": url,
        "parent_id": str(item.get("parent_id") or ""),
        "link_id": str(item.get("link_id") or ""),
        "content_hash": _content_hash(text),
    }
    return row


def normalize_posts(
    *,
    submissions_path: Path | str = REDDIT_SUBMISSIONS_PATH,
    comments_path: Path | str = REDDIT_COMMENTS_PATH,
    output_path: Path | str = REDDIT_POSTS_PATH,
) -> pd.DataFrame:
    frames = []
    for path in (submissions_path, comments_path):
        frame = read_table(path)
        if not frame.empty:
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["post_id", "timestamp", "text"])
        out = out.drop_duplicates("post_id", keep="last").sort_values("timestamp").reset_index(drop=True)
    return write_table(out, output_path)


def _checkpoint_key(kind: str, subreddit: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{kind}:{subreddit}:{start.strftime('%Y-%m-%d')}:{end.strftime('%Y-%m-%d')}"


def collect_reddit_history(
    *,
    start: str,
    end: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    kinds: Iterable[str] = ("submission", "comment"),
    q: str | None = None,
    client: PullPushClient | None = None,
    resume: bool = True,
    checkpoint_path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    client = client or PullPushClient()
    checkpoint = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_DIR / "pullpush_reddit_history.json"
    state = read_json(checkpoint, default={"completed": []}) or {"completed": []}
    completed = set(state.get("completed", []))

    chunks = _month_chunks(start, end)
    for kind in kinds:
        if kind not in {"submission", "comment"}:
            raise ValueError("kinds must contain only 'submission' and/or 'comment'")
        output_path = REDDIT_SUBMISSIONS_PATH if kind == "submission" else REDDIT_COMMENTS_PATH
        for subreddit in subreddits:
            for chunk_start, chunk_end in chunks:
                key = _checkpoint_key(kind, subreddit, chunk_start, chunk_end)
                if resume and key in completed:
                    continue
                rows = [
                    normalize_pullpush_item(kind, item)
                    for item in client.iter_search(
                        kind=kind,
                        subreddit=subreddit,
                        after=int(chunk_start.timestamp()),
                        before=int(chunk_end.timestamp()),
                        q=q,
                    )
                ]
                frame = pd.DataFrame(rows)
                if not frame.empty:
                    merge_write_table(frame, output_path, dedupe_cols=["post_id"])
                completed.add(key)
                state["completed"] = sorted(completed)
                write_json(state, checkpoint)
                logger.info("Collected %s %s rows for r/%s %s", len(frame), kind, subreddit, chunk_start.date())

    posts = normalize_posts()
    submissions = read_table(REDDIT_SUBMISSIONS_PATH)
    comments = read_table(REDDIT_COMMENTS_PATH)
    logger.info("Normalized reddit_posts=%s submissions=%s comments=%s", len(posts), len(submissions), len(comments))
    return submissions, comments

