"""Optional PRAW enrichment for Reddit records."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from social_attention.config import REDDIT_CLIENT_ID_ENV, REDDIT_CLIENT_SECRET_ENV, REDDIT_USER_AGENT_ENV


def _read_env_file(path: str | os.PathLike | None = ".env") -> dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    values: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _env_value(file_values: dict[str, str], key: str) -> str | None:
    return file_values.get(key) or os.getenv(key)


def reddit_from_env(env_file: str | os.PathLike | None = ".env"):
    try:
        import praw
    except ImportError as exc:
        raise ImportError("Install praw to use PRAW Reddit enrichment.") from exc

    values = _read_env_file(env_file)
    client_id = _env_value(values, REDDIT_CLIENT_ID_ENV)
    client_secret = _env_value(values, REDDIT_CLIENT_SECRET_ENV)
    user_agent = _env_value(values, REDDIT_USER_AGENT_ENV)
    if not client_id or not client_secret or not user_agent:
        raise ValueError(
            f"Set {REDDIT_CLIENT_ID_ENV}, {REDDIT_CLIENT_SECRET_ENV}, and {REDDIT_USER_AGENT_ENV}."
        )
    return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)


def _safe_author_name(obj: object) -> str:
    author = getattr(obj, "author", None)
    return str(getattr(author, "name", "") or author or "")


def enrich_posts_with_praw(posts: pd.DataFrame, *, reddit=None, env_file: str | os.PathLike | None = ".env") -> pd.DataFrame:
    if posts.empty:
        return posts.copy()
    reddit = reddit or reddit_from_env(env_file)
    rows = []
    for _, row in posts.iterrows():
        kind = str(row.get("kind") or "")
        reddit_id = str(row.get("reddit_id") or "").strip()
        if not reddit_id:
            continue
        try:
            obj = reddit.submission(id=reddit_id) if kind == "submission" else reddit.comment(id=reddit_id)
            rows.append(
                {
                    "post_id": row["post_id"],
                    "praw_score": float(getattr(obj, "score", np.nan)),
                    "praw_num_comments": float(getattr(obj, "num_comments", np.nan)) if kind == "submission" else np.nan,
                    "praw_upvote_ratio": float(getattr(obj, "upvote_ratio", np.nan)) if kind == "submission" else np.nan,
                    "praw_permalink": str(getattr(obj, "permalink", "") or ""),
                    "praw_author": _safe_author_name(obj),
                    "praw_removed": bool(str(getattr(obj, "selftext", "") or getattr(obj, "body", "") or "").strip().lower() in {"[removed]", "[deleted]"}),
                }
            )
        except Exception:
            rows.append(
                {
                    "post_id": row["post_id"],
                    "praw_score": np.nan,
                    "praw_num_comments": np.nan,
                    "praw_upvote_ratio": np.nan,
                    "praw_permalink": "",
                    "praw_author": "",
                    "praw_removed": np.nan,
                }
            )
    enrich = pd.DataFrame(rows)
    if enrich.empty:
        return posts.copy()
    return posts.merge(enrich, on="post_id", how="left")

