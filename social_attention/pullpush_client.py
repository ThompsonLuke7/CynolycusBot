"""PullPush Reddit archive client."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterator

import requests

from social_attention.config import DEFAULT_PULLPUSH_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class PullPushClient:
    """Thin client for PullPush comment/submission search endpoints."""

    base_url: str = DEFAULT_PULLPUSH_BASE_URL
    timeout: float = 30.0
    sleep_seconds: float = 1.0
    max_retries: int = 4
    session: requests.Session = field(default_factory=requests.Session)

    def _endpoint(self, kind: str) -> str:
        normalized = kind.lower().strip()
        if normalized not in {"comment", "submission"}:
            raise ValueError("kind must be 'comment' or 'submission'")
        return f"{self.base_url.rstrip('/')}/reddit/search/{normalized}/"

    def search(
        self,
        *,
        kind: str,
        subreddit: str | None = None,
        after: int | None = None,
        before: int | None = None,
        q: str | None = None,
        ids: str | None = None,
        link_id: str | None = None,
        size: int = 100,
        sort: str = "asc",
        sort_type: str = "created_utc",
    ) -> dict:
        params: dict[str, object] = {
            "size": min(int(size), 100),
            "sort": sort,
            "sort_type": sort_type,
        }
        if subreddit:
            params["subreddit"] = subreddit
        if after is not None:
            params["after"] = int(after)
        if before is not None:
            params["before"] = int(before)
        if q:
            params["q"] = q
        if ids:
            params["ids"] = ids
        if link_id:
            params["link_id"] = link_id

        url = self._endpoint(kind)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(f"{response.status_code} from PullPush")
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = self.sleep_seconds * (2 ** attempt)
                logger.warning("PullPush request failed (%s); retrying in %.1fs", exc, delay)
                time.sleep(delay)
        raise RuntimeError(f"PullPush request failed after retries: {last_exc}") from last_exc

    def iter_search(
        self,
        *,
        kind: str,
        subreddit: str | None,
        after: int,
        before: int,
        q: str | None = None,
        size: int = 100,
    ) -> Iterator[dict]:
        cursor = int(after)
        end = int(before)
        while cursor < end:
            payload = self.search(
                kind=kind,
                subreddit=subreddit,
                after=cursor,
                before=end,
                q=q,
                size=size,
                sort="asc",
                sort_type="created_utc",
            )
            rows = payload.get("data") or []
            if not rows:
                break
            max_created = cursor
            for row in rows:
                created = int(row.get("created_utc") or 0)
                if created <= cursor and len(rows) == 1:
                    continue
                max_created = max(max_created, created)
                yield row
            if max_created <= cursor:
                break
            cursor = max_created + 1
            time.sleep(self.sleep_seconds)

