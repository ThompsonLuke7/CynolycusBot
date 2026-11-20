"""Polite read-only client for AfterHour's unofficial public endpoints.

There is no official API: endpoints were documented by a third party
(github.com/mphinance/aftermath, AFTERHOUR_API_SPEC.md). AfterHour's Terms
prohibit automated scraping, so this client is single-threaded, rate limited,
never authenticates, and never writes (no posts, reactions or follows).
"""
from __future__ import annotations

import time
from typing import Any

import requests

BASE = "https://api.afterhour.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


class AfterHourClient:
    def __init__(self, min_interval_s: float = 1.2, retries: int = 6, timeout_s: int = 60):
        self.min_interval_s = min_interval_s
        self.retries = retries
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Accept": "application/json"})
        self._last = 0.0
        self.requests_made = 0

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET JSON; retry 5xx/timeouts with backoff (the gateway 504s under load)."""
        for attempt in range(self.retries):
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests_made += 1
            try:
                r = self.session.get(BASE + path, params=params, timeout=self.timeout_s)
            except (requests.ConnectionError, requests.Timeout):
                if attempt == self.retries - 1:
                    raise
            else:
                if r.status_code < 500:
                    r.raise_for_status()
                    return r.json()
                if attempt == self.retries - 1:
                    r.raise_for_status()
            time.sleep(min(1.5 * 2 ** attempt, 20))
        raise RuntimeError("unreachable")

    def profile(self, username: str) -> dict:
        return self.get("/public/profile", {"username": username})

    def feed_page(self, take: int, cursor: int | str | None = None, **filters: Any) -> dict:
        params = {"contentTypes": "post", "sortMethod": "newest", "take": take, **filters}
        if cursor is not None:
            params["cursor"] = cursor
        return self.get("/social/feed", params)

    def post(self, post_hash: str) -> dict:
        return self.get(f"/public/post/{post_hash}")
