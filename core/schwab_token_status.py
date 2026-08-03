"""Read-only status of the Schwab OAuth refresh token.

WHY: Schwab refresh tokens expire on a fixed 7-day clock and renewal is an
*interactive* browser login — nothing in an unattended run can satisfy it. When
one lapses, every Schwab-backed job fails the same way and keeps retrying: on
2026-07-30 the dealer-positioning runner logged 535 consecutive
``invalid_grant`` failures across five underlyings between 09:31 and the crash,
and the whole day's dealer chain capture was lost. The expiry was entirely
predictable — the token had been minted on 2026-07-22 23:30 ET — but nothing
looked at it until things were already broken.

This module answers "how long do I have left?" from the token file alone. It
performs no network I/O, needs no credentials, and never mutates the token, so
it is safe to call on every startup.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

# Schwab's documented refresh-token lifetime. The access token inside the same
# file rotates automatically and is not what expires here.
REFRESH_TOKEN_LIFETIME = dt.timedelta(days=7)

# Warn this far ahead so a renewal can be scheduled deliberately rather than
# discovered from a wall of 400s mid-session.
WARN_WITHIN = dt.timedelta(days=2)

DEFAULT_TOKEN_PATH = (
    Path(__file__).resolve().parents[1] / "core/API/Schwab_API/schwab_token.json"
)

REAUTH_COMMAND = ".venv/bin/python core/API/Schwab_API/schwab_client.py --reauth"


@dataclass(frozen=True)
class SchwabTokenStatus:
    """Everything derivable about the refresh token without calling Schwab."""

    path: Path
    found: bool
    issued_at: dt.datetime | None
    expires_at: dt.datetime | None
    remaining: dt.timedelta | None
    #: ``True`` when the token is already dead — every Schwab job will fail.
    expired: bool
    #: ``True`` when it expires inside :data:`WARN_WITHIN`.
    expiring_soon: bool
    #: Human-readable one-liner, always populated.
    message: str

    @property
    def severity(self) -> str:
        """``"error"`` | ``"warning"`` | ``"info"`` — how loudly to report."""
        if not self.found or self.expired:
            return "error"
        return "warning" if self.expiring_soon else "info"


def _humanize(delta: dt.timedelta) -> str:
    total = int(abs(delta).total_seconds())
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def schwab_token_status(
    token_path: Path | str | None = None,
    *,
    now: dt.datetime | None = None,
) -> SchwabTokenStatus:
    """Report refresh-token expiry from ``schwab_token.json``.

    ``creation_timestamp`` is the epoch second at which the *refresh* token was
    minted by the manual login flow. The file's mtime is not a substitute: an
    access-token refresh rewrites the file without extending the refresh token,
    so mtime moves while the real deadline does not.
    """
    path = Path(token_path) if token_path is not None else DEFAULT_TOKEN_PATH
    now = now or dt.datetime.now(dt.timezone.utc)

    if not path.exists():
        return SchwabTokenStatus(
            path=path,
            found=False,
            issued_at=None,
            expires_at=None,
            remaining=None,
            expired=True,
            expiring_soon=False,
            message=(
                f"Schwab token file not found at {path} — every Schwab job will "
                f"fail. Re-auth with: {REAUTH_COMMAND}"
            ),
        )

    try:
        payload = json.loads(path.read_text())
        created = float(payload["creation_timestamp"])
    except Exception as exc:  # noqa: BLE001 - any malformed token is equally fatal
        return SchwabTokenStatus(
            path=path,
            found=True,
            issued_at=None,
            expires_at=None,
            remaining=None,
            expired=True,
            expiring_soon=False,
            message=(
                f"Schwab token at {path} is unreadable ({exc}) — treat as expired. "
                f"Re-auth with: {REAUTH_COMMAND}"
            ),
        )

    issued_at = dt.datetime.fromtimestamp(created, dt.timezone.utc)
    expires_at = issued_at + REFRESH_TOKEN_LIFETIME
    remaining = expires_at - now
    expired = remaining <= dt.timedelta(0)
    expiring_soon = not expired and remaining <= WARN_WITHIN

    local_expiry = expires_at.astimezone()
    stamp = local_expiry.strftime("%Y-%m-%d %H:%M %Z")
    if expired:
        message = (
            f"Schwab refresh token EXPIRED {_humanize(remaining)} ago "
            f"(expiry {stamp}); dealer positioning and every other Schwab job "
            f"will fail until re-auth: {REAUTH_COMMAND}"
        )
    elif expiring_soon:
        message = (
            f"Schwab refresh token expires in {_humanize(remaining)} ({stamp}) — "
            f"renew soon: {REAUTH_COMMAND}"
        )
    else:
        message = f"Schwab refresh token valid for {_humanize(remaining)} (expires {stamp})"

    return SchwabTokenStatus(
        path=path,
        found=True,
        issued_at=issued_at,
        expires_at=expires_at,
        remaining=remaining,
        expired=expired,
        expiring_soon=expiring_soon,
        message=message,
    )


def banner_line(status: SchwabTokenStatus | None = None) -> str:
    """One aligned line for the combined_server startup banner."""
    status = status or schwab_token_status()
    if not status.found or status.expired:
        return f"  Schwab token:            EXPIRED — re-auth needed ({REAUTH_COMMAND})"
    stamp = status.expires_at.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    flag = "  <-- RENEW SOON" if status.expiring_soon else ""
    return f"  Schwab token:            {_humanize(status.remaining)} left (expires {stamp}){flag}"


if __name__ == "__main__":  # pragma: no cover - operator convenience
    st = schwab_token_status()
    print(st.message)
    raise SystemExit(0 if st.found and not st.expired else 1)
