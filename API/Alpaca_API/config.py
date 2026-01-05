from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _load_env_file(path: str | os.PathLike = ".env") -> None:
    """
    Minimal .env reader (no external deps). Sets env vars if not already set.
    Lines beginning with # are ignored. Only KEY=VALUE pairs are parsed.
    """
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


@dataclass
class AlpacaConfig:
    """
    Holds Alpaca credentials/URLs. Values are read from environment variables.
      APCA_API_KEY_ID
      APCA_API_SECRET_KEY
      APCA_API_BASE_URL (optional, defaults to https://data.alpaca.markets)
    """

    key_id: str
    secret_key: str
    base_url: str = "https://data.alpaca.markets"

    @classmethod
    def from_env(cls, env_file: Optional[str | os.PathLike] = ".env") -> "AlpacaConfig":
        # Load local .env first (without overriding existing environment)
        if env_file:
            _load_env_file(env_file)

        key = os.getenv("APCA_API_KEY_ID")
        secret = os.getenv("APCA_API_SECRET_KEY")
        base = os.getenv("APCA_API_BASE_URL", cls.base_url)

        if not key or not secret:
            raise ValueError(
                "Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your environment."
            )

        return cls(key_id=key, secret_key=secret, base_url=base)
