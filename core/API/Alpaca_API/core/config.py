from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _split_env_file_profile(path: str | os.PathLike | None) -> tuple[str | os.PathLike | None, str | None]:
    if path is None:
        return None, None
    text = str(path)
    if "#" not in text:
        return path, None
    env_path, profile = text.rsplit("#", 1)
    return env_path or ".env", profile.strip().upper() or None


def _read_env_file(path: str | os.PathLike = ".env") -> dict[str, str]:
    """
    Minimal .env reader (no external deps).
    Lines beginning with # are ignored. Only KEY=VALUE pairs are parsed.
    """
    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            values[key] = val
    return values


def _load_env_file(path: str | os.PathLike = ".env") -> None:
    """
    Backward-compatible process env loader. New code should prefer
    AlpacaConfig.from_env(env_file=...) so multiple accounts can coexist.
    """
    for key, val in _read_env_file(path).items():
        if key not in os.environ:
            os.environ[key] = val


def _env_value(file_values: dict[str, str], *names: str) -> str | None:
    for name in names:
        if file_values.get(name):
            return file_values[name]
    for name in names:
        if os.getenv(name):
            return os.getenv(name)
    return None


def _profile_env_value(file_values: dict[str, str], profile: str | None, *names: str) -> str | None:
    if profile:
        profiled_names = tuple(f"{name}_{profile}" for name in names) + tuple(
            f"{profile}_{name}" for name in names
        )
        value = _env_value(file_values, *profiled_names)
        if value:
            return value
    return _env_value(file_values, *names)


@dataclass
class AlpacaConfig:
    """
    Holds Alpaca credentials/URLs. Values are read from environment variables.
      APCA_API_KEY_ID (preferred)
      APCA_API_SECRET_KEY (preferred)
      ALPACA_API_KEY (fallback)
      ALPACA_SECRET_KEY (fallback)
      APCA_API_BASE_URL (optional, defaults to https://data.alpaca.markets)
    """

    key_id: str
    secret_key: str
    base_url: str = "https://data.alpaca.markets"

    @classmethod
    def from_env(cls, env_file: Optional[str | os.PathLike] = ".env") -> "AlpacaConfig":
        # Prefer the explicit env file for this client instance, then fall back
        # to process env. This lets paper and live clients run side-by-side.
        env_path, profile = _split_env_file_profile(env_file)
        file_values = _read_env_file(env_path) if env_path else {}

        key = _profile_env_value(file_values, profile, "APCA_API_KEY_ID", "ALPACA_API_KEY", "ALPACA_API_KEY_ID")
        secret = _profile_env_value(file_values, profile, "APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY", "ALPACA_API_SECRET_KEY")
        base = _profile_env_value(file_values, profile, "APCA_API_BASE_URL") or cls.base_url

        if not key or not secret:
            raise ValueError(
                "Missing Alpaca credentials. Set APCA_API_KEY_ID/APCA_API_SECRET_KEY "
                "or ALPACA_API_KEY/ALPACA_SECRET_KEY in your environment."
            )

        return cls(key_id=key, secret_key=secret, base_url=base)
