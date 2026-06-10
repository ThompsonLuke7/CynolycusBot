"""Thin wrapper around the Anthropic SDK for dynamic_theme labeling calls.

LLM is ONLY used for:
  - Theme naming
  - Theme hierarchy
  - Theme relationships

Never used for trading decisions.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from themes.dynamic_theme.config import ANTHROPIC_API_KEY, CLAUDE_MAX_TOKENS, CLAUDE_MODEL

logger = logging.getLogger(__name__)


def _client():
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("Install anthropic: pip install anthropic") from exc
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file and reload the shell."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def call_claude(
    prompt: str,
    *,
    model: str = CLAUDE_MODEL,
    max_tokens: int = CLAUDE_MAX_TOKENS,
    retries: int = 3,
    backoff_base: float = 2.0,
) -> str:
    """Call Claude and return the raw text response.

    Retries with exponential backoff on transient errors.
    """
    client = _client()
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as exc:
            last_exc = exc
            wait = backoff_base ** attempt
            logger.warning("Claude call failed (attempt %d/%d): %s — retrying in %.1fs", attempt + 1, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Claude call failed after {retries} attempts") from last_exc


def call_claude_json(
    prompt: str,
    *,
    model: str = CLAUDE_MODEL,
    max_tokens: int = CLAUDE_MAX_TOKENS,
    retries: int = 3,
) -> Any:
    """Call Claude and parse the response as JSON.

    Strips markdown fences if Claude wraps the JSON in ```json ... ```.
    """
    raw = call_claude(prompt, model=model, max_tokens=max_tokens, retries=retries)
    text = raw.strip()
    # strip optional markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Claude response as JSON: %s\nRaw: %s", exc, raw[:500])
        raise
