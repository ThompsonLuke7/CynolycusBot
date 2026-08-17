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
            # Stream so large max_tokens (relationship graph) don't trip the
            # SDK's "streaming required for long requests" guard. Streaming is
            # equally fine for the small labeling calls.
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                return stream.get_final_message().content[0].text
        except Exception as exc:
            last_exc = exc
            wait = backoff_base ** attempt
            logger.warning("Claude call failed (attempt %d/%d): %s — retrying in %.1fs", attempt + 1, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Claude call failed after {retries} attempts") from last_exc


def _extract_json_payload(text: str) -> str | None:
    """Return the outermost JSON object/array embedded in ``text``.

    Ambiguous clusters draw a reasoning preamble before the JSON ("Looking at
    this cluster: - **CMPR** (Cimpress) ..."), which fence-stripping alone does
    not handle. On 2026-08-17 that silently cost 12 of 186 clusters their
    labels. Scanning for a balanced delimiter run — while ignoring braces and
    brackets inside strings — recovers the payload without loosening what
    counts as valid JSON.
    """
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def call_claude_json(
    prompt: str,
    *,
    model: str = CLAUDE_MODEL,
    max_tokens: int = CLAUDE_MAX_TOKENS,
    retries: int = 3,
) -> Any:
    """Call Claude and parse the response as JSON.

    Strips markdown fences if Claude wraps the JSON in ```json ... ```, and
    falls back to extracting an embedded JSON payload when the response opens
    with prose. A response carrying no parseable JSON still raises — callers
    must not receive a silently degraded result.
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
        payload = _extract_json_payload(text)
        if payload is not None:
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                pass
        logger.error("Failed to parse Claude response as JSON: %s\nRaw: %s", exc, raw[:500])
        raise
