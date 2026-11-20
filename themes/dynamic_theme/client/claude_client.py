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

from themes.dynamic_theme.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_FALLBACK_MODEL,
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL,
)

logger = logging.getLogger(__name__)

# Server-side refusal fallback (beta). Only models listed here get one.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACKS = {CLAUDE_MODEL: CLAUDE_FALLBACK_MODEL}


class ClaudeResponseError(RuntimeError):
    """The call completed but produced no usable answer (refusal or truncation).

    Not retried: the same request gets the same outcome. Unlike
    ``ClaudeUnavailableError`` it concerns one request, so callers may shrink
    the request (e.g. one cluster per call) and try again.
    """


class ClaudeUnavailableError(RuntimeError):
    """The API refused the request for a reason no retry can fix.

    Covers 400 (including "credit balance is too low"), 401, 403 and 404. On
    2026-11-16 the prepaid balance ran out mid-run and every remaining cluster
    still burned three attempts plus backoff before falling back to a
    placeholder label. Callers should stop the stage on this error, not degrade.
    """


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


def _response_text(message: Any, *, structured: bool) -> str:
    """Join the text blocks of a final message, refusing unusable stops.

    Read by block type, never position: with adaptive thinking the first block
    is a ``thinking`` block, and a fallback adds a ``fallback`` block. A
    truncated structured answer is invalid by construction, so it raises; a
    truncated plain answer is returned as is (step07 salvages complete
    objects out of it).
    """
    if message.stop_reason == "refusal":
        raise ClaudeResponseError("Claude declined the request (stop_reason=refusal)")
    if structured and message.stop_reason == "max_tokens":
        raise ClaudeResponseError("Claude response hit max_tokens before finishing")
    return "".join(block.text for block in message.content if block.type == "text")


def _log_usage(message: Any) -> None:
    """One INFO line per call; ``record.claude_usage`` carries the numbers."""
    usage = getattr(message, "usage", None)
    if usage is None:
        return
    numbers = {
        "model": getattr(message, "model", "?"),
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }
    logger.info("Claude usage: %(model)s in=%(input_tokens)d out=%(output_tokens)d", numbers,
                extra={"claude_usage": numbers})


def _request_params(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    system: str | None,
    effort: str | None,
    json_schema: dict | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        params["system"] = system
    output_config: dict[str, Any] = {}
    if effort:
        output_config["effort"] = effort
    if json_schema:
        output_config["format"] = {"type": "json_schema", "schema": json_schema}
    if output_config:
        params["output_config"] = output_config
    if model in _FALLBACKS:
        params["betas"] = [_FALLBACK_BETA]
        params["fallbacks"] = [{"model": _FALLBACKS[model]}]
    return params


def call_claude(
    prompt: str,
    *,
    system: str | None = None,
    model: str = CLAUDE_MODEL,
    max_tokens: int = CLAUDE_MAX_TOKENS,
    effort: str | None = None,
    json_schema: dict | None = None,
    retries: int = 3,
    backoff_base: float = 2.0,
) -> str:
    """Call Claude and return the response text.

    Retries with exponential backoff on transient errors. Raises
    ``ClaudeUnavailableError`` at once, without retrying, when the API rejects
    the request itself (no credit, bad key, no access, unknown model), and
    ``ClaudeResponseError`` on a refusal or a truncated answer. ``json_schema``
    turns on structured outputs: the text is then valid JSON for that schema.
    """
    client = _client()
    import anthropic

    non_retryable = (
        anthropic.BadRequestError,
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
    )
    params = _request_params(
        prompt, model=model, max_tokens=max_tokens, system=system,
        effort=effort, json_schema=json_schema,
    )
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            # Stream so large max_tokens (relationship graph, thinking) don't trip
            # the SDK's "streaming required for long requests" guard. Beta
            # endpoint: the refusal fallback is a beta parameter.
            with client.beta.messages.stream(**params) as stream:
                message = stream.get_final_message()
            _log_usage(message)
            return _response_text(message, structured=json_schema is not None)
        except non_retryable as exc:
            raise ClaudeUnavailableError(
                f"Claude API refused the request ({type(exc).__name__}): {exc}"
            ) from exc
        except ClaudeResponseError:
            raise
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
    system: str | None = None,
    model: str = CLAUDE_MODEL,
    max_tokens: int = CLAUDE_MAX_TOKENS,
    effort: str | None = None,
    json_schema: dict | None = None,
    retries: int = 3,
) -> Any:
    """Call Claude and parse the response as JSON.

    With ``json_schema`` the API guarantees schema-valid JSON. Without it,
    strips markdown fences if Claude wraps the JSON in ```json ... ```, and
    falls back to extracting an embedded JSON payload when the response opens
    with prose. A response carrying no parseable JSON still raises — callers
    must not receive a silently degraded result.
    """
    raw = call_claude(
        prompt, system=system, model=model, max_tokens=max_tokens,
        effort=effort, json_schema=json_schema, retries=retries,
    )
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
