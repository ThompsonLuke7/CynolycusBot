"""Streaming JSON, JSONL, and CSV parsers with exact record locators."""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Iterator
from typing import BinaryIO


# A JSON document or JSONL record above this bound is quarantined before
# ``json.loads`` receives it.  The source artifact hash remains authoritative
# for the complete oversized bytes.
MAX_JSON_DOCUMENT_BYTES = 64 * 1024 * 1024
MAX_JSON_RECORD_BYTES = 64 * 1024 * 1024
_INVALID_UTF8_PREFIX = "base64-raw-bytes:"


@dataclass(frozen=True)
class RawImportItem:
    source_path: Path
    record_locator: str
    raw_payload: dict[str, object]
    raw_text: str | None = None


@dataclass(frozen=True)
class ParseIssue:
    source_path: Path
    record_locator: str
    error_code: str
    error_message: str
    raw_text: str | None
    skippable: bool = False


def _decode(raw_bytes: bytes) -> str:
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # PostgreSQL text cannot carry surrogate escapes.  Encode invalid
        # source bytes as an unambiguous ASCII transport representation that
        # can be restored with ``raw_text_to_bytes``.
        return _INVALID_UTF8_PREFIX + base64.b64encode(raw_bytes).decode("ascii")


def raw_text_to_bytes(raw_text: str) -> bytes:
    """Restore a quarantined raw-text transport value to its source bytes."""

    if raw_text.startswith(_INVALID_UTF8_PREFIX):
        encoded = raw_text[len(_INVALID_UTF8_PREFIX) :]
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise ValueError("invalid raw-byte quarantine transport") from exc
    return raw_text.encode("utf-8")


def _json_item(path: Path, locator: str, raw_bytes: bytes) -> RawImportItem | ParseIssue:
    raw_text = _decode(raw_bytes)
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return ParseIssue(
            source_path=path,
            record_locator=locator,
            error_code="MALFORMED_JSON",
            error_message=str(exc),
            raw_text=raw_text,
        )
    if not isinstance(payload, dict):
        return ParseIssue(
            source_path=path,
            record_locator=locator,
            error_code="JSON_OBJECT_REQUIRED",
            error_message="historical import records must be JSON objects",
            raw_text=raw_text,
        )
    return RawImportItem(path, locator, payload, raw_text)


def _iter_jsonl_events(path: Path, source_file: BinaryIO) -> Iterator[RawImportItem | ParseIssue]:
    for line_number, raw_line in enumerate(_bounded_lines(source_file), start=1):
        locator = f"line:{line_number}"
        if raw_line is None:
            yield ParseIssue(
                source_path=path,
                record_locator=locator,
                error_code="JSON_RECORD_TOO_LARGE",
                error_message=(
                    f"JSONL record exceeds MAX_JSON_RECORD_BYTES={MAX_JSON_RECORD_BYTES}"
                ),
                raw_text=None,
            )
            continue
        raw_text = _decode(raw_line)
        if not raw_text.strip():
            yield ParseIssue(
                source_path=path,
                record_locator=locator,
                error_code="BLANK_LINE",
                error_message="blank JSONL line skipped",
                raw_text=raw_text,
                skippable=True,
            )
            continue
        yield _json_item(path, locator, raw_line)


def _bounded_lines(source_file: BinaryIO) -> Iterator[bytes | None]:
    """Yield complete bounded lines, draining oversized lines incrementally."""

    while True:
        raw_line = source_file.readline(MAX_JSON_RECORD_BYTES + 1)
        if not raw_line:
            return
        if len(raw_line) <= MAX_JSON_RECORD_BYTES:
            yield raw_line
            continue
        # Continue in bounded reads until the oversized record terminates so
        # later JSONL records retain their correct locators.
        while raw_line and not raw_line.endswith(b"\n"):
            raw_line = source_file.readline(MAX_JSON_RECORD_BYTES + 1)
        yield None


def _iter_json_events(path: Path, source_file: BinaryIO) -> Iterator[RawImportItem | ParseIssue]:
    try:
        raw_bytes = source_file.read(MAX_JSON_DOCUMENT_BYTES + 1)
    except OSError as exc:
        yield ParseIssue(
            source_path=path,
            record_locator="document:1",
            error_code="UNREADABLE_SOURCE",
            error_message=str(exc),
            raw_text=None,
        )
        return
    if len(raw_bytes) > MAX_JSON_DOCUMENT_BYTES:
        yield ParseIssue(
            source_path=path,
            record_locator="document:1",
            error_code="JSON_DOCUMENT_TOO_LARGE",
            error_message=(
                f"JSON document exceeds MAX_JSON_DOCUMENT_BYTES={MAX_JSON_DOCUMENT_BYTES}"
            ),
            raw_text=None,
        )
        return
    yield _json_item(path, "document:1", raw_bytes)


class _TrackedCsvLines:
    def __init__(self, source_file: BinaryIO):
        self._source_file = source_file
        self._raw_lines: list[bytes] = []

    def __iter__(self) -> "_TrackedCsvLines":
        return self

    def __next__(self) -> str:
        raw_line = self._source_file.readline()
        if not raw_line:
            raise StopIteration
        self._raw_lines.append(raw_line)
        return _decode(raw_line)

    def take_raw_text(self) -> str:
        raw_text = _decode(b"".join(self._raw_lines))
        self._raw_lines.clear()
        return raw_text


def _iter_csv_events(path: Path, source_file: BinaryIO) -> Iterator[RawImportItem | ParseIssue]:
    try:
        tracked_lines = _TrackedCsvLines(source_file)
        reader = csv.DictReader(tracked_lines)
        if reader.fieldnames is None:
            yield ParseIssue(
                source_path=path,
                record_locator="row:1",
                error_code="CSV_HEADER_REQUIRED",
                error_message="CSV source must contain a header row",
                raw_text=tracked_lines.take_raw_text(),
            )
            return
        tracked_lines.take_raw_text()  # header is not a data row
        for row_number, row in enumerate(reader, start=1):
            raw_text = tracked_lines.take_raw_text()
            if None in row:
                yield ParseIssue(
                    source_path=path,
                    record_locator=f"row:{row_number}",
                    error_code="CSV_EXTRA_COLUMNS",
                    error_message="CSV row contains more fields than its header",
                    raw_text=raw_text,
                )
                continue
            payload = {str(key): value for key, value in row.items() if key is not None}
            yield RawImportItem(
                source_path=path,
                record_locator=f"row:{row_number}",
                raw_payload=payload,
                raw_text=raw_text,
            )
    except (OSError, csv.Error, UnicodeError) as exc:
        yield ParseIssue(
            source_path=path,
            record_locator="row:1",
            error_code="MALFORMED_CSV",
            error_message=str(exc),
            raw_text=None,
        )


def iter_source_events(
    path: Path,
    source_file: BinaryIO | None = None,
) -> Iterator[RawImportItem | ParseIssue]:
    """Stream one source, optionally from a caller-owned immutable snapshot."""

    source_path = Path(path)
    suffix = source_path.suffix.lower()

    def iter_open_file(open_file: BinaryIO) -> Iterator[RawImportItem | ParseIssue]:
        if suffix == ".jsonl":
            yield from _iter_jsonl_events(source_path, open_file)
        elif suffix == ".json":
            yield from _iter_json_events(source_path, open_file)
        elif suffix == ".csv":
            yield from _iter_csv_events(source_path, open_file)
        else:
            yield ParseIssue(
                source_path=source_path,
                record_locator="document:1",
                error_code="UNSUPPORTED_SOURCE_FORMAT",
                error_message=f"unsupported source suffix: {source_path.suffix}",
                raw_text=None,
            )

    if source_file is not None:
        yield from iter_open_file(source_file)
        return
    try:
        with source_path.open("rb") as opened_file:
            yield from iter_open_file(opened_file)
    except OSError as exc:
        yield ParseIssue(
            source_path=source_path,
            record_locator="document:1",
            error_code="UNREADABLE_SOURCE",
            error_message=str(exc),
            raw_text=None,
        )
def iter_raw_import_items(path: Path) -> Iterator[RawImportItem]:
    for event in iter_source_events(Path(path)):
        if isinstance(event, ParseIssue):
            raise ValueError(
                f"{event.source_path} {event.record_locator}: "
                f"{event.error_code}: {event.error_message}"
            )
        yield event


def parse_jsonl(path: Path) -> Iterator[RawImportItem]:
    yield from iter_raw_import_items(Path(path))


def parse_json(path: Path) -> Iterator[RawImportItem]:
    yield from iter_raw_import_items(Path(path))


def parse_csv(path: Path) -> Iterator[RawImportItem]:
    yield from iter_raw_import_items(Path(path))


__all__ = [
    "MAX_JSON_DOCUMENT_BYTES",
    "MAX_JSON_RECORD_BYTES",
    "ParseIssue",
    "RawImportItem",
    "iter_raw_import_items",
    "iter_source_events",
    "parse_csv",
    "parse_json",
    "parse_jsonl",
    "raw_text_to_bytes",
]
