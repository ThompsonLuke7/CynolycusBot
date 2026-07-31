"""Streaming JSON, JSONL, and CSV parsers with exact record locators."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path
from collections.abc import Iterator


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
    return raw_bytes.decode("utf-8", errors="replace")


def _json_item(path: Path, locator: str, raw_text: str) -> RawImportItem | ParseIssue:
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


def _iter_jsonl_events(path: Path) -> Iterator[RawImportItem | ParseIssue]:
    with path.open("rb") as source_file:
        for line_number, raw_line in enumerate(source_file, start=1):
            raw_text = _decode(raw_line)
            locator = f"line:{line_number}"
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
            yield _json_item(path, locator, raw_text)


def parse_jsonl(path: Path) -> Iterator[RawImportItem]:
    for event in _iter_jsonl_events(Path(path)):
        if isinstance(event, ParseIssue):
            raise ValueError(
                f"{event.source_path} {event.record_locator}: "
                f"{event.error_code}: {event.error_message}"
            )
        yield event


def _iter_json_events(path: Path) -> Iterator[RawImportItem | ParseIssue]:
    source_path = Path(path)
    try:
        raw_bytes = source_path.read_bytes()
    except OSError as exc:
        yield ParseIssue(
            source_path=source_path,
            record_locator="document:1",
            error_code="UNREADABLE_SOURCE",
            error_message=str(exc),
            raw_text=None,
        )
        return
    yield _json_item(source_path, "document:1", _decode(raw_bytes))


def parse_json(path: Path) -> Iterator[RawImportItem]:
    for event in _iter_json_events(Path(path)):
        if isinstance(event, ParseIssue):
            raise ValueError(
                f"{event.source_path} {event.record_locator}: "
                f"{event.error_code}: {event.error_message}"
            )
        yield event


class _TrackedCsvLines:
    def __init__(self, source_file: io.BufferedReader):
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


def _iter_csv_events(path: Path) -> Iterator[RawImportItem | ParseIssue]:
    source_path = Path(path)
    try:
        with source_path.open("rb") as source_file:
            tracked_lines = _TrackedCsvLines(source_file)
            reader = csv.DictReader(tracked_lines)
            if reader.fieldnames is None:
                yield ParseIssue(
                    source_path=source_path,
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
                        source_path=source_path,
                        record_locator=f"row:{row_number}",
                        error_code="CSV_EXTRA_COLUMNS",
                        error_message="CSV row contains more fields than its header",
                        raw_text=raw_text,
                    )
                    continue
                payload = {str(key): value for key, value in row.items() if key is not None}
                yield RawImportItem(
                    source_path=source_path,
                    record_locator=f"row:{row_number}",
                    raw_payload=payload,
                    raw_text=raw_text,
                )
    except (OSError, csv.Error, UnicodeError) as exc:
        yield ParseIssue(
            source_path=source_path,
            record_locator="row:1",
            error_code="MALFORMED_CSV",
            error_message=str(exc),
            raw_text=None,
        )


def parse_csv(path: Path) -> Iterator[RawImportItem]:
    for event in _iter_csv_events(Path(path)):
        if isinstance(event, ParseIssue):
            raise ValueError(
                f"{event.source_path} {event.record_locator}: "
                f"{event.error_code}: {event.error_message}"
            )
        yield event


def iter_source_events(path: Path) -> Iterator[RawImportItem | ParseIssue]:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl_events(source_path)
    elif suffix == ".json":
        yield from _iter_json_events(source_path)
    elif suffix == ".csv":
        yield from _iter_csv_events(source_path)
    else:
        yield ParseIssue(
            source_path=source_path,
            record_locator="document:1",
            error_code="UNSUPPORTED_SOURCE_FORMAT",
            error_message=f"unsupported source suffix: {source_path.suffix}",
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


__all__ = [
    "ParseIssue",
    "RawImportItem",
    "iter_raw_import_items",
    "iter_source_events",
    "parse_csv",
    "parse_json",
    "parse_jsonl",
]
