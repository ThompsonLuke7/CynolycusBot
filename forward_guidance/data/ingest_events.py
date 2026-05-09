"""Earnings event ingestion from CSV plus free SEC/web sources."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from forward_guidance.config import EVENTS_PATH, RAW_DIR, ensure_data_dirs
from forward_guidance.data.schema import EarningsEvent, event_from_record, events_to_frame, raw_event_dir
from forward_guidance.data.sec_client import SecClient, extract_basic_xbrl_metrics
from forward_guidance.features.nlp import extract_forward_sections
from forward_guidance.utils.io import read_json, write_dataframe, write_json

logger = logging.getLogger(__name__)


RAW_TEXT_FILES = {
    "press_release": "press_release.txt",
    "transcript": "transcript.txt",
    "guidance_section": "guidance_section.txt",
    "qa_section": "qa_section.txt",
}


def load_events(path: Path | str = EVENTS_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    if str(p).lower().endswith(".csv"):
        return pd.read_csv(p)
    return pd.read_parquet(p)


def write_events(events: Iterable[EarningsEvent], path: Path | str = EVENTS_PATH) -> pd.DataFrame:
    ensure_data_dirs()
    df = events_to_frame(list(events))
    write_dataframe(df, path)
    return df


def load_events_from_csv(path: Path | str) -> list[EarningsEvent]:
    df = pd.read_csv(path)
    return [event_from_record(row) for _, row in df.iterrows()]


def event_text_path(event: EarningsEvent, kind: str) -> Path:
    name = RAW_TEXT_FILES.get(kind, f"{kind}.txt")
    return raw_event_dir(RAW_DIR, event) / name


def read_event_text(event: EarningsEvent, kind: str) -> str:
    path = event_text_path(event, kind)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def write_event_text(event: EarningsEvent, kind: str, text: str) -> Path:
    path = event_text_path(event, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")
    return path


def ingest_event(
    event: EarningsEvent,
    *,
    sec_client: SecClient | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Fetch/cache free SEC data for one event and write raw text + manifest files."""
    ensure_data_dirs()
    sec = sec_client or SecClient(cache_dir=raw_event_dir(RAW_DIR, event) / "_sec_cache")
    out_dir = raw_event_dir(RAW_DIR, event)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "event": event.to_record(),
        "availability": {
            "press_release": False,
            "transcript": False,
            "guidance_section": False,
            "qa_section": False,
            "metrics": False,
            "sec_filings": False,
        },
        "sources": [],
    }

    cik = event.cik or sec.ticker_to_cik(event.clean_ticker)
    if cik:
        manifest["event"]["cik"] = cik

    press_path = event_text_path(event, "press_release")
    if cik and (force or not press_path.exists()):
        filings = sec.find_nearby_filings(cik=cik, earnings_date=event.earnings_date)
        if filings:
            filing = filings[0]
            text = sec.download_filing_text(filing, cache_path=out_dir / "sec_primary_document.html", force=force)
            write_event_text(event, "press_release", text)
            manifest["availability"]["press_release"] = bool(text)
            manifest["availability"]["sec_filings"] = True
            manifest["sources"].append(
                {
                    "type": "sec_filing",
                    "form": filing.form,
                    "filing_date": filing.filing_date,
                    "url": filing.document_url,
                    "accession_number": filing.accession_number,
                }
            )
    elif press_path.exists():
        manifest["availability"]["press_release"] = True

    source_text = read_event_text(event, "press_release") or read_event_text(event, "transcript")
    if source_text and (force or not event_text_path(event, "guidance_section").exists()):
        sections = extract_forward_sections(source_text)
        write_event_text(event, "guidance_section", sections.get("forward_guidance", ""))
        write_event_text(event, "qa_section", sections.get("qa", ""))

    for key in ("press_release", "transcript", "guidance_section", "qa_section"):
        manifest["availability"][key] = bool(read_event_text(event, key).strip())

    metrics_path = out_dir / "metrics.json"
    if cik and (force or not metrics_path.exists()):
        try:
            facts = sec.companyfacts(cik)
            metrics = extract_basic_xbrl_metrics(facts, event.fiscal_period)
            write_json(metrics, metrics_path)
        except Exception as exc:
            logger.warning("[%s] SEC companyfacts failed: %s", event.event_id, exc)
    metrics = read_json(metrics_path, default={}) or {}
    manifest["availability"]["metrics"] = bool(metrics)

    write_json(manifest, out_dir / "source_manifest.json")
    return manifest


def ingest_events_from_csv(
    csv_path: Path | str,
    *,
    force: bool = False,
    write_manifest: bool = True,
) -> pd.DataFrame:
    events = load_events_from_csv(csv_path)
    write_events(events)
    if write_manifest:
        sec = SecClient(cache_dir=RAW_DIR / "_sec_cache")
        for event in events:
            try:
                ingest_event(event, sec_client=sec, force=force)
            except Exception as exc:
                logger.warning("[%s] ingestion failed: %s", event.event_id, exc)
    return events_to_frame(events)
