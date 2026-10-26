import pytest

from scripts.discord_ledger.reconcile_export_refresh import annotation, preserve_prior_annotations
from scripts.discord_ledger.stage_export_refresh import merge_versions, semantic_key


def _message():
    return {"message_id": "123", "author_id": "ACE", "channel_id": "alerts", "text": "Mostly out",
            "timestamp_utc": "2026-10-07T16:09Z", "is_edited": False, "source_file": "original.html",
            "attachments": [{"filename": "image.jpg", "url": "https://cdn.discordapp.com/attachments/1/2/image.jpg?ex=old"}],
            "original_html_metadata": {"source_file_sha256": "abc"}}


def test_rotating_cdn_query_is_not_a_message_edit():
    original = _message()
    newer = {**original, "attachments": [{"filename": "image.jpg", "url": original["attachments"][0]["url"].replace("ex=old", "ex=new")}]}
    assert semantic_key(original) == semantic_key(newer)
    merged, changes, summary = merge_versions([original], [newer])
    assert len(merged) == 1 and len(merged[0]["source_observations"]) == 2
    assert changes == [] and summary["same_id_version_changes"] == 0


def test_removed_prior_source_remains_in_merged_view_and_changed_version_is_audited():
    original = _message()
    removed = {**original, "message_id": "removed"}
    changed = {**original, "text": "Actually all out"}
    merged, changes, summary = merge_versions([original, removed], [changed])
    assert {r["message_id"] for r in merged} == {"123", "removed"}
    assert summary["prior_ids_absent_from_current_preserved"] == 1
    assert len(changes[0]["variants"]) == 2


def test_prior_annotation_transfer_fails_if_source_changed():
    original = _message()
    event = {"event_id": "e", "message_id": "123", "field_sources": {"symbol": ["123"]}}
    with pytest.raises(ValueError, match="re-review"):
        preserve_prior_annotations([event], {"123": original}, {"123": {**original, "text": "Changed"}})


def test_transfer_strips_stale_lifecycle_assignment_but_retains_fact_sources():
    source = _message()
    event = {"event_id": "e", "message_id": "123", "trade_id": "old", "field_sources": {"symbol": ["123"]}}
    result = preserve_prior_annotations([event], {"123": source}, {"123": source})[0]
    assert "trade_id" not in result and result["field_sources"] == event["field_sources"]


def test_partial_exit_mark_is_not_a_sale_price_or_full_close():
    fact = {"symbol": "MU", "kind": "partial_exit_claim_with_mark", "option_type": "call", "strike": 1100, "option_mark": 8.95}
    row = annotation(fact, _message(), 0)
    assert row["event_type"] == "trim" and row["price"] is None
    assert row["option_mark"] == 8.95 and row["confirmed_sale_price"] is None
    assert row["quantity"] is None


def test_zone_hit_remains_underlying_plan_observation_not_filled_entry():
    fact = {"symbol": "ARM", "kind": "underlying_zone_tag_claim", "entry_zone": [298, 298], "targets": [319.41]}
    row = annotation(fact, _message(), 0)
    assert row["event_type"] == "watchlist" and row["speech_act"] == "observation"
    assert row["instrument_type"] == "underlying_plan" and row["price"] is None
