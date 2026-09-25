from __future__ import annotations

from scripts.discord_ledger.run_ace_setup_replay import _bucket, _tags


def test_source_tags_are_literal_and_preserve_unlabelled_rows() -> None:
    assert _tags("Optional lotto, 0 DTE") == ("lotto", "0dte")
    assert _bucket(_tags("Optional lotto, 0 DTE")) == "lotto"
    assert _tags("Call breakout") == ()
    assert _bucket(()) == "unlabelled"


def test_source_bucket_keeps_overlapping_explicit_setup_words() -> None:
    tags = _tags("Small swing lotto")
    assert tags == ("lotto", "swing")
    assert _bucket(tags) == "multi_tagged_lotto_swing"
