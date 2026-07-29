import numpy as np

from scripts.capstone.never_profitable_exit import _qualifies


def test_no_up_close_rejects_any_green_close_from_entry_path():
    assert _qualifies("no_up_close", 10.0, np.array([10.0, 9.9, 9.8]))
    assert not _qualifies("no_up_close", 10.0, np.array([10.0, 9.9, 9.95]))


def test_never_profitable_close_is_distinct_from_no_up_close():
    # A green bar below entry clears the no-up-close rule but has not produced
    # a profitable close relative to entry.
    closes = np.array([10.0, 9.5, 9.7])
    assert not _qualifies("no_up_close", 10.0, closes)
    assert _qualifies("never_profitable_close", 10.0, closes)
