"""Caller and channel identities shared by the Discord ledger scripts.

From 2026-10-12 ACE posts as ``Zach_ace`` in a new ``ace-free-alerts`` channel. The original
account's last message is 2026-10-08 16:54 UTC and the new account was created 2026-10-09
00:18 UTC; the alias rests on that timing, ownership of the same alert/profit channels and
members addressing him as Ace. It is inferred, never authenticated, so rows keep the posting
``author_id`` and only the caller *label* is shared. Lifecycle linking stays per author_id.
"""
ACE_ID = "1079391263083733072"
ACE_SUCCESSOR_ID = "1557910141632782409"
FT_ID = "720901855995101225"

ACE_IDS = frozenset({ACE_ID, ACE_SUCCESSOR_ID})
CALLERS = {ACE_ID: "ACE", ACE_SUCCESSOR_ID: "ACE", FT_ID: "FT"}

FREE_ALERTS_CHANNEL_ID = "1416372120388243516"
SUCCESSOR_ALERTS_CHANNEL_ID = "1559060301955989554"
ALERT_CHANNEL_IDS = frozenset({FREE_ALERTS_CHANNEL_ID, SUCCESSOR_ALERTS_CHANNEL_ID})
