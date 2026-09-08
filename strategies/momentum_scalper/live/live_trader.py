"""Paper-safe momentum scalper orchestration.

This module intentionally exposes shadow intents only.  A caller must build a
complete shared decision record and invoke the paper gateway explicitly; this
strategy has no live broker-write path.
"""
from __future__ import annotations

from strategies.momentum_scalper.live.shadow import MomentumShadowRunner, ShadowResult


__all__ = ["MomentumShadowRunner", "ShadowResult"]
