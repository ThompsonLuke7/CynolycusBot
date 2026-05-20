"""Momentum scalper research package.

The package is intentionally event-driven and parquet-first: historical
downloaders write cache files, scanner/feature/label builders consume those
caches, and the replay engine wires the same pieces together minute by minute.
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
