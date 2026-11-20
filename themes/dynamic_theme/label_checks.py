"""Publish gate for the week's primary themes (2026-11-12).

Theme names are minted by the labeler every week, so the hand-labeled cases in
``label_regression.csv`` are written as patterns over names, not as exact names:

  must_match       if the ticker is labeled, its theme name must match this regex
                   (scored as a pass rate; an unclassified ticker is skipped when
                   unclassified_ok, else it counts as a miss)
  must_not_match   the theme name must never match this regex (any hit fails the gate)

Also checked: no ticker sits in another country's basket theme, theme_map_v4
agreement (the 2026-10-04 metric: for each placed ticker, the share of its theme's
clustered members that share a hand-curated v4 theme with it), the unclassified
share, and the published theme count against the previous run.

A check whose inputs are too thin to judge is reported as "not measured", never
as a pass or a failure. Pure functions except ``load_*``.
"""
from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from themes.dynamic_theme.config import (
    LABEL_GATE_MAX_UNCLASSIFIED,
    LABEL_GATE_MIN_MATCH_RATE,
    LABEL_GATE_MIN_THEME_RATIO,
    LABEL_GATE_MIN_V4_AGREEMENT,
    LABEL_REGRESSION_PATH,
    THEME_MAP_V4_CANDIDATES,
    UNCLASSIFIED_THEME,
)

MIN_V4_SCORED = 100        # placed tickers with a comparable v4 neighbour, else not measured
MIN_TICKERS_FOR_SHARE = 100
MIN_PRIOR_THEMES = 20


class LabelGateError(RuntimeError):
    """The week's primary themes failed the publish gate; features were not written."""


def load_regression_cases(path: Path = LABEL_REGRESSION_PATH) -> list[dict]:
    with Path(path).open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("ticker") or "").strip()]
    for r in rows:
        r["ticker"] = r["ticker"].strip().upper()
        r["unclassified_ok"] = str(r.get("unclassified_ok", "")).strip().lower() in {"true", "1", "yes"}
        for key in ("must_match", "must_not_match"):
            r[key] = (r.get(key) or "").strip()
            if r[key]:
                re.compile(r[key])   # fail fast on a bad pattern
    return rows


def load_v4_map(candidates=THEME_MAP_V4_CANDIDATES) -> dict[str, set[str]] | None:
    for path in candidates:
        if Path(path).exists():
            df = pd.read_csv(path)
            cols = [c for c in ("theme_1", "theme_2", "theme_3") if c in df.columns]
            return {str(r["ticker"]).upper(): {str(r[c]) for c in cols if pd.notna(r[c]) and str(r[c])}
                    for r in df.to_dict("records")}
    return None


def regression_check(primary: pd.DataFrame, cases: list[dict]) -> dict:
    theme = dict(zip(primary["ticker"].astype(str), primary["primary_theme"].astype(str)))
    forbidden, misses, passes, skipped = [], [], 0, 0
    for c in cases:
        th = theme.get(c["ticker"])
        if th is None:
            continue
        if c["must_not_match"] and re.search(c["must_not_match"], th):
            forbidden.append(f"{c['ticker']} -> {th} (must not match /{c['must_not_match']}/)")
        if not c["must_match"]:
            continue
        if th == UNCLASSIFIED_THEME:
            if c["unclassified_ok"]:
                skipped += 1
            else:
                misses.append(f"{c['ticker']} -> unclassified (expected /{c['must_match']}/)")
        elif re.search(c["must_match"], th):
            passes += 1
        else:
            misses.append(f"{c['ticker']} -> {th} (expected /{c['must_match']}/)")
    scored = passes + len(misses)
    return {"forbidden": forbidden, "misses": misses, "passes": passes, "scored": scored,
            "skipped_unclassified": skipped, "match_rate": passes / scored if scored else None}


def v4_agreement(primary: pd.DataFrame, v4: Mapping[str, set[str]] | None) -> dict:
    if not v4:
        return {"agreement": None, "n": 0, "note": "theme_map_v4 not found"}
    clustered = primary[primary["source"] == "cluster"] if "source" in primary.columns else primary
    members = clustered.groupby("primary_theme")["ticker"].apply(list).to_dict()
    scores = []
    for t, th in zip(primary["ticker"].astype(str), primary["primary_theme"].astype(str)):
        if th == UNCLASSIFIED_THEME or t not in v4:
            continue
        others = [m for m in members.get(th, []) if m != t and m in v4]
        if others:
            scores.append(sum(bool(v4[t] & v4[m]) for m in others) / len(others))
    return {"agreement": sum(scores) / len(scores) if scores else None, "n": len(scores)}


def basket_violations(primary: pd.DataFrame, countries: Mapping[str, str]) -> list[str]:
    """Tickers in a place-named theme for other countries (hand pins exempt)."""
    from themes.dynamic_theme.stages.step08_memberships import closed_to

    country = pd.Series(dict(countries or {}), dtype=object)
    judged = primary[primary["source"] != "pinned"] if "source" in primary.columns else primary
    bad = judged[closed_to(judged["primary_theme"], judged["ticker"].map(country))]
    return [f"{t} ({country.get(t)}) -> {th}" for t, th in zip(bad["ticker"], bad["primary_theme"])]


def previous_theme_count(features_path: Path) -> int | None:
    if not Path(features_path).exists():
        return None
    f = pd.read_parquet(features_path, columns=["date", "primary_theme"])
    if f.empty:
        return None
    last = f[f["date"] == f["date"].max()]
    return int(last.loc[last["primary_theme"] != UNCLASSIFIED_THEME, "primary_theme"].nunique())


def run_label_checks(primary: pd.DataFrame, *, countries: Mapping[str, str] | None,
                     cases: list[dict], v4: Mapping[str, set[str]] | None,
                     previous_themes: int | None, as_of=None) -> dict:
    failures: list[str] = []
    reg = regression_check(primary, cases)
    if reg["forbidden"]:
        failures.append(f"{len(reg['forbidden'])} forbidden label(s): " + "; ".join(reg["forbidden"][:10]))
    if reg["match_rate"] is not None and reg["match_rate"] < LABEL_GATE_MIN_MATCH_RATE:
        failures.append(f"regression match rate {reg['match_rate']:.2f} < {LABEL_GATE_MIN_MATCH_RATE:.2f} "
                        f"({reg['passes']}/{reg['scored']})")
    violations = basket_violations(primary, countries or {})
    if violations:
        failures.append(f"{len(violations)} ticker(s) in another country's basket: " + "; ".join(violations[:10]))
    agree = v4_agreement(primary, v4)
    agreement = agree["agreement"] if agree["n"] >= MIN_V4_SCORED else None
    if agreement is not None and agreement < LABEL_GATE_MIN_V4_AGREEMENT:
        failures.append(f"theme_map_v4 agreement {agreement:.3f} < {LABEL_GATE_MIN_V4_AGREEMENT:.2f}")
    n = len(primary)
    unclassified = float((primary["primary_theme"] == UNCLASSIFIED_THEME).mean()) if n else None
    if n >= MIN_TICKERS_FOR_SHARE and unclassified is not None and unclassified > LABEL_GATE_MAX_UNCLASSIFIED:
        failures.append(f"unclassified share {unclassified:.1%} > {LABEL_GATE_MAX_UNCLASSIFIED:.0%}")
    themes = int(primary.loc[primary["primary_theme"] != UNCLASSIFIED_THEME, "primary_theme"].nunique())
    if previous_themes and previous_themes >= MIN_PRIOR_THEMES and themes < LABEL_GATE_MIN_THEME_RATIO * previous_themes:
        failures.append(f"{themes} themes published vs {previous_themes} last run "
                        f"(< {LABEL_GATE_MIN_THEME_RATIO:.0%})")
    return {
        "ok": not failures, "failures": failures,
        "checked_at": datetime.now(timezone.utc).isoformat(), "as_of": str(as_of) if as_of is not None else None,
        "metrics": {"tickers": n, "themes": themes, "previous_themes": previous_themes,
                    "unclassified_share": unclassified, "v4_agreement": agreement, "v4_scored": agree["n"],
                    "regression_match_rate": reg["match_rate"],
                    "sources": primary["source"].value_counts().to_dict() if "source" in primary.columns else {}},
        "regression": reg, "basket_violations": violations,
    }
