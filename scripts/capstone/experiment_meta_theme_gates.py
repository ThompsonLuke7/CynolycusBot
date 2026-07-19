"""Validation-selected theme gates for Meta Ranker new-entry cohorts.

This is intentionally an entry-quality experiment, rather than an option/PnL
simulation.  It mirrors the live runner's candidate eligibility (combo rank,
liquidity, quality) and the "fresh cross-in" condition, then asks whether
slow theme context improves the 25x4H forward underlying-return label.

All theme values are joined strictly from the previous available theme date.
Thresholds are selected on the validation period and reported once on the
later frozen test period.  It writes compact CSV artifacts under
research/capstone/theme_gate_experiment/.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
META_OOF = REPO / "signals/meta_context/meta_ranker/models"
META_MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
THEME_SCORES = REPO / "theme_expansion/outputs/theme_scores.parquet"
THEME_ML_OOF = REPO / "theme_expansion/models/bundle/oof_preds.parquet"
THEME_MAP = REPO / "theme_expansion/data/theme_map_v4.csv"
OUT = REPO / "research/capstone/theme_gate_experiment"

VALIDATION_END = pd.Timestamp("2025-12-31", tz="UTC")
# A theme gate is allowed to concentrate the book, but it must still leave a
# meaningful entry cohort.  The combined gate is necessarily more selective.
MIN_COVERAGE = {"rule": 0.20, "ml": 0.20, "combined": 0.08}
RULE_RANKS = (3, 5, 8, 12)
ML_PCTS = (0.50, 0.60, 0.70, 0.80)


def _load_meta() -> pd.DataFrame:
    upside = pd.read_parquet(META_OOF / "upside/oof_preds.parquet").reset_index()
    quality = pd.read_parquet(META_OOF / "quality/oof_preds.parquet").reset_index()
    # The stored OOF files have a small overlapping-fold boundary (986 duplicate
    # keys, concentrated at one boundary bar).  Capstone's locked OOF builder
    # resolves this deterministically with the first fold prediction.
    upside = upside.drop_duplicates(subset=["timestamp", "ticker"], keep="first")
    quality = quality.drop_duplicates(subset=["timestamp", "ticker"], keep="first")
    quality = quality[["timestamp", "ticker", "score"]].rename(columns={"score": "s_quality"})
    upside = upside.rename(columns={"score": "s_upside"})
    out = upside.merge(quality, on=["timestamp", "ticker"], validate="one_to_one")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    matrix = pd.read_parquet(META_MATRIX, columns=["dollar_vol_pctile_252"]).reset_index()
    matrix["timestamp"] = pd.to_datetime(matrix["timestamp"], utc=True)
    out = out.merge(matrix, on=["timestamp", "ticker"], how="inner", validate="one_to_one")
    out = out.dropna(subset=["fwd_close_return", "fwd_max_return", "fwd_max_drawdown", "s_upside", "s_quality"])
    out["s_combo"] = (
        out.groupby("timestamp")["s_upside"].rank(pct=True)
        + out.groupby("timestamp")["s_quality"].rank(pct=True)
    ) / 2.0
    out["combo_rank"] = out.groupby("timestamp")["s_combo"].rank(ascending=False, method="first")
    return out.sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def _prior_theme_join(entries: pd.DataFrame) -> pd.DataFrame:
    mapping = pd.read_csv(THEME_MAP, usecols=["ticker", "theme_1"])
    mapping["ticker"] = mapping["ticker"].astype(str).str.upper()
    mapping = mapping.rename(columns={"theme_1": "legacy_theme"}).dropna().drop_duplicates("ticker")
    out = entries.merge(mapping, on="ticker", how="left", validate="many_to_one")
    out["signal_date"] = out["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()

    rule = pd.read_parquet(THEME_SCORES, columns=["date", "theme", "theme_regime_rank"])
    rule["date"] = pd.to_datetime(rule["date"]).dt.normalize()
    rule = rule.rename(columns={"theme": "legacy_theme", "date": "theme_date"})
    rule = rule.sort_values(["theme_date", "legacy_theme"])

    ml = pd.read_parquet(THEME_ML_OOF, columns=["date", "theme", "score"])
    ml["date"] = pd.to_datetime(ml["date"]).dt.normalize()
    # Rank is cross-sectional within the score's available theme day.  Its raw
    # scale is not stable enough for an absolute cross-period threshold.
    ml["theme_ml_pct"] = ml.groupby("date")["score"].rank(pct=True)
    ml = ml.rename(columns={"theme": "legacy_theme", "date": "ml_date"})
    ml = ml[["legacy_theme", "ml_date", "theme_ml_pct"]].sort_values(["ml_date", "legacy_theme"])

    # Strictly previous theme close: a 4H bar on T cannot consume the completed
    # daily theme aggregate for T.  merge_asof's allow_exact_matches=False
    # handles weekends/holidays without forward filling.
    left = out.sort_values(["signal_date", "legacy_theme"])
    left = pd.merge_asof(
        left, rule, left_on="signal_date", right_on="theme_date", by="legacy_theme",
        direction="backward", allow_exact_matches=False,
    )
    left = left.sort_values(["signal_date", "legacy_theme"])
    left = pd.merge_asof(
        left, ml, left_on="signal_date", right_on="ml_date", by="legacy_theme",
        direction="backward", allow_exact_matches=False,
    )
    return left.sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def _baseline_entries(df: pd.DataFrame) -> pd.DataFrame:
    # Same new-entry eligibility as the live Meta runner.  "cross-in" prevents
    # treating each bar of one continuing position as a separate new order.
    out = df.copy()
    out["prev_combo_rank"] = out.groupby("ticker")["combo_rank"].shift(1)
    eligible = (
        (out["s_combo"] >= 0.90)
        & (out["dollar_vol_pctile_252"].fillna(0.0) >= 0.60)
        & (out["s_quality"] >= 0.40)
        & (out["combo_rank"] <= 10)
    )
    cross_in = out["prev_combo_rank"].isna() | (out["prev_combo_rank"] > 10)
    return out[eligible & cross_in].copy()


def _summary(frame: pd.DataFrame, name: str, split: str, baseline_n: int) -> dict[str, float | int | str]:
    r = frame["fwd_close_return"]
    return {
        "split": split,
        "variant": name,
        "n_entries": len(frame),
        "coverage_vs_baseline": len(frame) / baseline_n if baseline_n else np.nan,
        "mean_fwd_close": r.mean(),
        "median_fwd_close": r.median(),
        "win_rate": (r > 0).mean(),
        "mean_fwd_max": frame["fwd_max_return"].mean(),
        "mean_max_drawdown": frame["fwd_max_drawdown"].mean(),
        "return_std": r.std(),
        "return_t_stat": r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 and r.std(ddof=1) else np.nan,
    }


def _select_variant(validation: pd.DataFrame, baseline_n: int, kind: str) -> tuple[str, pd.Series, pd.DataFrame]:
    candidates: list[tuple[str, pd.Series]] = []
    if kind == "rule":
        candidates = [(f"rule_rank<={rank}", validation["theme_regime_rank"] <= rank) for rank in RULE_RANKS]
    elif kind == "ml":
        candidates = [(f"ml_pct>={pct:.2f}", validation["theme_ml_pct"] >= pct) for pct in ML_PCTS]
    elif kind == "combined":
        candidates = [
            (f"rule_rank<={rank} + ml_pct>={pct:.2f}", (validation["theme_regime_rank"] <= rank) & (validation["theme_ml_pct"] >= pct))
            for rank, pct in product(RULE_RANKS, ML_PCTS)
        ]
    else:
        raise ValueError(kind)

    scored: list[tuple[float, str, pd.Series]] = []
    audit_rows: list[dict] = []
    for name, mask in candidates:
        picked = validation[mask.fillna(False)]
        coverage = len(picked) / baseline_n if baseline_n else 0.0
        if len(picked) < 20 or coverage < MIN_COVERAGE[kind]:
            audit_rows.append({"gate_family": kind, "variant": name, "n_entries": len(picked),
                               "coverage_vs_baseline": coverage, "eligible_for_selection": False,
                               "objective": np.nan})
            continue
        # Select for mean return, with a modest adverse-excursion tie-breaker.
        objective = float(picked["fwd_close_return"].mean() - 0.10 * picked["fwd_max_drawdown"].mean())
        scored.append((objective, name, mask))
        audit_rows.append({"gate_family": kind, "variant": name, "n_entries": len(picked),
                           "coverage_vs_baseline": coverage, "eligible_for_selection": True,
                           "objective": objective})
    if not scored:
        raise RuntimeError(f"No eligible {kind} candidate met coverage floor")
    _objective, name, mask = max(scored, key=lambda item: item[0])
    return name, mask, pd.DataFrame(audit_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scored = _prior_theme_join(_load_meta())
    entries = _baseline_entries(scored)
    entries = entries.dropna(subset=["theme_regime_rank", "theme_ml_pct"]).copy()
    validation = entries[entries["timestamp"] <= VALIDATION_END].copy()
    test = entries[entries["timestamp"] > VALIDATION_END].copy()
    if validation.empty or test.empty:
        raise RuntimeError("Validation/test split produced an empty cohort")

    chosen: dict[str, str] = {}
    rows: list[dict] = []
    selections: list[dict] = []
    selection_audits: list[pd.DataFrame] = []
    for split_name, split in (("validation", validation), ("frozen_test", test)):
        base_n = len(split)
        rows.append(_summary(split, "baseline", split_name, base_n))
        if split_name == "validation":
            for kind in ("rule", "ml", "combined"):
                name, mask, audit = _select_variant(split, base_n, kind)
                chosen[kind] = name
                selection_audits.append(audit)
                selections.append({"gate_family": kind, "selected_variant": name, "selection_split": split_name})
                rows.append(_summary(split[mask.fillna(False)], name, split_name, base_n))
        else:
            for kind, name in chosen.items():
                if kind == "rule":
                    rank = int(name.split("<=")[1])
                    mask = split["theme_regime_rank"] <= rank
                elif kind == "ml":
                    pct = float(name.split(">=")[1])
                    mask = split["theme_ml_pct"] >= pct
                else:
                    left, right = name.split(" + ")
                    rank = int(left.split("<=")[1])
                    pct = float(right.split(">=")[1])
                    mask = (split["theme_regime_rank"] <= rank) & (split["theme_ml_pct"] >= pct)
                rows.append(_summary(split[mask.fillna(False)], name, split_name, base_n))

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "selected_gates.csv", index=False)
    pd.concat(selection_audits, ignore_index=True).to_csv(OUT / "validation_gate_grid.csv", index=False)
    entries.to_parquet(OUT / "entry_cohort_with_prior_theme_context.parquet", index=False)
    print("=== Validation-selected theme-gate experiment ===")
    print(f"strict-prior theme coverage: {len(entries):,} baseline cross-in entries")
    print(pd.DataFrame(selections).to_string(index=False))
    show = result.copy()
    for col in ["coverage_vs_baseline", "mean_fwd_close", "median_fwd_close", "win_rate", "mean_fwd_max", "mean_max_drawdown", "return_std"]:
        show[col] = (show[col] * 100).round(2)
    show["return_t_stat"] = show["return_t_stat"].round(2)
    print(show.to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
