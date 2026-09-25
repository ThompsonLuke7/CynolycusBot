"""Stratify the frozen ACE replay and context studies without using outcomes.

SPY is predeclared separately because it constitutes 54/120 source candidates
and is structurally different from ACE's individual-stock alerts.  ETF versus
single-name is a secondary descriptive split.  These are compatibility/context
rates only, not a profitability screen or a full-universe discovery test.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any, Callable

from scripts.discord_ledger.run_ace_context_selectivity import summarize


ETF_SYMBOLS = {"SPY", "QQQ", "IWM", "GLD", "XLE", "XLF", "SLV"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _modeled_entry(value: Any) -> bool:
    return int(value or 0) > 0


def _paired_exact_p(source_only: int, control_only: int) -> float:
    """Two-sided exact McNemar/binomial p value for discordant paired rows."""

    discordant = source_only + control_only
    if not discordant:
        return 1.0
    low = min(source_only, control_only)
    lower_tail = sum(comb(discordant, index) for index in range(low + 1)) / 2 ** discordant
    return min(1.0, 2 * lower_tail)


def engine_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, tuple[str, Callable[[dict[str, Any]], bool]]] = {
        "setup_detected": ("structure_setup_detected", bool),
        "explicit_confirmation": ("structure_explicit_confirmation", bool),
        "modeled_entry": ("structure_modeled_trade_count", _modeled_entry),
    }
    result: dict[str, Any] = {"n": len(pairs)}
    for name, (column, converter) in metrics.items():
        source = sum(converter(pair["source"].get(column)) for pair in pairs) / len(pairs) if pairs else None
        control = sum(converter(pair["control"].get(column)) for pair in pairs) / len(pairs) if pairs else None
        source_only = sum(converter(pair["source"].get(column)) and not converter(pair["control"].get(column))
                          for pair in pairs)
        control_only = sum(not converter(pair["source"].get(column)) and converter(pair["control"].get(column))
                           for pair in pairs)
        result[name] = {
            "source_rate": source, "control_rate": control,
            "source_minus_control_rate": None if source is None or control is None else source - control,
            "source_only_pairs": source_only, "control_only_pairs": control_only,
            "discordant_pairs": source_only + control_only,
            "paired_exact_p": _paired_exact_p(source_only, control_only),
        }
    return result


def stratum_name(symbol: str) -> list[str]:
    clean = str(symbol).upper()
    return ["all", "spy" if clean == "SPY" else "non_spy", "etf" if clean in ETF_SYMBOLS else "single_name"]


def analyze(pairs: list[dict[str, Any]], context_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("all", "spy", "non_spy", "etf", "single_name"):
        pair_group = [pair for pair in pairs if name in stratum_name(pair["source"]["symbol"])]
        context_group = [row for row in context_rows if name in stratum_name(row["symbol"])]
        result[name] = {"engine": engine_summary(pair_group), "context": summarize(context_group)}
    return result


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def write_report(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# ACE structure/context stratum analysis", "",
        "SPY is separated prospectively because it is 45% of the source sample. ETF versus single-name is descriptive. "
        "No returns, option prices, or source-reported outcome claims are included.", "",
        "## Intraday Structure compatibility", "",
        "| Stratum | n | Source confirmation | Control confirmation | Difference | Confirmation discordant (source/control) | Exact paired p | Source modeled entry | Control modeled entry | Difference |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, value in results.items():
        engine = value["engine"]
        confirm, entry = engine["explicit_confirmation"], engine["modeled_entry"]
        lines.append(
            f"| {name} | {engine['n']} | {_pct(confirm['source_rate'])} | {_pct(confirm['control_rate'])} | "
            f"{_pct(confirm['source_minus_control_rate'])} | {confirm['source_only_pairs']}/{confirm['control_only_pairs']} | "
            f"{confirm['paired_exact_p']:.3f} | {_pct(entry['source_rate'])} | "
            f"{_pct(entry['control_rate'])} | {_pct(entry['source_minus_control_rate'])} |"
        )
    lines.extend(["", "## Pre-candidate context", "",
                  "| Stratum | Alert n | Watchlist 7d difference | News-article presence difference |",
                  "| --- | ---: | ---: | ---: |"])
    for name, value in results.items():
        context = value["context"]
        rates = context["alert_minus_control"]["rates"]
        lines.append(f"| {name} | {context['alert']['rows']} | {_pct(rates['watchlist_7d'])} | {_pct(rates['news_24h_has_article'])} |")
    lines.extend(["", "## Limits", "",
                  "This retains injected source symbol/direction and therefore cannot measure independent discovery. "
                  "The underlying source/control sample is small in some strata; differences are descriptive, not a model-selection result."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite research output: {args.out}")
    results = analyze(read_jsonl(args.paired), read_jsonl(args.context))
    args.out.mkdir(parents=True)
    (args.out / "results.json").write_text(json.dumps({
        "schema": "ace_stratum_analysis_v1", "research_only": True,
        "strata": "SPY/non-SPY primary; ETF/single-name secondary", "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    write_report(args.out / "report.md", results)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
