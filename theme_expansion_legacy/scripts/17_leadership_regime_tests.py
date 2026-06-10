from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    DAILY_BARS_PATH,
    LEADERS_PER_THEME,
    OUTPUT_DIR,
    THEME_LEADERS_PATH,
    THEME_SCORES_PATH,
    TRANSACTION_COST_BPS,
)

OUT_DIR = OUTPUT_DIR / "leadership_regime_tests"
ENTER_RANK = 3
EXIT_RANK = 12
MIN_HOLD_DAYS = 5
TOP_REGIME_THEMES = 5


def load_returns() -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    return bars.pivot(index="date", columns="ticker", values="px").sort_index().pct_change().fillna(0.0)


def tradable_scores(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    out["date"] = pd.to_datetime(out["date"])
    if "is_tradable" in out.columns:
        out = out[out["is_tradable"].fillna(False).astype(bool)]
    return out.dropna(subset=["theme_regime_rank"]).sort_values(["date", "theme_regime_rank"])


def pick_theme_leaders(leaders: pd.DataFrame, date: pd.Timestamp, theme: str) -> list[str]:
    day = leaders[(leaders["date"].eq(date)) & (leaders["theme"].eq(theme))]
    return day.sort_values("leader_rank").head(LEADERS_PER_THEME)["ticker"].dropna().tolist()


def perf_stats(returns: pd.Series, turnover: pd.Series) -> dict[str, float | None]:
    returns = returns.fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = max(len(returns) / 252.0, 1e-9)
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    loss_sum = abs(losers.sum())
    return {
        "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
        "sharpe": float(np.sqrt(252.0) * returns.mean() / returns.std()) if returns.std() else 0.0,
        "max_dd": float(drawdown.min()),
        "turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "profit_factor": float(winners.sum() / loss_sum) if loss_sum > 0 else None,
    }


def build_leadership_regime(scores: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(scores["date"].unique())
    prev_top: set[str] = set()
    streaks: dict[str, int] = {}
    rows = []

    for date in dates:
        day = scores[scores["date"].eq(date)].sort_values("theme_regime_rank")
        top = day.head(TOP_REGIME_THEMES).copy()
        top_set = set(top["theme"])

        for theme in list(streaks):
            if theme not in top_set:
                streaks[theme] = 0
        for theme in top_set:
            streaks[theme] = streaks.get(theme, 0) + 1

        durations = [streaks.get(theme, 0) for theme in top_set]
        overlap = len(top_set & prev_top)
        theme_turnover = 1.0 - overlap / max(len(top_set), 1)
        prev_top = top_set

        rows.append(
            {
                "date": date,
                "mean_theme_duration": float(np.mean(durations)) if durations else 0.0,
                "top_theme_duration": float(np.max(durations)) if durations else 0.0,
                "theme_rank_stability": float(top["rank_stability_20d"].mean())
                if "rank_stability_20d" in top.columns
                else np.nan,
                "theme_entropy": float(top["entropy_score"].mean()) if "entropy_score" in top.columns else np.nan,
                "theme_turnover": float(theme_turnover),
                "top_themes": "|".join(top["theme"].astype(str).tolist()),
            }
        )

    out = pd.DataFrame(rows).sort_values("date")
    out["theme_turnover_5d"] = out["theme_turnover"].rolling(5, min_periods=1).mean()
    out["mean_theme_duration_5d"] = out["mean_theme_duration"].rolling(5, min_periods=1).mean()
    stability_floor = out["theme_rank_stability"].quantile(0.40)
    entropy_floor = out["theme_entropy"].quantile(0.35)

    persistent = (
        (out["top_theme_duration"] >= 8)
        & (out["mean_theme_duration_5d"] >= 5)
        & (out["theme_turnover_5d"] <= 0.45)
        & (out["theme_rank_stability"].fillna(0.0) >= stability_floor)
        & (out["theme_entropy"].fillna(0.0) >= entropy_floor)
    )
    chaotic = (
        (out["top_theme_duration"] < 3)
        | (out["mean_theme_duration_5d"] < 3)
        | (out["theme_turnover_5d"] >= 0.70)
    )
    out["leadership_regime"] = np.select([persistent, chaotic], ["persistent", "chaotic"], default="rotating")

    # Apply close-known regime to the next trading day.
    out["decision_regime"] = out["leadership_regime"].shift(1).fillna("chaotic")
    return out


def exposure_for(regime: str, mode: str) -> float:
    if mode == "baseline":
        return 1.0
    if mode == "cash_when_chaotic":
        return {"persistent": 1.0, "rotating": 0.70, "chaotic": 0.0}.get(regime, 0.70)
    if mode == "reduced_when_chaotic":
        return {"persistent": 1.0, "rotating": 0.70, "chaotic": 0.30}.get(regime, 0.70)
    if mode == "persistent_only":
        return 1.0 if regime == "persistent" else 0.0
    return 1.0


def simulate(
    scores: pd.DataFrame,
    leaders: pd.DataFrame,
    returns: pd.DataFrame,
    regimes: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    dates = sorted(set(scores["date"]).intersection(returns.index))
    score_lookup = scores.set_index(["date", "theme"])
    regime_lookup = regimes.set_index("date")["decision_regime"]
    active: dict[str, dict] = {}
    prev_weights = pd.Series(dtype=float)
    rows = []

    for i, date in enumerate(dates):
        regime = regime_lookup.get(date, "chaotic")
        target_exposure = exposure_for(regime, mode)

        gross = 0.0
        if active and target_exposure > 0:
            holdings = []
            for state in active.values():
                for ticker in state["tickers"]:
                    holdings.append({"ticker": ticker, "weight": target_exposure * state["weight"] / len(state["tickers"])})
            weights = pd.DataFrame(holdings).groupby("ticker")["weight"].sum()
            gross = float((returns.loc[date].reindex(weights.index).fillna(0.0) * weights).sum())

        updated = dict(active)
        for theme, state in list(active.items()):
            held_days = i - state["start_i"] + 1
            row = score_lookup.loc[(date, theme)] if (date, theme) in score_lookup.index else pd.Series(dtype=float)
            rank = row.get("theme_regime_rank", np.nan)
            if held_days >= MIN_HOLD_DAYS and (pd.isna(rank) or rank > EXIT_RANK):
                updated.pop(theme, None)

        day = scores[scores["date"].eq(date)].sort_values("theme_regime_rank")
        entries = day[day["theme_regime_rank"] <= ENTER_RANK]
        for entry in entries.itertuples(index=False):
            if entry.theme in updated:
                continue
            tickers = [t for t in pick_theme_leaders(leaders, date, entry.theme) if t in returns.columns]
            if not tickers:
                continue
            updated[entry.theme] = {"tickers": tickers, "start_i": i + 1, "weight": 1.0}

        if updated and target_exposure > 0:
            theme_weight = 1.0 / len(updated)
            for state in updated.values():
                state["weight"] = theme_weight
            new_holdings = []
            for state in updated.values():
                for ticker in state["tickers"]:
                    new_holdings.append({"ticker": ticker, "weight": target_exposure * state["weight"] / len(state["tickers"])})
            new_weights = pd.DataFrame(new_holdings).groupby("ticker")["weight"].sum()
        else:
            new_weights = pd.Series(dtype=float)

        turnover = float(new_weights.sub(prev_weights, fill_value=0.0).abs().sum())
        prev_weights = new_weights
        active = updated
        rows.append(
            {
                "date": date,
                "mode": mode,
                "leadership_regime": regime,
                "strategy_return": gross - turnover * TRANSACTION_COST_BPS / 10_000.0,
                "turnover": turnover,
                "exposure": target_exposure,
                "active_themes": "|".join(sorted(active)),
            }
        )
    return pd.DataFrame(rows)


def period_2022_stats(bt: pd.DataFrame) -> dict[str, float | None]:
    sub = bt[(bt["date"] >= pd.Timestamp("2022-01-01")) & (bt["date"] <= pd.Timestamp("2022-12-31"))]
    if sub.empty:
        return {"cagr_2022": None, "sharpe_2022": None, "max_dd_2022": None}
    stats = perf_stats(sub["strategy_return"], sub["turnover"])
    return {"cagr_2022": stats["cagr"], "sharpe_2022": stats["sharpe"], "max_dd_2022": stats["max_dd"]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = tradable_scores(pd.read_parquet(THEME_SCORES_PATH))
    leaders = pd.read_parquet(THEME_LEADERS_PATH)
    leaders["date"] = pd.to_datetime(leaders["date"])
    returns = load_returns()

    regimes = build_leadership_regime(scores)
    regimes.to_csv(OUT_DIR / "leadership_regime_daily.csv", index=False)

    cases = [
        ("v3_baseline", "baseline"),
        ("leadership_scaled", "reduced_when_chaotic"),
        ("leadership_cash_chaos", "cash_when_chaotic"),
        ("leadership_persistent_only", "persistent_only"),
    ]
    rows = []
    curves = {}
    for case_name, mode in cases:
        bt = simulate(scores, leaders, returns, regimes, mode=mode)
        bt.to_csv(OUT_DIR / f"{case_name}_daily.csv", index=False)
        stats = perf_stats(bt["strategy_return"], bt["turnover"])
        stats.update(period_2022_stats(bt))
        rows.append({"case": case_name, **stats})
        curves[case_name] = (1.0 + bt.set_index("date")["strategy_return"]).cumprod()

    result = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    result.to_csv(OUT_DIR / "leadership_regime_test_results.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT_DIR / "leadership_regime_equity_curves.csv")
    (OUT_DIR / "leadership_regime_test_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(f"saved leadership regime tests -> {OUT_DIR}")


if __name__ == "__main__":
    main()
