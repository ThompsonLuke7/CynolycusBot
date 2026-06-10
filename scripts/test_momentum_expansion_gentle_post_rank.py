from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_momentum_expansion_label_model_experiment import _selection_summary


DEFAULT_MATRIX = Path("strategies/momentum_expansion/data/processed/training_matrix_4h.parquet")
DEFAULT_PREDS = Path("strategies/momentum_expansion/data/processed/label_model_experiment/holdout_predictions.parquet")
DEFAULT_OUT = Path("strategies/momentum_expansion/data/processed/gentle_post_rank_experiment")


def _date_rank(df: pd.DataFrame, col: str, *, ascending: bool = True) -> pd.Series:
    return df.groupby(level="timestamp")[col].rank(pct=True, ascending=ascending)


def _variant_scores(df: pd.DataFrame, pred: pd.Series) -> dict[str, pd.Series]:
    work = df.copy()
    work["pred"] = pred
    work["atr_rank"] = _date_rank(work, "atr_pct_14")
    work["ret20_rank"] = _date_rank(work, "ret_20")
    work["vol_rank"] = _date_rank(work, "realized_vol_20")
    if "drawdown_from_60h" in work.columns:
        # Lower drawdown_from_60h means more extended near the 60-bar high.
        work["extension_rank"] = _date_rank(work, "drawdown_from_60h", ascending=False)
    else:
        work["extension_rank"] = work["ret20_rank"]

    return {
        "BASE_EXP_C": work["pred"],
        "GENTLE_025_ATR_025_RET20": work["pred"] - 0.025 * work["atr_rank"] - 0.025 * work["ret20_rank"],
        "GENTLE_050_ATR_050_RET20": work["pred"] - 0.050 * work["atr_rank"] - 0.050 * work["ret20_rank"],
        "GENTLE_025_ATR_025_EXTENSION": work["pred"] - 0.025 * work["atr_rank"] - 0.025 * work["extension_rank"],
        "GENTLE_050_ATR_050_EXTENSION": work["pred"] - 0.050 * work["atr_rank"] - 0.050 * work["extension_rank"],
        "GENTLE_050_VOL_050_EXTENSION": work["pred"] - 0.050 * work["vol_rank"] - 0.050 * work["extension_rank"],
    }


def _rank_presence_for_examples(df: pd.DataFrame, scores: dict[str, pd.Series]) -> pd.DataFrame:
    examples = ["DELL", "MU", "AAOI"]
    rows: list[dict[str, object]] = []
    tickers = df.index.get_level_values("ticker")
    for name, score in scores.items():
        work = df[["fwd_max_return", "fwd_max_drawdown", "fwd_close_return"]].copy()
        work["score"] = score
        work = work.dropna(subset=["score"]).copy()
        work["rank_in_bar"] = work.groupby(level="timestamp")["score"].rank(ascending=False, method="first")
        for ticker in examples:
            if ticker not in set(tickers):
                rows.append({"variant": name, "ticker": ticker, "rows": 0})
                continue
            g = work.loc[work.index.get_level_values("ticker") == ticker]
            if g.empty:
                rows.append({"variant": name, "ticker": ticker, "rows": 0})
                continue
            top5 = g["rank_in_bar"] <= 5
            top10 = g["rank_in_bar"] <= 10
            gt20 = g["fwd_max_return"] >= 0.20
            rows.append(
                {
                    "variant": name,
                    "ticker": ticker,
                    "rows": int(len(g)),
                    "max_fwd": float(g["fwd_max_return"].max()),
                    "top5_rows": int(top5.sum()),
                    "top10_rows": int(top10.sum()),
                    "gt20_rows": int(gt20.sum()),
                    "top5_gt20_rows": int((top5 & gt20).sum()),
                    "top10_gt20_rows": int((top10 & gt20).sum()),
                    "avg_fwd_when_top5": float(g.loc[top5, "fwd_max_return"].mean()) if top5.any() else np.nan,
                    "median_rank_when_gt20": float(g.loc[gt20, "rank_in_bar"].median()) if gt20.any() else np.nan,
                    "best_rank_when_gt20": float(g.loc[gt20, "rank_in_bar"].min()) if gt20.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run(matrix_path: Path, preds_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.read_parquet(preds_path)
    pred_col = "EXP_C_xsec_score_regression"
    if pred_col not in preds.columns:
        raise ValueError(f"Missing {pred_col} in {preds_path}")
    feature_cols = [
        "atr_pct_14",
        "realized_vol_20",
        "ret_20",
        "drawdown_from_60h",
        "fwd_max_return",
        "fwd_max_drawdown",
        "fwd_close_return",
    ]
    df = pd.read_parquet(matrix_path, columns=[c for c in feature_cols if c])
    df = df.loc[preds.index].copy()
    df[pred_col] = preds[pred_col]
    df = df.dropna(subset=[pred_col, "fwd_max_return", "fwd_max_drawdown", "fwd_close_return"])

    scores = _variant_scores(df, df[pred_col])
    selection = pd.concat(
        [pd.DataFrame(_selection_summary(df, score, name)) for name, score in scores.items()],
        ignore_index=True,
    )
    examples = _rank_presence_for_examples(df, scores)

    selection.to_csv(out_dir / "selection_quality.csv", index=False)
    examples.to_csv(out_dir / "example_ticker_rank_presence.csv", index=False)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "matrix": str(matrix_path),
                "predictions": str(preds_path),
                "base_prediction_column": pred_col,
                "rows": int(len(df)),
                "note": "Post-ranking penalties are applied only to holdout predictions; no model was retrained.",
            },
            indent=2,
        )
    )

    print("Selection quality")
    print(selection.to_string(index=False))
    print()
    print("Example ticker rank presence")
    print(examples.to_string(index=False))
    print()
    print(f"Wrote {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--preds", type=Path, default=DEFAULT_PREDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.matrix, args.preds, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
