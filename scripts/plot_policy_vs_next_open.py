from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current 1m execution policy vs naive next-1m-open execution from 10m meta signals."
    )
    parser.add_argument(
        "--trace",
        default="Data/inference/spy/10min/meta/meta_trace_policy_last_month.csv",
        help="Mixed replay trace CSV.",
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol to plot.")
    parser.add_argument("--start", default=None, help="Optional UTC start timestamp.")
    parser.add_argument("--end", default=None, help="Optional UTC end timestamp.")
    parser.add_argument("--tz", default="America/New_York", help="Display timezone.")
    parser.add_argument("--sessions-per-fig", type=int, default=4, help="Sessions per PNG.")
    parser.add_argument(
        "--out-dir",
        default="Data/inference/spy/10min/plots/policy_vs_next_open",
        help="Output directory for comparison plots.",
    )
    return parser.parse_args()


def _load_trace(path: Path, *, symbol: str, start: str | None, end: str | None, tz: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df[df["symbol"].astype(str).str.upper() == str(symbol).upper()].copy()
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    if "trace_kind" not in df.columns:
        df["trace_kind"] = "meta_10m"
    else:
        df["trace_kind"] = df["trace_kind"].fillna("meta_10m")
    df["ts_local"] = df["timestamp"].dt.tz_convert(tz)
    df["session_date"] = df["ts_local"].dt.normalize()
    meta = df[df["trace_kind"].astype(str) == "meta_10m"].copy().sort_values("timestamp")
    policy = df[df["trace_kind"].astype(str) == "policy_1m"].copy().sort_values("timestamp")
    if meta.empty or policy.empty:
        raise ValueError("Trace must contain both meta_10m and policy_1m rows.")
    return meta, policy


def _derive_policy_markers(policy_df: pd.DataFrame) -> pd.DataFrame:
    out = policy_df.sort_values("timestamp").copy()
    long_pos = pd.to_numeric(out.get("policy_long_contracts"), errors="coerce").fillna(0).astype(int)
    short_pos = pd.to_numeric(out.get("policy_short_contracts"), errors="coerce").fillna(0).astype(int)
    prev_long = long_pos.shift(1).fillna(0).astype(int)
    prev_short = short_pos.shift(1).fillna(0).astype(int)
    out["entry_long"] = (long_pos > 0) & (prev_long <= 0)
    out["exit_long"] = (prev_long > 0) & (long_pos <= 0)
    out["entry_short"] = (short_pos > 0) & (prev_short <= 0)
    out["exit_short"] = (prev_short > 0) & (short_pos <= 0)
    return out


def _target_side(current_open: bool, p_enter: float, p_exit: float, thr_enter: float, thr_exit: float) -> int:
    if current_open:
        if pd.notna(p_exit) and pd.notna(thr_exit) and float(p_exit) >= float(thr_exit):
            return 0
        return 1
    if pd.notna(p_enter) and pd.notna(thr_enter) and float(p_enter) >= float(thr_enter):
        return 1
    return 0


def _build_next_open_execution(meta_df: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    one_min = policy_df[["timestamp", "ts_local", "session_date", "open", "high", "low", "close", "volume"]].copy()
    one_min = one_min.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    current_long = 0
    current_short = 0
    rows: list[dict] = []

    for row in meta_df.itertuples(index=False):
        next_rows = one_min[one_min["timestamp"] > row.timestamp]
        if next_rows.empty:
            continue
        nxt = next_rows.iloc[0]
        target_long = _target_side(
            current_open=bool(current_long),
            p_enter=getattr(row, "p_enter_long", float("nan")),
            p_exit=getattr(row, "p_exit_long", float("nan")),
            thr_enter=getattr(row, "thr_enter_long", float("nan")),
            thr_exit=getattr(row, "thr_exit_long", float("nan")),
        )
        target_short = _target_side(
            current_open=bool(current_short),
            p_enter=getattr(row, "p_enter_short", float("nan")),
            p_exit=getattr(row, "p_exit_short", float("nan")),
            thr_enter=getattr(row, "thr_enter_short", float("nan")),
            thr_exit=getattr(row, "thr_exit_short", float("nan")),
        )

        event_parts: list[str] = []
        if current_long == 0 and target_long > 0:
            event_parts.append("open_long")
        elif current_long > 0 and target_long == 0:
            event_parts.append("close_long")
        if current_short == 0 and target_short > 0:
            event_parts.append("open_short")
        elif current_short > 0 and target_short == 0:
            event_parts.append("close_short")

        current_long = target_long
        current_short = target_short
        rows.append(
            {
                "timestamp": nxt["timestamp"],
                "ts_local": nxt["ts_local"],
                "session_date": nxt["session_date"],
                "open": nxt["open"],
                "high": nxt["high"],
                "low": nxt["low"],
                "close": nxt["close"],
                "volume": nxt["volume"],
                "naive_event": "+".join(event_parts) if event_parts else "hold",
                "naive_long_contracts": current_long,
                "naive_short_contracts": current_short,
                "signal_timestamp": row.timestamp,
            }
        )

    naive = pd.DataFrame(rows)
    if naive.empty:
        raise ValueError("No next-open execution rows were generated.")
    naive = naive.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    long_pos = pd.to_numeric(naive["naive_long_contracts"], errors="coerce").fillna(0).astype(int)
    short_pos = pd.to_numeric(naive["naive_short_contracts"], errors="coerce").fillna(0).astype(int)
    prev_long = long_pos.shift(1).fillna(0).astype(int)
    prev_short = short_pos.shift(1).fillna(0).astype(int)
    naive["entry_long"] = (long_pos > 0) & (prev_long <= 0)
    naive["exit_long"] = (prev_long > 0) & (long_pos <= 0)
    naive["entry_short"] = (short_pos > 0) & (prev_short <= 0)
    naive["exit_short"] = (prev_short > 0) & (short_pos <= 0)
    return naive


def _plot(meta: pd.DataFrame, policy: pd.DataFrame, naive: pd.DataFrame, *, out_dir: Path, sessions_per_fig: int, symbol: str, tz: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = _derive_policy_markers(policy)
    session_dates = sorted(policy["session_date"].dropna().unique().tolist())
    candle_width = 0.8 / (24 * 60)
    outputs: list[Path] = []

    for chunk_idx in range(int(ceil(len(session_dates) / max(1, sessions_per_fig)))):
        dates = session_dates[chunk_idx * sessions_per_fig:(chunk_idx + 1) * sessions_per_fig]
        n = len(dates)
        fig, axes = plt.subplots(
            n * 2,
            1,
            figsize=(18, max(7.0, 5.0 * n)),
            sharex=False,
            gridspec_kw={"height_ratios": [2.3, 1.0] * n},
        )
        if n == 1:
            axes = [axes[0], axes[1]]

        for idx, session_date in enumerate(dates):
            ax_price = axes[idx * 2]
            ax_prob = axes[idx * 2 + 1]
            pol = policy[policy["session_date"] == session_date].copy().sort_values("ts_local")
            met = meta[meta["session_date"] == session_date].copy().sort_values("ts_local")
            nai = naive[naive["session_date"] == session_date].copy().sort_values("ts_local")
            if pol.empty:
                continue

            x = mdates.date2num(pol["ts_local"].to_list())
            open_ = pd.to_numeric(pol["open"], errors="coerce").to_numpy()
            high = pd.to_numeric(pol["high"], errors="coerce").to_numpy()
            low = pd.to_numeric(pol["low"], errors="coerce").to_numpy()
            close = pd.to_numeric(pol["close"], errors="coerce").to_numpy()
            up = close >= open_
            down = ~up

            ax_price.vlines(x, low, high, color="#4a4a4a", linewidth=0.7, zorder=1)
            ax_price.bar(x[up], close[up] - open_[up], width=candle_width, bottom=open_[up], color="#1976D2", edgecolor="none", zorder=1.2, label="bull")
            ax_price.bar(x[down], close[down] - open_[down], width=candle_width, bottom=open_[down], color="#E53935", edgecolor="none", zorder=1.2, label="bear")

            spread = high - low
            offset = pd.Series(spread).replace(0, pd.NA).dropna().median()
            if pd.isna(offset) or offset <= 0:
                offset = max(close) * 0.0015

            def _draw_markers(frame: pd.DataFrame, *, alpha: float, prefix: str, filled: bool) -> None:
                if frame.empty:
                    return
                xf = mdates.date2num(frame["ts_local"].to_list())
                y_enter_long = pd.to_numeric(frame["low"], errors="coerce").to_numpy() - float(offset) * 1.2
                y_exit_long = pd.to_numeric(frame["high"], errors="coerce").to_numpy() + float(offset) * 0.8
                y_enter_short = pd.to_numeric(frame["high"], errors="coerce").to_numpy() + float(offset) * 1.2
                y_exit_short = pd.to_numeric(frame["low"], errors="coerce").to_numpy() - float(offset) * 0.8
                edge = None if filled else "#111111"
                face = None if filled else "none"
                for col, yvals, color, marker, label in (
                    ("entry_long", y_enter_long, "#2E7D32", "^", f"{prefix} enter long"),
                    ("exit_long", y_exit_long, "#8c564b", "v", f"{prefix} exit long"),
                    ("entry_short", y_enter_short, "#C62828", "v", f"{prefix} enter short"),
                    ("exit_short", y_exit_short, "#9467bd", "^", f"{prefix} exit short"),
                ):
                    mask = frame[col].fillna(False).to_numpy()
                    if mask.any():
                        ax_price.scatter(
                            xf[mask],
                            yvals[mask],
                            color=color if filled else face,
                            edgecolors=color if not filled else edge,
                            marker=marker,
                            s=36,
                            linewidths=1.3,
                            alpha=alpha,
                            label=label,
                            zorder=2.1 if filled else 2.0,
                        )

            _draw_markers(pol, alpha=0.95, prefix="policy", filled=True)
            _draw_markers(nai, alpha=0.95, prefix="next-open", filled=False)

            ax_price.set_title(f"{symbol} | {session_date.date()} | policy vs next-open")
            ax_price.set_ylabel("Price")
            ax_price.grid(True, alpha=0.25)
            handles, labels = ax_price.get_legend_handles_labels()
            dedup = dict(zip(labels, handles))
            ax_price.legend(dedup.values(), dedup.keys(), loc="upper left", fontsize=7, ncol=3)

            if not met.empty:
                mx = mdates.date2num(met["ts_local"].to_list())
                for col, color, label in (
                    ("p_enter_long", "#2ca02c", "p_enter_long"),
                    ("p_enter_short", "#d62728", "p_enter_short"),
                    ("p_exit_long", "#17becf", "p_exit_long"),
                    ("p_exit_short", "#ff7f0e", "p_exit_short"),
                ):
                    if col in met.columns:
                        y = pd.to_numeric(met[col], errors="coerce")
                        if y.notna().any():
                            ax_prob.step(mx, y.to_numpy(), where="post", color=color, linewidth=1.2, label=label)
                for col, color, label in (
                    ("thr_enter_long", "#2ca02c", "thr_enter_long"),
                    ("thr_enter_short", "#d62728", "thr_enter_short"),
                    ("thr_exit_long", "#17becf", "thr_exit_long"),
                    ("thr_exit_short", "#ff7f0e", "thr_exit_short"),
                ):
                    if col in met.columns:
                        y = pd.to_numeric(met[col], errors="coerce").dropna()
                        if not y.empty:
                            ax_prob.axhline(float(y.iloc[-1]), color=color, linestyle="--", linewidth=1.0, alpha=0.85, label=label)
            ax_prob.set_ylim(-0.02, 1.02)
            ax_prob.set_ylabel("Probability")
            ax_prob.set_xlabel(f"Session Time ({tz})")
            ax_prob.grid(True, alpha=0.25)
            handles, labels = ax_prob.get_legend_handles_labels()
            dedup = dict(zip(labels, handles))
            ax_prob.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=8, ncol=4)

            session_tz = pol["ts_local"].dt.tz
            locator = mdates.HourLocator(interval=1, tz=session_tz)
            formatter = mdates.DateFormatter("%H:%M", tz=session_tz)
            ax_price.xaxis.set_major_locator(locator)
            ax_price.xaxis.set_major_formatter(formatter)
            ax_prob.xaxis.set_major_locator(locator)
            ax_prob.xaxis.set_major_formatter(formatter)
            ax_price.tick_params(axis="x", labelbottom=False)
            ax_price.set_xlim(x.min() - candle_width * 2, x.max() + candle_width * 2)
            ax_prob.set_xlim(x.min() - candle_width * 2, x.max() + candle_width * 2)

        fig.tight_layout()
        out_path = out_dir / f"{symbol.lower()}_policy_vs_next_open_part{chunk_idx + 1}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def main() -> None:
    args = _parse_args()
    meta, policy = _load_trace(
        Path(args.trace),
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        tz=args.tz,
    )
    naive = _build_next_open_execution(meta, policy)
    outputs = _plot(
        meta,
        policy,
        naive,
        out_dir=Path(args.out_dir),
        sessions_per_fig=max(1, int(args.sessions_per_fig)),
        symbol=args.symbol,
        tz=args.tz,
    )
    print("Generated comparison plots:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
