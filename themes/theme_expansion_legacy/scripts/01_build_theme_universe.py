from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import THEME_MAP_PATH, ensure_dirs, load_theme_definitions, load_theme_memberships


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate default theme map and theme definitions.")
    parser.add_argument("--input", type=Path, help="Optional source CSV to validate and copy.")
    args = parser.parse_args()

    ensure_dirs()
    path = args.input or THEME_MAP_PATH
    memberships = load_theme_memberships(path)
    defs = load_theme_definitions()
    themes = memberships.groupby("theme")["ticker"].nunique().sort_values(ascending=False)
    defs_by_theme = defs.set_index("theme") if not defs.empty else pd.DataFrame()
    min_by_theme = defs_by_theme["min_constituents"] if "min_constituents" in defs_by_theme else pd.Series(dtype=float)
    max_by_theme = defs_by_theme["max_constituents"] if "max_constituents" in defs_by_theme else pd.Series(dtype=float)
    thin = themes[themes.index.map(lambda theme: themes[theme] < min_by_theme.get(theme, 5))]
    overfull = themes[themes.index.map(lambda theme: themes[theme] > max_by_theme.get(theme, 25))]
    print(f"validated {memberships['ticker'].nunique()} tickers across {len(themes)} themes from {path}")
    print(f"theme definitions: {len(defs)}")
    print(themes.to_string())
    if not thin.empty:
        print("\nwatchlist-only themes below min constituents:")
        print(thin.to_string())
    if not overfull.empty:
        print("\noverfull themes above max constituents:")
        print(overfull.to_string())


if __name__ == "__main__":
    main()
