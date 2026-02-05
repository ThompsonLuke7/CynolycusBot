import pandas as pd
import pandas_ta as ta
from typing import Iterable, List, Optional, Set


def prepare_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename common OHLCV column names to the lowercase names that pandas_ta
    expects by default: 'open', 'high', 'low', 'close', 'volume'.
    """
    rename_map = {}
    if "Open" in df.columns:
        rename_map["Open"] = "open"
    if "High" in df.columns:
        rename_map["High"] = "high"
    if "Low" in df.columns:
        rename_map["Low"] = "low"
    if "Close" in df.columns:
        rename_map["Close"] = "close"
    if "Adj Close" in df.columns and "adj_close" not in df.columns:
        rename_map["Adj Close"] = "adj_close"
    if "Volume" in df.columns:
        rename_map["Volume"] = "volume"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def add_all_pandasta_indicators(
    df: pd.DataFrame,
    exclude_indicators: Optional[Iterable[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Add *all* pandas_ta indicators to the DataFrame via df.ta.<indicator>(append=True),
    except those explicitly listed in `exclude_indicators`.

    Because the current pandas_ta API for `indicators` does not expose categories,
    we work from the flat list of indicator names and let you manually exclude any
    you don't want (e.g. "performance" or "statistics" ones once you identify them).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain OHLCV columns. We'll call prepare_ohlcv_columns() first.
    exclude_indicators : Iterable[str], optional
        List of indicator function names (e.g. "log_return", "percent_return")
        to skip. Default: None (include everything).
    verbose : bool
        If True, print progress and any failures.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with all indicator columns appended.
    """
    df = prepare_ohlcv_columns(df)

    if exclude_indicators is None:
        exclude_set: Set[str] = set()
    else:
        exclude_set = {name.lower() for name in exclude_indicators}

    # Get flat list of indicator names
    indicator_names: List[str] = df.ta.indicators(as_list=True)

    if verbose:
        print(f"Found {len(indicator_names)} pandas_ta indicators.")
        if exclude_set:
            print("Excluding indicators:", sorted(exclude_set))

    for name in indicator_names:
        lname = name.lower()

        if lname in exclude_set:
            if verbose:
                print(f"[SKIP] {name} (user-excluded)")
            continue

        # Some indicators may not work with default params; we attempt and skip failures
        try:
            func = getattr(df.ta, name)
        except AttributeError:
            if verbose:
                print(f"[SKIP] df.ta has no attribute '{name}'")
            continue

        try:
            func(append=True)
            if verbose:
                print(f"[OK] {name}")
        except Exception as e:
            if verbose:
                print(f"[FAIL] {name} -> {e}")

    return df


def list_all_indicator_names(df: pd.DataFrame) -> List[str]:
    """
    Helper to list indicator names available from df.ta.indicators(as_list=True).
    """
    df = prepare_ohlcv_columns(df)
    return df.ta.indicators(as_list=True)
