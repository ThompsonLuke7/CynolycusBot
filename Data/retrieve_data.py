import argparse
from pathlib import Path

import yfinance as yf


def normalize_ticker(ticker: str) -> str:
    """
    Normalize user input like "$SPY" to the plain uppercase ticker symbol.
    """
    clean = ticker.strip().upper()
    if clean.startswith("$"):
        clean = clean[1:]
    return clean


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"


def get_ticker_dir(ticker: str, base_dir: Path | None = None) -> Path:
    base = base_dir if base_dir is not None else DATA_DIR
    slug = normalize_ticker(ticker).lower()
    return base / slug


def get_raw_dir(ticker: str, base_dir: Path | None = None) -> Path:
    return get_ticker_dir(ticker, base_dir) / "raw"


def get_output_path(ticker: str, base_dir: Path | None = None) -> Path:
    """
    Build the CSV path for the ticker inside the Data directory.
    """
    slug = normalize_ticker(ticker).lower()
    raw_dir = get_raw_dir(ticker, base_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / f"{slug}_data.csv"


def retrieve_data(ticker: str, start="2015-01-01", interval="1d"):
    """
    Download historical data for the requested ticker.
    """
    clean_ticker = normalize_ticker(ticker)
    return yf.download(clean_ticker, start=start, interval=interval)


def main(ticker: str = "$SPY", start="2015-01-01", interval="1d") -> Path:
    data = retrieve_data(ticker, start=start, interval=interval)
    output_path = get_output_path(ticker)
    data.to_csv(output_path)
    print(f"Saved {normalize_ticker(ticker)} data to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download historical daily data for a ticker."
    )
    parser.add_argument(
        "--ticker",
        default="$SPY",
        help='Ticker symbol to download (e.g., "$AAPL"). Defaults to "$SPY".',
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Start date for the download (default: 2015-01-01).",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Data interval to download (default: 1d).",
    )
    args = parser.parse_args()
    main(ticker=args.ticker, start=args.start, interval=args.interval)
