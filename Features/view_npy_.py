from pathlib import Path

import numpy as np
import pandas as pd

from Data.load_data import get_ticker_processed_base_dir
from Data.retrieve_data import normalize_ticker

def _infer_prefix(processed_dir: Path, slug: str) -> str | None:
    candidates = sorted(processed_dir.glob(f"X_{slug}_*.npy"))
    if not candidates:
        candidates = sorted(processed_dir.glob(f"X_{slug}_*.parquet"))
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    name = newest.stem
    return name[2:] if name.startswith("X_") else None


ticker = "$SPY"
slug = normalize_ticker(ticker).lower()
processed_dir = get_ticker_processed_base_dir(ticker)
prefix = _infer_prefix(processed_dir, slug) or f"{slug}_daily"

data_path = processed_dir / f"X_{prefix}.npy"
features_txt = processed_dir / f"features_{prefix}.txt"

X = np.load(data_path)
print("X shape:", X.shape)

# Load column names if available; otherwise fall back to generic names
if features_txt.exists():
    cols = [
        line.strip() for line in features_txt.read_text().splitlines() if line.strip()
    ]
else:
    cols = [f"f{i}" for i in range(X.shape[1])]

df = pd.DataFrame(X, columns=cols)

print("\nFirst 5 rows (transposed):")
print(df.head().T)  # values stacked under each column for easier scanning

print("\nColumn summary:")
print(df.describe().T)

df.to_csv(processed_dir / f"X_{prefix}_preview.csv", index=False)
