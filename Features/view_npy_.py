import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
processed_dir = PROJECT_ROOT / "Data" / "processed" / "base"

data_path = processed_dir / "X_spy_daily.npy"
features_txt = processed_dir / "features_spy_daily.txt"

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

df.to_csv(processed_dir / "X_spy_daily_preview.csv", index=False)
