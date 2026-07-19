# LEGACY MANUAL HELPER ONLY.
# Production scheduling and the complete bounded pipeline are owned by
# scripts/run_live_server.sh -> UI.combined_server -> nightly_market_data.sh.
# Do not install this file as a Windows Scheduled Task.
param(
    [string]$Python = "python",
    [string]$LogPath = "news\data\processed\nightly_cron.log"
)

$ErrorActionPreference = "Continue"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$logDir = Split-Path $LogPath -Parent
if ($logDir) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    "[$stamp] $Message" | Tee-Object -FilePath $LogPath -Append
}

Write-Log "nightly_market_data.ps1 starting"
Write-Log "CBOE snapshot - full universe"
& $Python -u -m news.main --stage cboe-snapshot *>> $LogPath
Write-Log "CBOE snapshot exit=$LASTEXITCODE"

$target = (Get-Date).ToUniversalTime().Date.AddDays(-1)
while ($target.DayOfWeek -eq "Saturday" -or $target.DayOfWeek -eq "Sunday") {
    $target = $target.AddDays(-1)
}
Write-Log "FINRA pulling prior trading day $($target.ToString('yyyy-MM-dd'))"
$code = @'
from pathlib import Path
import pandas as pd
from news.sources import fetch_finra_short_volume_day

target = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)
while target.weekday() >= 5:
    target -= pd.Timedelta(days=1)
out_path = Path(f"news/data/processed/finra_daily/{target.strftime('%Y%m%d')}.parquet")
out_path.parent.mkdir(parents=True, exist_ok=True)
if out_path.exists():
    print(f"{out_path} already exists - skipping")
else:
    df = fetch_finra_short_volume_day(target)
    if df.empty:
        print(f"no rows for {target.date()} (likely market holiday)")
    else:
        df.to_parquet(out_path, index=False)
        print(f"wrote {len(df):,} rows to {out_path}")
'@
$code | & $Python - *>> $LogPath
Write-Log "FINRA fetch exit=$LASTEXITCODE"

Write-Log "FINRA compacting daily files"
$compact = @'
from pathlib import Path
import pandas as pd

daily_dir = Path("news/data/processed/finra_daily")
consolidated = Path("news/data/processed/finra_short_volume.parquet")
daily_dir.mkdir(parents=True, exist_ok=True)
dailies = sorted(daily_dir.glob("*.parquet"))
if not dailies:
    print("no daily files to compact")
    raise SystemExit(0)

frames = []
if consolidated.exists():
    frames.append(pd.read_parquet(consolidated))
for path in dailies:
    frames.append(pd.read_parquet(path))
all_df = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "ticker"])
all_df.to_parquet(consolidated, index=False)
print(f"consolidated: {len(all_df):,} rows -> {consolidated}")
'@
$compact | & $Python - *>> $LogPath
Write-Log "nightly_market_data.ps1 complete"
