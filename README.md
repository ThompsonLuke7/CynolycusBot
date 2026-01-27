# CynolycusBot
ML Trading Bot

## Entry Points and Args

Use the scripts below from the repo root. For full argument lists, run `python <script> -h`.

### Feature/Label Pipeline
Script: `main.py`

Common args:
- `--ticker` (default `$SPY`)
- `--timeframe` (Alpaca timeframe, default `1Hour`)
- `--start` / `--end` (ISO datetime strings)
- `--limit` (max bars)
- `--adjustment` (`raw|split|all`)
- `--refresh-data` / `--no-refresh-data`
- `--use-cached` / `--no-use-cached`
- `--save-processed` / `--no-save-processed`
- `--label-mode` (`leg|swing`)
- `--model` (`all|Tree|LSTM`)
- `--train-frac` / `--val-frac`
- `--plot-only`
- `--plot-timeframe` (e.g. `15T`, `1H`)
- `--plot-type` (comma-separated; see “Label Plots”)
- `--save-plot` (single plot output path)

Example:
```powershell
python .\main.py --ticker SPY --no-refresh-data --no-use-cached --model all --label-mode leg
```

### Regenerate Labels from plot_frame
Script: `Features/generate_labels.py`

Args:
- `--ticker` (default `$SPY`)
- `--dataset` (default `15min`)
- `--plot-frame` (optional override path to `plot_frame.parquet`)

Example:
```powershell
python -m Features.generate_labels --ticker SPY --dataset 15min
```

### iTransformer Training (with plots)
Script: `Models/iTransformer/run_train.py`

Core args:
- `--ticker`, `--dataset_name`
- `--label_mode` (`swing|leg|continuation|mfe|mae|mfe_mae|exhaustion`)
- `--sides` (e.g. `long,short`)
- `--x_filename` (defaults to `X_<dataset>_lstm.parquet`)
- `--seq_len`, `--epochs`, `--batch_size`, `--lr`, `--weight_decay`
- `--use_ga` plus GA args (`--ga_population_size`, `--ga_generations`, `--ga_max_features`, ...)

Custom data path (regression/binary):
- `--x_path`, `--y_path`, `--task`

Example:
```powershell
python .\Models\iTransformer\run_train.py --ticker SPY --dataset_name 15min --label_mode continuation --sides long,short
```

### iTransformer Training (no plots)
Script: `Models/iTransformer/itransformer_train.py`  
Same args as `run_train.py`, but no plotting.

### GA-XGBoost OOF/Probs Pipeline
Script: `Models/ga_xgboost/train.py`

Args:
- `--refresh-masks` (re-run GA selection on train split)

Example:
```powershell
python .\Models\ga_xgboost\train.py --refresh-masks
```

### Label Plots
Use `main.py --plot-only` with `--plot-type`:
- `atr_swing`
- `leg_segmentation`
- `continuation`
- `swing_state_machine`
- `mfe_mae`
- `bars_to_exhaustion`
- `all_labels`

Aliases: `atr`, `leg`, `state_machine`, `swing_state`, `mfe`, `mae`, `exhaustion`.

Example:
```powershell
python .\main.py --plot-only --plot-type mfe_mae --plot-timeframe 15T --no-refresh-data
```
