Backup of `Policy/Agent/run_train.py` parser defaults before magnitude-sweep update.

Date: 2026-02-18

Previous defaults:
- `--entropy-coef`: `0.004`
- `--policy-hidden-size`: `128`
- `--policy-dropout-p`: `0.05`
- `--convex-theta`: `0.00005`
- `--convex-risk-lambda`: `0.0`
- `--size-change-penalty-ret`: `0.00003`
- `--saturation-threshold`: `0.90`
- `--saturation-penalty-ret`: `0.0`

Updated defaults now set in `Policy/Agent/run_train.py`:
- `--entropy-coef`: `0.0015`
- `--policy-hidden-size`: `256`
- `--policy-dropout-p`: `0.03`
- `--convex-theta`: `0.0002`
- `--convex-risk-lambda`: `12.0`
- `--size-change-penalty-ret`: `0.0001`
- `--saturation-threshold`: `0.75`
- `--saturation-penalty-ret`: `0.0002`
