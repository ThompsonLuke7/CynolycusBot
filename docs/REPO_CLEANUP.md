# Repo Cleanup Runbook

Created 2026-06-10. State at creation: 5,493 tracked files, 286 MB packed git history.
4,747 of those tracked files (~457 MB on disk) are generated artifacts that now match
`.gitignore` — they were committed before the ignore rules existed.

What was already done (staged in the working tree, commit when ready):

- `.gitignore` rewritten: deduplicated, stale `theme_expansion/` paths fixed to
  `theme_expansion_legacy/`, Schwab token files ignored, and negation rules added so
  live dependencies stay tracked (`Data/*.py`, `Data/plots/*.py`, `.vscode/settings.json`,
  `**/config/*.{json,csv}`, `theme_expansion_legacy/data/*.csv`,
  `theme_expansion_legacy/outputs/universe_filter.csv` + its `.py` scripts,
  `multi_ticker_swing{,_htf}/models/*.json`).
- Removed tracked junk: `.codex` (empty file) and the two
  `*.tgz:Zone.Identifier` index entries (Windows ADS artifacts that forced this clone
  into sparse-checkout mode, because `:` is illegal in Windows filenames).
- Research docs moved to `docs/`: `ResearchPaperSummarySoFar.md`,
  `AdvisorResearchProjectUpdate.md`, `InvestingResearchProject.docx`, `pipeline.txt`.

---

> **2026-06-10 update:** the repo was reorganized into `core/`, `signals/`, `themes/`,
> and `strategies/` (see README Repo Map). Old paths below refer to git history;
> current working-tree paths use the new prefixes (e.g. `core/API/Schwab_API/`,
> `themes/theme_expansion_legacy/`). After pulling the reorg commit on another
> machine, run `bash scripts/migrate_layout_2026_06.sh` to move leftover untracked
> data into the new locations BEFORE running anything.

## 1. URGENT — Schwab OAuth tokens are committed

`core/API/Schwab_API/schwab_token.json` and `schwab_tokens.json` contain real
`access_token` / `refresh_token` / `id_token` values and are tracked in git, i.e. pushed
to GitHub and present in history.

1. Check repo visibility at https://github.com/ThompsonLuke7/CynolycusBot — if it is
   public, treat the tokens as fully compromised.
2. Rotate regardless: revoke the app authorization in the Schwab developer portal (or
   re-run the OAuth flow to invalidate the refresh token). Access tokens expire in
   ~30 min and refresh tokens in 7 days, but rotate anyway.
3. The files are now gitignored, but they remain tracked and in history until you run
   sections 2 and 4 below.

## 2. Untrack the 4,747 committed artifacts

**Run this on whichever machine has the fuller data / runs live trading** (probably the
other machine), not necessarily here. `git rm --cached` never touches the working tree
of the machine where it runs — but when the *other* clone pulls the commit, git deletes
its working copies of those files. Live SPY inference reads model artifacts from
`Data/models/ga_xgboost/`, so do not let the live machine pull this commit blind; make
the commit there instead, or copy `Data/` aside before pulling.

```powershell
# Pull the .gitignore changes first, then:
git ls-files -i -c --exclude-standard            # review the list
git ls-files -i -c --exclude-standard -z | git update-index --force-remove -z --stdin
git commit -m "Untrack generated artifacts now covered by .gitignore"
git push
```

What gets untracked (files stay on disk where the command runs):

| Path | Files | What it is |
|---|---|---|
| `Data/models/` | 4,253 | GA-XGBoost generation checkpoints |
| `Data/inference/` | 266 | live-run logs, broker-state.jsonl |
| `theme_expansion_legacy/outputs/` | ~100 | theme backtest outputs (keeps `universe_filter.csv` + `.py`) |
| `Data/processed/`, `Data/raw/`, `Data/outputs/` | ~58 | datasets, splits, PPO checkpoints |
| `UI/swing_audit/` | 32 | swing session audit logs |
| `momentum_expansion/plots/` etc. | ~35 | plot/backtest churn |
| `API/Schwab_API/schwab_token*.json` | 2 | leaked OAuth tokens |

On the *other* clone, after this commit lands: copy aside anything it deletes that you
still need (everything is regenerable or recoverable via
`git checkout <pre-cleanup-sha> -- <path>`).

## 3. Disable sparse checkout (this machine)

Sparse checkout exists only to dodge the two `Zone.Identifier` filenames. After the
commit that removes them is created, run on this machine:

```powershell
git sparse-checkout disable
```

## 4. Purge history (shrinks 286 MB → small, erases leaked tokens)

Do this after sections 1–2, at a moment when both machines are pushed/clean. Work in a
**fresh clone** (git-filter-repo requires it), then force-push; every other clone must
re-clone (or `git fetch && git reset --hard origin/main`).

```powershell
pip install git-filter-repo
git clone https://github.com/ThompsonLuke7/CynolycusBot CynolycusBot-rewrite
cd CynolycusBot-rewrite
git filter-repo --invert-paths `
  --path API/Schwab_API/schwab_token.json `
  --path API/Schwab_API/schwab_tokens.json `
  --path core/API/Schwab_API/schwab_token.json `
  --path core/API/Schwab_API/schwab_tokens.json `
  --path Data/models --path Data/inference `
  --path UI/swing_audit `
  --path-glob 'Data/processed/**' --path-glob 'Data/raw/**' `
  --path-glob 'Data/outputs/**' `
  --path-glob 'theme_expansion_legacy/outputs/**/*.parquet' `
  --path-glob 'themes/theme_expansion_legacy/outputs/**/*.parquet' `
  --path-glob '*.tgz:Zone.Identifier'
git remote add origin https://github.com/ThompsonLuke7/CynolycusBot
git push origin --force --all
git push origin --force --tags
```

Note: `--path Data/models` etc. removes those paths from *all* commits, including HEAD —
that is why section 2 (untracking) must be committed first, and why this runs only after
artifacts are safe on disk. GitHub support can also clear cached views of old commits if
the repo was ever public ("remove sensitive data" request).

## 5. Optional follow-ups

- Retention policy for `Data/models/ga_xgboost/` (4,197 files locally): keep best-N
  runs, archive the rest outside the repo.
- Large artifacts that *should* sync between machines (datasets, model binaries) belong
  in DVC, git-lfs, or a cloud bucket — not plain git.
