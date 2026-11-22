# Stage 3 pick: forward (live shadow) evaluation, pre-registered

Fixed 2026-11-20, before any forward outcome existed. It is supplementary to the sealed test
in `27_stage3_preregistration.md`, which stays the formal confirmation (about 2027-06).

## What runs

- **Boosters:**
  - two final boosters of the frozen pick L2_20__pit (selection 621d0acaa1cd), trained
    2024-04-10 to 2026-04-10 with the walk-forward recipe;
  - stored in `Data/research/stage3_final/`, from `stage3_final.tar` built by
    `stage3_final_colab.py`;
  - validation log-loss 0.4966 against 0.5006 for the base rate; best iterations 143 and 108.
- **Shadow scoring:** each night (a post-stamp step in `nightly_data_readiness.sh`),
  `score_archive_stage3.py` scores every archived live momentum decision and writes to
  `Data/research/decision_state_shadow/stage3_L2_20/`. Nothing trades on these scores.
- **Plumbing check:**
  - Oct 2023 – Feb 2024 lies outside both this booster's training window and the
    walk-forward fold-9 model's; that fold's out-of-fold scores cover it.
  - Within a bar, the final booster and the fold-9 model rank names with a Spearman of only
    0.30. Same recipe, different 2-year windows: the model is window-dependent, as validation
    already showed by year.
- **First live run after the fix (2026-11-20 14:25 ET pass):** 2,805 names scored.
  - Spearman with the live momentum score over the snapshot: +0.25.
  - Top-20 overlap: 5.

## Rule (`forward_eval_stage3.py`)

- **Rows:**
  - post-fix runs only (decided from 2026-11-19 14:00 UTC);
  - the first run for each decision bar;
  - bars with at least 200 scored names;
  - outcome status "ok".
- **Score:** the shadow score's percentile rank, residualised within the bar on ranks of
  daily ATR%, 60-bar beta, price and bar dollar volume. These are the archive's stand-ins for
  the ATR/beta/px/ADV controls; ADV is proxied by bar dollar volume.
- **Outcome:** SPY-excess return from the executable entry (`decision_state_outcomes`),
  demeaned within the bar's ATR decile. This is the L2 construction.
- **Metric:** mean within-bar Spearman IC at h = 10 and 20 sessions, with a week-block
  bootstrap CI.
- **Interpretation:**
  - A read counts as evidence only when its MDE is at or below 0.03. At validation's rate
    that takes about 12 months of bars.
  - Before then it is monitoring only: it can expose a sign flip or a bug, but it cannot
    confirm or refute the pick.
  - The live momentum score is reported on the same rows as a comparison, not a gate.
- **No re-cut:** none of the windows, controls or horizons may be changed after outcomes
  arrive. Any change starts a new pre-registration.
