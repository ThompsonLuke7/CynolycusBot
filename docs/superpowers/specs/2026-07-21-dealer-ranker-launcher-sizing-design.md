# Dealer Ranker Launcher Sizing Design

## Goal

Restore supervised live-server startup while retaining the Dealer Ranker's new dollar-notional sizing model and its $5,000-per-entry default.

## Root cause

Commit `e1c590e` migrated `UI.combined_server` and `strategies/dealer_positioning/live_ranked_options.py` from fixed `--contracts` sizing to `--target-notional`. `scripts/run_live_server.sh` was not updated in the same commit, so it still passes the removed `--dealer-ranker-contracts` option and argparse exits before the server starts.

## Design

Change the supervisor's Dealer Ranker argument to `--dealer-ranker-target-notional` and expose the matching `DEALER_RANKER_TARGET_NOTIONAL` environment override with a default of `5000`. Do not restore the fixed-contract option or add a compatibility alias because the current stack has one intentional sizing model: contract quantity is derived from option premium and target dollars.

Add a focused regression test that extracts Dealer Ranker flags passed by `scripts/run_live_server.sh` and asserts that every one is declared by `UI.combined_server`. This tests the launcher/parser boundary that failed and catches future partial CLI migrations without starting any trading process.

## Safety and validation

The change does not enable live routing, submit orders, or start the server. Validate with the new regression test, the relevant combined-server tests, shell syntax checking, Python compilation, and a help-output check confirming the new option is accepted and the removed option is absent.
