#!/usr/bin/env bash
# One-time layout migration helper for the 2026-06 repo reorganization.
#
# Run this on a machine AFTER pulling the reorg commit. `git pull` moves only
# tracked files; untracked data/artifacts stay behind in the old directories.
# This script moves that leftover untracked content into the new locations,
# then deletes the old directories if they are empty.
#
# Usage: bash scripts/migrate_layout_2026_06.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

declare -A MOVES=(
  [API]=core/API
  [shared_universe]=core/shared_universe
  [shared_plotting]=core/shared_plotting
  [news]=signals/news
  [events]=signals/events
  [catalysts]=signals/catalysts
  [social_attention]=signals/social_attention
  [meta_context]=signals/meta_context
  [dynamic_theme]=themes/dynamic_theme
  [theme_expansion_legacy]=themes/theme_expansion_legacy
  [Features]=strategies/spy_intraday/Features
  [Models]=strategies/spy_intraday/Models
  [Policy]=strategies/spy_intraday/Policy
  [multi_ticker_swing]=strategies/multi_ticker_swing
  [multi_ticker_swing_htf]=strategies/multi_ticker_swing_htf
  [momentum_expansion]=strategies/momentum_expansion
  [momentum_scalper]=strategies/momentum_scalper
)

for old in "${!MOVES[@]}"; do
  new="${MOVES[$old]}"
  [ -d "$old" ] || continue
  echo "Migrating leftover files: $old/ -> $new/"
  mkdir -p "$new"
  # Merge old into new without overwriting anything git already placed.
  (cd "$old" && find . -type f -print0) | while IFS= read -r -d '' f; do
    src="$old/${f#./}"
    dst="$new/${f#./}"
    if [ -e "$dst" ]; then
      echo "  SKIP (exists): $dst"
    else
      mkdir -p "$(dirname "$dst")"
      mv "$src" "$dst"
    fi
  done
  find "$old" -type d -empty -delete
  if [ -d "$old" ]; then
    echo "  NOTE: $old/ not empty after merge (conflicting files were skipped) — review manually."
  fi
done

echo "Done. Verify with: python scripts/run_safe_smoke_tests.py"
