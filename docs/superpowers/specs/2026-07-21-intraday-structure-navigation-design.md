# Intraday Structure Navigation

## Goal

Make the already-running, paper-only Intraday Structure dashboard reachable
from the shared combined-server navigation.

## Design

Add one entry to `UI.ui_chrome.NAV_PORTS`:

- Label: `Intraday Structure`
- Port: `8774`

`NAV_HTML` derives links from this shared port map, so the entry will appear
on the hub and every combined-server dashboard. The existing Hub card,
combined-server startup wiring, candidate universe, and paper-only behavior
remain unchanged.

## Error Handling

The navigation link is static and follows the existing navigation pattern. If
the dashboard is intentionally disabled with `--no-intraday-structure`, its
link may return a connection error; that behavior matches the existing fixed
port-map links and does not affect server startup or trading paths.

## Verification

Add a focused UI test that asserts the shared navigation output includes the
`Intraday Structure` label and port `8774`, then run the hub and intraday
structure UI tests plus a Python compile check.
