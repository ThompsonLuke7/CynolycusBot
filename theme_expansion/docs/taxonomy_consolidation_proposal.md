# Theme taxonomy: consolidation + universe-expansion proposal

Status: **DRAFT for review** — nothing has been merged. This documents the proposed
changes before I rewrite `theme_map_v3_1.csv` / `theme_definition_v3_1.csv`.

## Why

1. **Universe expansion.** Align the theme universe with the momentum_expansion pool
   (1,110 tickers, the largest). 535 momentum tickers are not yet in any theme; each
   must be assigned to a theme to participate (blank theme = dropped by
   `load_theme_memberships`).
2. **Consolidate near-duplicates.** Several themes are legacy/narrow cuts paired with a
   proper v3.1 cut. Splitting a small constituent pool across synonyms makes each theme
   index noisier, and the label is *excess return vs the theme's benchmark*, so duplicate
   themes with different benchmarks dilute the signal.
3. **Add 3 genuinely-missing alpha themes:** `cooling`, `foundry`, `eda`.
4. **Make the hierarchy explicit:** add Level-1 `sector` + Level-2 `industry_group`
   metadata columns (non-breaking; pipeline still scores at the Level-3 `theme`).

## Consolidation rule (evidence-based)

In every duplicate group, the **SPY-benchmarked** theme is the legacy/base version and the
**sector-ETF-benchmarked** theme is the canonical v3.1 cut (confirmed by `benchmark` and by
`biotech` already being `is_tradable=False, is_watchlist_only=True`).

> **Rule:** keep the sector-ETF-benchmarked theme as canonical; merge the legacy SPY
> duplicate into it (union constituents); delete the legacy theme.

---

## Table A — Confident merges (recommend applying)

| Merge (legacy, SPY) | → Canonical (benchmark) | Notes |
|---|---|---|
| `memory`, `storage` | `memory_storage` (SMH) | but move `DELL`,`HPE`,`HPQ` → `hardware_devices` (PC/server, not storage) |
| `networking` | `networking_optical` (QQQ) | |
| `retail` | `retail_consumer` (XLY) | |
| `consumer` | `restaurants_food` (XLY) | ⚠️ `consumer` constituents are all restaurants (CMG/DPZ/DRI/SBUX/TXRH/YUM) — mislabeled |
| `restaurants` | `restaurants_food` (XLY) | |
| `fintech`, `payments` | `fintech_payments` (ARKF) | |
| `nuclear`, `uranium` | `nuclear_uranium` (URA) | |
| `defense` | `defense_aerospace` (ITA) | |
| `managed_care` | `managed_care_insurance` (XLV) | |
| `medtech` | `medtech_devices` (IHI) | |
| `pharma` | `pharma_largecap` (XLV) | |
| `biotech` (already watchlist) | `biotech_innovation` (XBI) | |
| `asset_managers` | `asset_managers_alts` (XLF) | |
| `copper` | `copper_metals` (XME) | |

Net: removes 16 legacy themes.

## Table B — Flagged: distinct vs. duplicate (need your call; my lean in **bold**)

| Group | Members | My recommendation |
|---|---|---|
| Data centers | `data_centers`(SPY), `datacenter_infra`(XLI), `datacenter_reits_towers`(VNQ) | **Keep infra + reits distinct; dissolve `data_centers`**, routing compute names (CORZ, SMCI, PSTG) → `ai_infrastructure`, VRT → new `cooling`, DLR/EQIX → reits |
| Aerospace | `aerospace`(SPY), `defense_aerospace`(ITA) | **Keep both** — commercial vs defense are different cycles (your RBICS point) |
| Precious metals | `gold`(SPY), `silver`(SPY), `gold_silver_miners`(GDX) | **Merge gold+silver → `gold_silver_miners`** (small pools, same driver) |
| Semis | `semis`(SMH), `semis_compute`(SMH), `semicap_equipment`(SMH), `advanced_packaging`(SMH) | **Merge `semis` → `semis_compute`; keep semicap + packaging distinct**; add `foundry` + `eda` |
| AI/Software | `ai_apps`(SPY), `ai_apps_agents`(IGV), `cloud`(XLK), `cloud_neocloud`(QQQ) | **Merge `ai_apps` → `ai_apps_agents`; keep cloud vs cloud_neocloud** (neocloud = miners→AI) |
| Energy umbrella | `energy`(SPY, n=28), `oil_gas`(XLE) | **Dissolve generic `energy`** into the specific oil sub-themes; keep `oil_integrated/_exploration/_services`, `natural_gas` |
| Banks | `mega_banks`, `regional_banks`, `banks_brokers_exchanges` | **Keep all** — distinct rate/credit drivers |

## New themes to add

| Theme | Sector / industry_group | Benchmark | Seed constituents |
|---|---|---|---|
| `cooling` | Industrials / AI Infrastructure | XLI | VRT, modine (MOD), nVent (NVT), … (refine from profiles) |
| `foundry` | Technology / Semiconductors | SMH | TSM, GFS, UMC, … |
| `eda` | Technology / Semiconductors | SMH | SNPS, CDNS, (ANSS) |

## Hierarchy columns (added to `theme_definition`)

- **`sector`** (L1): Technology, Healthcare, Financials, Industrials, Energy, Materials,
  Consumer, Utilities, Real Estate, Communication, Macro.
- **`industry_group`** (L2): e.g. Semiconductors, Software, AI Infrastructure, Networking,
  Defense, Biotech, Pharma, Power, Nuclear, Crypto, Space, Banks, Insurance, …
- Existing `category` column retained for backward compat (pipeline reads it).

## Next steps after you approve this

1. Finish yfinance profile fetch for the 535 (running now).
2. Produce `theme_map` additions: assign each of the 535 to a (consolidated) theme using
   Yahoo industry + business summary + your seed examples → reviewable CSV diff.
3. Apply consolidation + new themes + hierarchy columns to `theme_definition`.
4. Re-run `02 download → 03 → 04 → 26 → export` on the expanded universe.
