"""
Build candidate v4 theme taxonomy + universe expansion (NON-DESTRUCTIVE).

Reads:
  data/theme_definition_v3_1.csv
  data/theme_map_v3_1.csv
  data/ticker_profiles_new.csv      (Yahoo profiles for the 535 new tickers)

Writes (candidates for review; v3_1 untouched):
  data/theme_definition_v4.csv
  data/theme_map_v4.csv
  data/taxonomy_v4_assignment_review.csv   (per-new-ticker decision + confidence)
  data/taxonomy_v4_diff_report.txt

Applies the approved consolidation rule (legacy SPY -> canonical sector-ETF),
the flagged dissolves, the 3 new alpha themes, hierarchy columns, and assigns the
509 new equities from Yahoo industry (+ business-summary heuristics).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# --- 1. Confident merges: legacy theme -> canonical theme (whole-theme) ---------
CONSOLIDATION = {
    "memory": "memory_storage",
    "storage": "memory_storage",          # DELL/HPE/HPQ rerouted below
    "networking": "networking_optical",
    "retail": "retail_consumer",
    "consumer": "restaurants_food",        # 'consumer' is actually restaurants
    "restaurants": "restaurants_food",
    "fintech": "fintech_payments",
    "payments": "fintech_payments",
    "nuclear": "nuclear_uranium",
    "uranium": "nuclear_uranium",
    "defense": "defense_aerospace",
    "managed_care": "managed_care_insurance",
    "medtech": "medtech_devices",
    "pharma": "pharma_largecap",
    "biotech": "biotech_innovation",
    "asset_managers": "asset_managers_alts",
    "copper": "copper_metals",
    # flagged merges (approved):
    "gold": "gold_silver_miners",
    "silver": "gold_silver_miners",
    "semis": "semis_compute",
    "ai_apps": "ai_apps_agents",
}

# Per-ticker overrides when a *specific* legacy ticker belongs elsewhere
TICKER_REROUTE = {
    "DELL": "hardware_devices", "HPE": "hardware_devices", "HPQ": "hardware_devices",
}

# --- 2. Dissolves: theme removed, constituents routed per-ticker (default fallback) ---
DISSOLVE = {
    "data_centers": {
        "_default": "ai_infrastructure",
        "DLR": "datacenter_reits_towers", "EQIX": "datacenter_reits_towers",
        "CSCO": "networking_optical", "PSTG": "memory_storage", "VRT": "cooling",
        "CORZ": "cloud_neocloud",
    },
    "energy": {
        "_default": "oil_gas",
        "ENPH": "solar", "FSLR": "solar", "RUN": "solar",
        "CAT": "industrial_automation",
    },
}

# --- 3. New alpha themes (kept granular per approval) ---------------------------
NEW_THEMES = [
    # theme, category, min, max, allow_etfs, benchmark, is_tradable, is_watchlist_only
    ("cooling",   "Industrials",  3, 15, False, "XLI", True, False),
    ("foundry",   "Technology",   3, 12, False, "SMH", True, False),
    ("eda",       "Technology",   3, 10, False, "SMH", True, False),
    ("telecom",   "Communication", 6, 25, False, "XLC", True, False),
    ("chemicals", "Materials",    6, 30, False, "XLB", True, False),
]
# Tickers whose PRIMARY theme should become the new alpha theme (their core identity).
# The previous theme_1 is preserved by shifting it into a free secondary slot.
NEW_THEME_PRIMARY = {
    "TSM": "foundry", "GFS": "foundry", "UMC": "foundry", "TSEM": "foundry",
    "SNPS": "eda", "CDNS": "eda", "ANSS": "eda",
    "VRT": "cooling", "NVT": "cooling", "AAON": "cooling", "SPXC": "cooling",
}
# Tickers that should also carry the new theme as a SECONDARY membership.
NEW_THEME_SECONDARY = {
    "MOD": "cooling",   # Modine: autos primary, data-center thermal secondary
}

# --- 4. Level-2 industry_group per theme (Level-1 sector derived from category) --
INDUSTRY_GROUP = {
    "semis_compute": "Semiconductors", "semicap_equipment": "Semiconductors",
    "advanced_packaging": "Semiconductors", "foundry": "Semiconductors",
    "eda": "Semiconductors", "memory_storage": "Semiconductors",
    "networking_optical": "Networking", "hardware_devices": "Hardware",
    "electronics_components": "Hardware", "electronic_components": "Hardware",
    "ai_infrastructure": "AI Infrastructure", "cloud": "Software",
    "cloud_neocloud": "AI Infrastructure", "cooling": "AI Infrastructure",
    "data_centers": "AI Infrastructure", "datacenter_infra": "AI Infrastructure",
    "datacenter_reits_towers": "AI Infrastructure",
    "enterprise_software": "Software", "business_apps": "Software",
    "workflow_software": "Software", "vertical_software": "Software",
    "creative_software": "Software", "dev_tools": "Software",
    "ai_apps_agents": "Software", "cybersecurity": "Software",
    "fintech_payments": "Fintech", "quantum_computing": "Emerging Compute",
    "biotech_innovation": "Biotech", "pharma_largecap": "Pharma",
    "medtech_devices": "Medtech", "diagnostics_tools": "Life Sciences Tools",
    "managed_care_insurance": "Healthcare Services", "obesity_glp1_health": "Pharma",
    "defense_aerospace": "Defense", "aerospace": "Aerospace",
    "drones_autonomy": "Defense", "space_satellites": "Space",
    "robotics": "Automation", "industrial_automation": "Automation",
    "additive_manufacturing": "Automation", "construction_infrastructure": "Construction",
    "power_grid_electrification": "Power", "nuclear_uranium": "Nuclear",
    "solar": "Renewables", "renewables_storage": "Renewables",
    "utilities_yield": "Utilities", "oil_gas": "Oil & Gas",
    "oil_integrated": "Oil & Gas", "oil_exploration": "Oil & Gas",
    "oil_services": "Oil & Gas", "natural_gas": "Oil & Gas",
    "gold_silver_miners": "Precious Metals", "copper_metals": "Industrial Metals",
    "lithium_battery_materials": "Battery Materials",
    "rare_earth_critical_minerals": "Critical Minerals", "steel": "Steel",
    "banks_brokers_exchanges": "Banks", "mega_banks": "Banks",
    "regional_banks": "Banks", "insurance": "Insurance",
    "asset_managers_alts": "Asset Management", "crypto_infrastructure": "Crypto",
    "real_estate_reits": "Real Estate", "homebuilders_housing": "Homebuilders",
    "autos": "Autos", "retail_consumer": "Retail", "restaurants_food": "Restaurants",
    "apparel_beauty_lifestyle": "Consumer Brands", "consumer_staples": "Staples",
    "consumer_internet": "Internet", "consumer_internet_marketplaces": "Internet",
    "china_internet": "Internet", "mega_cap_platforms": "Internet",
    "digital_ads_media": "Media", "streaming_entertainment": "Media",
    "gaming_sports_betting": "Gaming", "travel": "Travel", "airlines_travel": "Travel",
    "transport_logistics": "Transport", "agriculture_food_inputs": "Agriculture",
    "telecom": "Telecom", "chemicals": "Chemicals",
}

# Keyword -> finer software theme, in priority order (first match wins). Applied to
# names Yahoo lumps under generic "Software"/"IT Services" -> enterprise_software.
SOFTWARE_KEYWORDS = [
    ("cybersecurity", ["cybersecur", "endpoint", "firewall", "threat detection",
                        "zero trust", "identity and access", "security software", "malware",
                        "network security"]),
    ("fintech_payments", ["payment", "merchant", "fintech", "billing software",
                          "lending platform", "banking software", "point-of-sale"]),
    ("dev_tools", ["developer", "devops", "ci/cd", "observability", "application monitoring",
                   "software development", "api platform", "version control"]),
    ("creative_software", ["design software", "creative", "3d ", "rendering", "video editing",
                           "content creation", "computer-aided design"]),
    ("ai_apps_agents", ["generative ai", "ai-powered", "artificial intelligence platform",
                        "machine learning platform", "large language model", "ai agent"]),
    ("cloud", ["data warehouse", "cloud database", "cloud infrastructure", "data platform",
               "cloud computing platform", "data lake"]),
    ("workflow_software", ["human capital", "human resources", "workflow", "collaboration software",
                           "customer relationship management", "crm", "erp", "supply chain software",
                           "productivity"]),
    ("vertical_software", ["healthcare software", "construction software", "restaurant",
                           "real estate software", "government software", "legal software",
                           "insurance software", "for the ", "industry-specific", "veterinary"]),
]


def refine_software(summary: str) -> str | None:
    if not isinstance(summary, str) or not summary:
        return None
    s = summary.lower()
    for theme, kws in SOFTWARE_KEYWORDS:
        if any(k in s for k in kws):
            return theme
    return None

# --- 5. Yahoo industry -> theme (for the 509 new equities). conf: H/M/L ----------
# (theme, confidence)
INDUSTRY_TO_THEME = {
    "Biotechnology": ("biotech_innovation", "H"),
    "Drug Manufacturers - Specialty & Generic": ("pharma_largecap", "H"),
    "Diagnostics & Research": ("diagnostics_tools", "H"),
    "Medical Devices": ("medtech_devices", "H"),
    "Medical Instruments & Supplies": ("medtech_devices", "H"),
    "Health Information Services": ("medtech_devices", "M"),
    "Medical Care Facilities": ("managed_care_insurance", "M"),
    "Software - Application": ("enterprise_software", "M"),
    "Software - Infrastructure": ("enterprise_software", "M"),
    "Information Technology Services": ("enterprise_software", "M"),
    "Internet Content & Information": ("consumer_internet", "M"),
    "Internet Retail": ("consumer_internet_marketplaces", "M"),
    "Semiconductors": ("semis_compute", "H"),
    "Semiconductor Equipment & Materials": ("semicap_equipment", "H"),
    "Communication Equipment": ("networking_optical", "H"),
    "Computer Hardware": ("hardware_devices", "H"),
    "Consumer Electronics": ("hardware_devices", "H"),
    "Electronic Components": ("electronics_components", "H"),
    "Electronics & Computer Distribution": ("electronics_components", "M"),
    "Scientific & Technical Instruments": ("diagnostics_tools", "M"),
    "Electronic Gaming & Multimedia": ("gaming_sports_betting", "H"),
    "Aerospace & Defense": ("defense_aerospace", "H"),
    "Specialty Industrial Machinery": ("industrial_automation", "M"),
    "Farm & Heavy Construction Machinery": ("industrial_automation", "M"),
    "Metal Fabrication": ("industrial_automation", "M"),
    "Tools & Accessories": ("industrial_automation", "M"),
    "Electrical Equipment & Parts": ("power_grid_electrification", "M"),
    "Pollution & Treatment Controls": ("industrial_automation", "L"),
    "Security & Protection Services": ("defense_aerospace", "L"),
    "Industrial Distribution": ("construction_infrastructure", "L"),
    "Engineering & Construction": ("construction_infrastructure", "H"),
    "Building Materials": ("construction_infrastructure", "M"),
    "Building Products & Equipment": ("construction_infrastructure", "M"),
    "Residential Construction": ("homebuilders_housing", "H"),
    "Specialty Business Services": ("enterprise_software", "L"),
    "Rental & Leasing Services": ("transport_logistics", "L"),
    "Integrated Freight & Logistics": ("transport_logistics", "H"),
    "Trucking": ("transport_logistics", "H"),
    "Marine Shipping": ("transport_logistics", "M"),
    "Airlines": ("airlines_travel", "H"),
    "Auto Parts": ("autos", "H"),
    "Auto Manufacturers": ("autos", "H"),
    "Auto & Truck Dealerships": ("autos", "M"),
    "Recreational Vehicles": ("autos", "L"),
    "Capital Markets": ("banks_brokers_exchanges", "H"),
    "Financial Data & Stock Exchanges": ("banks_brokers_exchanges", "H"),
    "Asset Management": ("asset_managers_alts", "H"),
    "Banks - Regional": ("regional_banks", "H"),
    "Credit Services": ("fintech_payments", "M"),
    "Mortgage Finance": ("regional_banks", "L"),
    "Insurance - Property & Casualty": ("insurance", "H"),
    "Insurance - Life": ("insurance", "H"),
    "Financial Conglomerates": ("banks_brokers_exchanges", "L"),
    "Oil & Gas Equipment & Services": ("oil_services", "H"),
    "Oil & Gas Refining & Marketing": ("oil_integrated", "M"),
    "Oil & Gas Drilling": ("oil_services", "H"),
    "Oil & Gas Midstream": ("natural_gas", "M"),
    "Utilities - Regulated Gas": ("utilities_yield", "H"),
    "Utilities - Regulated Electric": ("utilities_yield", "H"),
    "Utilities - Renewable": ("renewables_storage", "H"),
    "Utilities - Independent Power Producers": ("power_grid_electrification", "M"),
    "Steel": ("copper_metals", "M"),       # no dedicated steel theme; XME covers steel
    "Aluminum": ("copper_metals", "M"),
    "Copper": ("copper_metals", "H"),
    "Other Industrial Metals & Mining": ("copper_metals", "M"),
    "Other Precious Metals & Mining": ("gold_silver_miners", "H"),
    "Gold": ("gold_silver_miners", "H"),
    "Silver": ("gold_silver_miners", "H"),
    "Uranium": ("nuclear_uranium", "H"),
    "Coking Coal": ("copper_metals", "L"),
    "Specialty Chemicals": ("chemicals", "M"),
    "Chemicals": ("chemicals", "M"),
    "Agricultural Inputs": ("agriculture_food_inputs", "H"),
    "Packaging & Containers": ("construction_infrastructure", "L"),
    "Specialty Retail": ("retail_consumer", "H"),
    "Apparel Retail": ("retail_consumer", "H"),
    "Apparel Manufacturing": ("apparel_beauty_lifestyle", "H"),
    "Footwear & Accessories": ("apparel_beauty_lifestyle", "H"),
    "Luxury Goods": ("apparel_beauty_lifestyle", "H"),
    "Household & Personal Products": ("apparel_beauty_lifestyle", "M"),
    "Furnishings, Fixtures & Appliances": ("retail_consumer", "L"),
    "Restaurants": ("restaurants_food", "H"),
    "Resorts & Casinos": ("gaming_sports_betting", "M"),
    "Gambling": ("gaming_sports_betting", "H"),
    "Leisure": ("travel", "M"),
    "Travel Services": ("travel", "H"),
    "Lodging": ("travel", "M"),
    "Grocery Stores": ("consumer_staples", "H"),
    "Food Distribution": ("consumer_staples", "M"),
    "Packaged Foods": ("consumer_staples", "H"),
    "Beverages - Wineries & Distilleries": ("consumer_staples", "H"),
    "Tobacco": ("consumer_staples", "H"),
    "Education & Training Services": ("consumer_internet", "L"),
    "Personal Services": ("consumer_internet", "L"),
    "Entertainment": ("streaming_entertainment", "H"),
    "Broadcasting": ("streaming_entertainment", "M"),
    "Publishing": ("digital_ads_media", "M"),
    "Advertising Agencies": ("digital_ads_media", "H"),
    "Telecom Services": ("telecom", "H"),
    "REIT - Office": ("real_estate_reits", "H"),
    "REIT - Specialty": ("datacenter_reits_towers", "M"),
    "REIT - Diversified": ("real_estate_reits", "H"),
    "REIT - Hotel & Motel": ("real_estate_reits", "H"),
    "REIT - Healthcare Facilities": ("real_estate_reits", "H"),
    "REIT - Mortgage": ("real_estate_reits", "M"),
    "Real Estate Services": ("real_estate_reits", "M"),
    "Real Estate - Development": ("homebuilders_housing", "M"),
    "Conglomerates": ("industrial_automation", "L"),
    "Specialty Business Services ": ("enterprise_software", "L"),
    "Shell Companies": (None, "L"),       # SPAC/holdco -> exclude
}


def main() -> None:
    defs = pd.read_csv(DATA / "theme_definition_v3_1.csv")
    cmap = pd.read_csv(DATA / "theme_map_v3_1.csv")
    prof = pd.read_csv(DATA / "ticker_profiles_new.csv")

    report = []

    # ---- canonical theme set ----
    removed = set(CONSOLIDATION) | set(DISSOLVE)
    canon = [t for t in defs["theme"] if t not in removed]
    canon += [r[0] for r in NEW_THEMES]
    canon_set = set(canon)

    def remap(theme: str, ticker: str) -> str | None:
        """Map an (old theme, ticker) to a canonical theme (or None to drop)."""
        if ticker in TICKER_REROUTE and theme in ("memory", "storage", "memory_storage"):
            return TICKER_REROUTE[ticker]
        if theme in DISSOLVE:
            d = DISSOLVE[theme]
            return d.get(ticker, d["_default"])
        return CONSOLIDATION.get(theme, theme)

    # ---- rebuild theme_definition v4 ----
    new_def = defs[defs["theme"].isin(canon_set)].copy()
    for (th, cat, mn, mx, etf, bm, trad, watch) in NEW_THEMES:
        new_def = pd.concat([new_def, pd.DataFrame([{
            "theme": th, "category": cat, "min_constituents": mn, "max_constituents": mx,
            "allow_etfs": etf, "benchmark": bm, "is_tradable": trad, "is_watchlist_only": watch,
        }])], ignore_index=True)
    new_def["sector"] = new_def["category"].astype(str).str.split("/").str[0]
    new_def["industry_group"] = new_def["theme"].map(INDUSTRY_GROUP).fillna(new_def["sector"])
    new_def = new_def.sort_values("theme").reset_index(drop=True)

    # ---- remap existing map (theme_1/2/3) ----
    theme_cols = [c for c in ["theme_1", "theme_2", "theme_3"] if c in cmap.columns]
    out_map = cmap.copy()
    for _, row in out_map.iterrows():
        for c in theme_cols:
            t = str(row[c]).strip()
            if t and t.lower() != "nan":
                row[c] = remap(t, str(row["ticker"]).strip().upper()) or ""
    # apply via vectorized re-loop (iterrows copy doesn't persist)
    for c in theme_cols:
        out_map[c] = [
            (remap(str(t).strip(), str(tk).strip().upper()) or "")
            if str(t).strip() and str(t).strip().lower() != "nan" else ""
            for t, tk in zip(cmap[c], cmap["ticker"])
        ]

    # ---- assign the 509 new equities ----
    prof = prof.copy()
    prof["ticker"] = prof["ticker"].str.upper()
    is_equity = prof["quoteType"].eq("EQUITY")
    review_rows = []
    add_rows = []
    for _, r in prof.iterrows():
        tk = r["ticker"]
        if not (is_equity.loc[r.name]):
            review_rows.append({"ticker": tk, "sector": r["sector"], "industry": r["industry"],
                                "theme": "", "confidence": "EXCLUDE", "reason": f"non-equity ({r['quoteType']})"})
            continue
        ind = r["industry"]
        theme, conf = INDUSTRY_TO_THEME.get(ind, (None, "L"))
        if theme is None or theme not in canon_set:
            review_rows.append({"ticker": tk, "sector": r["sector"], "industry": ind,
                                "theme": theme or "", "confidence": "UNMAPPED",
                                "reason": "no industry mapping" if theme is None else f"target '{theme}' not canonical"})
            continue
        reason = ""
        if theme == "enterprise_software":
            refined = refine_software(r.get("longBusinessSummary"))
            if refined and refined in canon_set:
                reason = f"summary-refined from enterprise_software (industry={ind})"
                theme, conf = refined, "M"
        review_rows.append({"ticker": tk, "sector": r["sector"], "industry": ind,
                            "theme": theme, "confidence": conf, "reason": reason})
        add_rows.append({"ticker": tk, "asset_type": "stock",
                         "theme_1": theme, "theme_2": "", "theme_3": ""})

    add_df = pd.DataFrame(add_rows)
    # align columns with existing map
    for c in out_map.columns:
        if c not in add_df.columns:
            add_df[c] = ""
    add_df = add_df[out_map.columns]
    final_map = pd.concat([out_map, add_df], ignore_index=True).drop_duplicates("ticker", keep="first")
    final_map["ticker"] = final_map["ticker"].str.upper()

    # ---- populate new alpha themes by reassigning existing tickers ----
    def _slots(row) -> list[str]:
        return [str(row[c]).strip() for c in theme_cols
                if str(row[c]).strip() and str(row[c]).strip().lower() != "nan"]

    def set_primary(tk: str, theme: str) -> None:
        idx = final_map.index[final_map["ticker"] == tk]
        if len(idx) == 0:  # absent -> add as new row
            new = {c: "" for c in final_map.columns}
            new.update({"ticker": tk, "asset_type": "stock", "theme_1": theme})
            final_map.loc[len(final_map)] = new
            return
        i = idx[0]
        prev = [t for t in _slots(final_map.loc[i]) if t != theme]
        ordered = [theme] + prev
        for j, c in enumerate(theme_cols):
            final_map.at[i, c] = ordered[j] if j < len(ordered) else ""

    def add_secondary(tk: str, theme: str) -> None:
        idx = final_map.index[final_map["ticker"] == tk]
        if len(idx) == 0:
            set_primary(tk, theme)
            return
        i = idx[0]
        cur = _slots(final_map.loc[i])
        if theme in cur:
            return
        for c in theme_cols:
            if not (str(final_map.at[i, c]).strip() and str(final_map.at[i, c]).strip().lower() != "nan"):
                final_map.at[i, c] = theme
                return
        final_map.at[i, theme_cols[-1]] = theme  # no free slot -> overwrite last

    for tk, th in NEW_THEME_PRIMARY.items():
        set_primary(tk, th)
    for tk, th in NEW_THEME_SECONDARY.items():
        add_secondary(tk, th)

    review = pd.DataFrame(review_rows)

    # ---- write ----
    new_def.to_csv(DATA / "theme_definition_v4.csv", index=False)
    final_map.to_csv(DATA / "theme_map_v4.csv", index=False)
    review.to_csv(DATA / "taxonomy_v4_assignment_review.csv", index=False)

    # ---- diff report ----
    report.append(f"themes: {len(defs)} -> {len(new_def)} (removed {len(removed)}, added {len(NEW_THEMES)})")
    report.append(f"removed themes: {sorted(removed)}")
    report.append(f"map tickers: {cmap['ticker'].nunique()} -> {final_map['ticker'].nunique()}")
    report.append(f"new equities assigned: {(review['confidence'].isin(['H','M','L'])).sum()}")
    report.append(f"  by confidence: " + review[review['confidence'].isin(['H','M','L'])]['confidence'].value_counts().to_dict().__str__())
    report.append(f"excluded (non-equity): {(review['confidence']=='EXCLUDE').sum()}")
    report.append(f"UNMAPPED (need manual): {(review['confidence']=='UNMAPPED').sum()}")
    unmapped = review[review['confidence'] == 'UNMAPPED']
    if len(unmapped):
        report.append("  unmapped tickers: " + ", ".join(f"{x.ticker}({x.industry})" for x in unmapped.itertuples()))
    report.append("\nnew constituents per theme (top 25):")
    counts = add_df["theme_1"].value_counts().head(25)
    report.append(counts.to_string())
    text = "\n".join(str(x) for x in report)
    (DATA / "taxonomy_v4_diff_report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
