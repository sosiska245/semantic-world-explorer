"""Fetch domain-specific data for domain knowledge embeddings.

Sources:
  - World Bank API (military personnel, LPI, R&D, high-tech exports)
  - Static dicts (NATO membership, nuclear weapon states, UNESCO site counts)
  - Existing OWID facts from data/processed/facts_by_country.csv

Output: data/interim/domain_facts.json
"""

import json
import os
import sys
import time

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

OUT_JSON = os.path.join(ROOT_DIR, "data", "interim", "domain_facts.json")

NATO_MEMBERS = {
    "ALB", "BEL", "BGR", "CAN", "HRV", "CZE", "DNK", "EST", "FIN",
    "FRA", "DEU", "GRC", "HUN", "ISL", "ITA", "LVA", "LTU", "LUX",
    "MNE", "NLD", "MKD", "NOR", "POL", "PRT", "ROU", "SVK", "SVN",
    "ESP", "SWE", "TUR", "GBR", "USA",
}

# Declared or confirmed nuclear weapon states
NUCLEAR_STATES = {
    "USA": "declared nuclear power",
    "RUS": "declared nuclear power",
    "CHN": "declared nuclear power",
    "GBR": "declared nuclear power",
    "FRA": "declared nuclear power",
    "IND": "declared nuclear power",
    "PAK": "declared nuclear power",
    "PRK": "declared nuclear power",
    "ISR": "undeclared nuclear state (presumed)",
}

# NATO nuclear sharing: non-nuclear NATO members that host US weapons
NATO_NUCLEAR_SHARING = {"BEL", "DEU", "ITA", "NLD", "TUR"}

# UNESCO World Heritage Site counts per country (ISO3), as of 2024.
# Countries not listed default to 0.
UNESCO_SITES = {
    "ITA": 59, "CHN": 57, "DEU": 52, "FRA": 52, "ESP": 50, "IND": 42,
    "MEX": 35, "GBR": 33, "RUS": 31, "IRN": 27, "JPN": 25, "USA": 24,
    "BRA": 23, "CAN": 22, "TUR": 21, "AUS": 20, "GRC": 19, "KOR": 16,
    "CZE": 17, "POL": 17, "PRT": 17, "BEL": 15, "SWE": 15, "NLD": 13,
    "CHE": 13, "PER": 13, "AUT": 12, "ARG": 12, "IDN": 10, "ZAF": 10,
    "HRV": 10, "BGR": 10, "DNK": 10, "COL": 9, "ETH": 9, "MAR": 9,
    "ISR": 9, "ROU": 9, "CUB": 9, "UKR": 8, "VNM": 8, "LKA": 8,
    "TUN": 8, "HUN": 8, "FIN": 7, "SVK": 7, "EGY": 7, "SAU": 7,
    "SEN": 7, "TZA": 7, "KEN": 7, "BOL": 6, "IRQ": 6, "SYR": 6,
    "JOR": 6, "LBN": 6, "PAK": 6, "PHL": 6, "THA": 6, "MNG": 6,
    "ECU": 5, "KAZ": 5, "OMN": 5, "UZB": 5, "ZWE": 5, "GEO": 4,
    "MNE": 4, "SRB": 4, "SVN": 4, "NPL": 4, "CRI": 4, "ISL": 3,
    "VNM": 8, "ARM": 3, "AZE": 3, "KGZ": 3, "TKM": 3, "MDG": 3,
    "VEN": 3, "GTM": 3, "BGD": 3, "CIV": 3, "BFA": 3, "NER": 3,
    "BHR": 3, "UGA": 3, "LVA": 4, "LTU": 4, "MDA": 1, "BLR": 4,
    "EST": 2, "BTN": 2, "AFG": 2, "ALB": 4, "BIH": 2, "MKD": 1,
    "GHA": 2, "NGA": 2, "CMR": 2, "BWA": 2, "NAM": 2, "TJK": 2,
    "PRK": 2, "PNG": 1, "VUT": 1, "FJI": 1, "NZL": 3, "DOM": 1,
    "HTI": 1, "BLZ": 1, "SLV": 1, "HND": 1, "NIC": 1, "PAN": 2,
    "KHM": 3, "LAO": 3, "MYS": 4, "MMR": 1, "MOZ": 1, "ANG": 1,
    "ZMB": 1, "MWI": 1, "RWA": 1, "SDN": 3, "MLI": 4, "BEN": 2,
    "TCD": 2, "ERI": 1, "ARE": 1, "QAT": 1, "URY": 1, "PRY": 1,
    "CHL": 7, "YEM": 4, "JAM": 1,
}

WB_INDICATORS = {
    "military_personnel": "MS.MIL.TOTL.P1",
    "lpi":                "LP.LPI.OVRL.XQ",
    "rd_pct_gdp":         "GB.XPD.RSDV.GD.ZS",
    "hitech_exports_pct": "TX.VAL.TECH.MF.ZS",
}


def fetch_wb_indicator(indicator_id, mrv=5, per_page=1000):
    """Fetch World Bank indicator for all countries. Returns {iso3: (value, year)}."""
    results = {}
    page = 1
    while True:
        url = (
            f"https://api.worldbank.org/v2/country/all/indicator/{indicator_id}"
            f"?format=json&mrv={mrv}&per_page={per_page}&page={page}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        meta, rows = data[0], data[1] or []

        for row in rows:
            iso3 = row.get("countryiso3code", "")
            val = row.get("value")
            year = row.get("date", "")
            if iso3 and val is not None and iso3 not in results:
                try:
                    results[iso3] = (float(val), int(year))
                except (ValueError, TypeError):
                    pass

        total_pages = meta.get("pages", 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.3)

    return results


def main():
    print("Fetching World Bank indicators...")
    wb = {}
    for name, ind_id in WB_INDICATORS.items():
        print(f"  {name} ({ind_id})...", end=" ", flush=True)
        try:
            wb[name] = fetch_wb_indicator(ind_id)
            print(f"{len(wb[name])} countries")
        except Exception as exc:
            print(f"WARN: {exc}")
            wb[name] = {}
        time.sleep(0.5)

    out = {
        "wb":      wb,
        "nato":    sorted(NATO_MEMBERS),
        "nuclear": NUCLEAR_STATES,
        "nato_nuclear_sharing": sorted(NATO_NUCLEAR_SHARING),
        "unesco":  UNESCO_SITES,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {OUT_JSON}")
    print(f"  NATO: {len(NATO_MEMBERS)} members")
    print(f"  Nuclear: {len(NUCLEAR_STATES)} states")
    print(f"  UNESCO: {len(UNESCO_SITES)} countries with data")
    for name, d in wb.items():
        print(f"  WB/{name}: {len(d)} countries")


if __name__ == "__main__":
    main()
