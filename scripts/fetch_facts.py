"""Fetch Our World in Data CSVs for Phase-1 "facts" indicators (energy,
agriculture, politics, geography) and write a long-format facts table.

Caches each raw CSV under data/raw/ (same pattern as
fetch_structured_stats.py). Output: data/interim/facts_by_country.csv with
columns:
  iso3, category, indicator, value, unit, year, source

One row per (iso3, indicator) when data exists for that country/indicator. A
country with no row for a given indicator has no data for it - this is read
downstream (src/facts_loader.py) as "missing", never inferred or defaulted.
"""

import os
import re

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")

OUT_CSV_PATH = os.path.join(INTERIM_DIR, "facts_by_country.csv")

# Some OWID datasets (e.g. population density) extend into UN projection
# years past "now" - cap at this year so we report the latest *actual*
# estimate, not a future projection.
MAX_YEAR = 2025

SOURCES = {
    "renewable_energy": "https://ourworldindata.org/grapher/renewable-share-energy.csv?v=1&csvType=full",
    "agriculture_gdp": "https://ourworldindata.org/grapher/agriculture-share-gdp.csv?v=1&csvType=full",
    "agriculture_employment": "https://ourworldindata.org/grapher/share-of-the-labor-force-employed-in-agriculture.csv?v=1&csvType=full",
    "democracy_index": "https://ourworldindata.org/grapher/electoral-democracy-index.csv?v=1&csvType=full",
    "political_regime": "https://ourworldindata.org/grapher/political-regime.csv?v=1&csvType=full",
    "population_density": "https://ourworldindata.org/grapher/population-density.csv?v=1&csvType=full",
    "land_area": "https://ourworldindata.org/grapher/land-area-km.csv?v=1&csvType=full",
}

ISO3_RE = re.compile(r"^[A-Z]{3}$")

# V-Dem "Regimes of the World" classification used by OWID's
# political-regime grapher.
REGIME_LABELS = {
    0: "closed autocracy",
    1: "electoral autocracy",
    2: "electoral democracy",
    3: "liberal democracy",
}


def load_owid_csv(name, url):
    cache_path = os.path.join(RAW_DIR, f"owid_{name}.csv")
    if os.path.exists(cache_path):
        print(f"Using cached {cache_path}")
        return pd.read_csv(cache_path)

    print(f"Fetching {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    os.makedirs(RAW_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(resp.content)
    return pd.read_csv(cache_path)


def latest_per_country(df, max_year=MAX_YEAR):
    """Keep one row per valid ISO3 country code: the most recent Year <= max_year."""
    df = df[df["Code"].astype(str).str.match(ISO3_RE, na=False)].copy()
    df = df[df["Year"] <= max_year]
    return df.sort_values("Year").groupby("Code", as_index=False).tail(1).set_index("Code")


def find_value_column(df):
    """First column that isn't Entity/Code/Year/an OWID region-grouping column."""
    for col in df.columns:
        if col in ("Entity", "Code", "Year") or col.startswith("World region"):
            continue
        return col
    return None


def numeric_fact_rows(df, category, indicator, unit, source, round_digits=1):
    latest = latest_per_country(df)
    col = find_value_column(latest)
    rows = []
    for iso3, row in latest.iterrows():
        val = row[col]
        if pd.isna(val):
            continue
        rows.append(
            {
                "iso3": iso3,
                "category": category,
                "indicator": indicator,
                "value": round(float(val), round_digits),
                "unit": unit,
                "year": int(row["Year"]),
                "source": source,
            }
        )
    return rows


def regime_fact_rows(df):
    latest = latest_per_country(df)
    col = find_value_column(latest)
    rows = []
    for iso3, row in latest.iterrows():
        code = row[col]
        if pd.isna(code):
            continue
        rows.append(
            {
                "iso3": iso3,
                "category": "politics",
                "indicator": "regime_type",
                "value": REGIME_LABELS.get(int(code), "unknown"),
                "unit": "",
                "year": int(row["Year"]),
                "source": "V-Dem via Our World in Data",
            }
        )
    return rows


def main():
    os.makedirs(INTERIM_DIR, exist_ok=True)
    rows = []

    df = load_owid_csv("renewable_energy", SOURCES["renewable_energy"])
    rows += numeric_fact_rows(df, "energy", "renewable_share", "%", "Our World in Data / Energy Institute", round_digits=1)

    df = load_owid_csv("agriculture_gdp", SOURCES["agriculture_gdp"])
    rows += numeric_fact_rows(df, "agriculture", "agriculture_share_gdp", "% of GDP", "World Bank via Our World in Data", round_digits=1)

    df = load_owid_csv("agriculture_employment", SOURCES["agriculture_employment"])
    rows += numeric_fact_rows(
        df, "agriculture", "agriculture_employment_share", "% of workforce", "World Bank/ILO via Our World in Data", round_digits=1
    )

    df = load_owid_csv("democracy_index", SOURCES["democracy_index"])
    rows += numeric_fact_rows(df, "politics", "democracy_index", "(0-1 scale)", "V-Dem via Our World in Data", round_digits=2)

    df = load_owid_csv("political_regime", SOURCES["political_regime"])
    rows += regime_fact_rows(df)

    df = load_owid_csv("population_density", SOURCES["population_density"])
    rows += numeric_fact_rows(df, "geography", "population_density", "people/km²", "UN via Our World in Data", round_digits=1)

    df = load_owid_csv("land_area", SOURCES["land_area"])
    rows += numeric_fact_rows(df, "geography", "land_area_km2", "km²", "World Bank via Our World in Data", round_digits=0)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV_PATH, index=False)

    n_countries = out["iso3"].nunique()
    print(f"Wrote {len(out)} fact rows for {n_countries} countries to {OUT_CSV_PATH}")
    for cat in sorted(out["category"].unique()):
        sub = out[out["category"] == cat]
        for ind in sorted(sub["indicator"].unique()):
            n = sub[sub["indicator"] == ind]["iso3"].nunique()
            print(f"  {cat}/{ind}: {n} countries")


if __name__ == "__main__":
    main()
