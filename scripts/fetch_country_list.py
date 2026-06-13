"""Fetch the base country list (ISO3, names, capital name, centroid coords)
from the mledoze/countries open dataset (the same dataset restcountries.com
itself is built from - used directly since restcountries.com's public API
versions are currently deprecated).

Caches the raw JSON response so re-runs don't depend on the source being up.
Output: data/interim/countries_base.csv
"""

import json
import os

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")

RAW_JSON_PATH = os.path.join(RAW_DIR, "countries_mledoze.json")
OUT_CSV_PATH = os.path.join(INTERIM_DIR, "countries_base.csv")

SOURCE_URL = "https://raw.githubusercontent.com/mledoze/countries/master/countries.json"


def fetch_raw():
    if os.path.exists(RAW_JSON_PATH):
        print(f"Using cached {RAW_JSON_PATH}")
        with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Fetching {SOURCE_URL}")
    resp = requests.get(SOURCE_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    os.makedirs(RAW_DIR, exist_ok=True)
    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Cached raw response to {RAW_JSON_PATH} ({len(data)} entries)")
    return data


def build_dataframe(data):
    rows = []
    for entry in data:
        iso3 = entry.get("cca3")
        iso2 = entry.get("cca2")
        name = entry.get("name") or {}
        name_common = name.get("common")
        name_official = name.get("official")

        if not iso3 or not name_common:
            continue

        capitals = entry.get("capital") or []
        capital_name = capitals[0] if capitals else None

        country_latlng = entry.get("latlng") or []
        country_lat = country_latlng[0] if len(country_latlng) == 2 else None
        country_lng = country_latlng[1] if len(country_latlng) == 2 else None

        rows.append(
            {
                "iso3": iso3,
                "iso2": iso2,
                "name_common": name_common,
                "name_official": name_official,
                "capital_name": capital_name,
                "country_lat": country_lat,
                "country_lng": country_lng,
                "region": entry.get("region"),
                "subregion": entry.get("subregion"),
                "has_capital": capital_name is not None,
            }
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="iso3").sort_values("name_common").reset_index(drop=True)
    return df


def main():
    os.makedirs(INTERIM_DIR, exist_ok=True)
    data = fetch_raw()
    df = build_dataframe(data)
    df.to_csv(OUT_CSV_PATH, index=False)
    n_with_capital = int(df["has_capital"].sum())
    print(f"Wrote {len(df)} countries to {OUT_CSV_PATH} ({n_with_capital} with a capital name)")


if __name__ == "__main__":
    main()
