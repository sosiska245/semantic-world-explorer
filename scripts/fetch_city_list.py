"""Build the curated city list: every country's capital (geocoded via the
Open-Meteo Geocoding API, since the country dataset no longer ships capital
coordinates) plus a hand-picked list of extra major non-capital cities
(data/raw/extra_cities.csv).

Geocoding results are cached to data/raw/geocode_cache.json (resumable).

Output: data/interim/cities_base.csv with columns:
  city_id, city_name, country_iso3, lat, lon, is_capital
"""

import json
import os
import time

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")

COUNTRIES_CSV = os.path.join(INTERIM_DIR, "countries_base.csv")
EXTRA_CITIES_CSV = os.path.join(RAW_DIR, "extra_cities.csv")
GEOCODE_CACHE_PATH = os.path.join(RAW_DIR, "geocode_cache.json")
OUT_CSV_PATH = os.path.join(INTERIM_DIR, "cities_base.csv")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


def load_cache():
    if os.path.exists(GEOCODE_CACHE_PATH):
        with open(GEOCODE_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(GEOCODE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def geocode(capital_name, country_iso2, cache, retries=3):
    key = f"{capital_name}|{country_iso2}"
    if key in cache and cache[key][0] is not None:
        return cache[key]

    for attempt in range(retries):
        try:
            resp = requests.get(
                GEOCODE_URL,
                params={"name": capital_name, "count": 10, "language": "en", "format": "json"},
                timeout=15,
            )
            resp.raise_for_status()
            results = (resp.json() or {}).get("results") or []
            break
        except requests.RequestException as exc:
            if attempt == retries - 1:
                print(f"  geocode failed for '{capital_name}' ({country_iso2}): {exc}")
                results = []
            else:
                time.sleep(2 * (attempt + 1))

    same_country = [r for r in results if r.get("country_code") == country_iso2]
    candidates = same_country or results

    chosen = None
    for r in candidates:
        if r.get("feature_code") == "PPLC":
            chosen = r
            break
    if chosen is None and candidates:
        chosen = candidates[0]

    if chosen:
        latlon = [chosen["latitude"], chosen["longitude"]]
    else:
        latlon = [None, None]

    cache[key] = latlon
    save_cache(cache)
    return latlon


def slugify(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("'", "")
        .replace(".", "")
    )


def main():
    countries = pd.read_csv(COUNTRIES_CSV)
    extras = pd.read_csv(EXTRA_CITIES_CSV)
    cache = load_cache()

    capitals = countries[countries["has_capital"]].copy()
    capital_rows = []
    for _, row in capitals.iterrows():
        lat, lon = geocode(row["capital_name"], row["iso2"], cache)
        if lat is None:
            print(f"  WARN: no geocode result for '{row['capital_name']}' ({row['iso2']}), skipping")
            continue
        capital_rows.append(
            {
                "city_name": row["capital_name"],
                "country_iso3": row["iso3"],
                "lat": lat,
                "lon": lon,
                "is_capital": True,
            }
        )

    capital_df = pd.DataFrame(capital_rows)

    extra_rows = extras.copy()
    extra_rows["is_capital"] = False

    cities = pd.concat([capital_df, extra_rows], ignore_index=True)

    # Drop duplicates: same (country_iso3, city_name) pair -> keep capital flag if either is capital
    cities["city_name_norm"] = cities["city_name"].str.strip().str.lower()
    cities = (
        cities.sort_values("is_capital", ascending=False)
        .drop_duplicates(subset=["country_iso3", "city_name_norm"], keep="first")
        .drop(columns="city_name_norm")
        .reset_index(drop=True)
    )

    cities["city_id"] = (
        "city_" + cities["country_iso3"] + "_" + cities["city_name"].map(slugify)
    )

    cities = cities[["city_id", "city_name", "country_iso3", "lat", "lon", "is_capital"]]
    cities = cities.sort_values(["country_iso3", "city_name"]).reset_index(drop=True)

    os.makedirs(INTERIM_DIR, exist_ok=True)
    cities.to_csv(OUT_CSV_PATH, index=False)
    print(f"Wrote {len(cities)} cities to {OUT_CSV_PATH} ({int(cities['is_capital'].sum())} capitals)")


if __name__ == "__main__":
    main()
