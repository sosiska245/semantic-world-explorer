"""Download/tidy Our World in Data CSVs (avg temperature, GDP/capita,
population) and join on ISO3 ("Code" column in OWID exports).

Caches each raw CSV under data/raw/ so re-runs don't re-download.
Output: data/interim/stats_by_country.csv with columns:
  iso3, avg_temp_c, population, gdp_per_capita_usd

Note: OWID's per-religion breakdown dataset ("religious-composition") has
been restructured upstream into a single "share religious" figure with no
per-religion columns, so religion stats are no longer included here -
Wikipedia's own "Religion"/"Demographics" sections (see build_profiles.py)
cover this for the embedded profile text instead.
"""

import os
import re

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")

OUT_CSV_PATH = os.path.join(INTERIM_DIR, "stats_by_country.csv")

SOURCES = {
    "temperature": "https://ourworldindata.org/grapher/average-annual-surface-temperature.csv?v=1&csvType=full",
    "gdp": "https://ourworldindata.org/grapher/gdp-per-capita-worldbank.csv?v=1&csvType=full",
    "population": "https://ourworldindata.org/grapher/population.csv?v=1&csvType=full",
}

ISO3_RE = re.compile(r"^[A-Z]{3}$")


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


def latest_per_country(df):
    """Keep one row per valid ISO3 country code: the most recent Year."""
    df = df[df["Code"].astype(str).str.match(ISO3_RE, na=False)].copy()
    df = df.sort_values("Year").groupby("Code", as_index=False).tail(1)
    return df.set_index("Code")


def find_numeric_column(df, keyword, exclude=()):
    """Find a data column (not Entity/Code/Year) whose name contains `keyword`
    (case-insensitive) and does not contain any of the `exclude` substrings."""
    keyword = keyword.lower()
    for col in df.columns:
        if col in ("Entity", "Code", "Year"):
            continue
        col_lower = col.lower()
        if keyword in col_lower and not any(ex in col_lower for ex in exclude):
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    return None


def build_simple_stat(df, keyword, out_col, exclude=(), round_digits=1):
    df = latest_per_country(df)
    col = find_numeric_column(df, keyword, exclude=exclude)
    print(f"{out_col}: using column '{col}'")
    if col is None:
        return pd.DataFrame(columns=[out_col])
    out = df[[col]].rename(columns={col: out_col})
    out[out_col] = out[out_col].round(round_digits)
    return out


def main():
    os.makedirs(INTERIM_DIR, exist_ok=True)

    temp_df = load_owid_csv("avg_temperature", SOURCES["temperature"])
    gdp_df = load_owid_csv("gdp_per_capita", SOURCES["gdp"])
    pop_df = load_owid_csv("population", SOURCES["population"])

    temp_stats = build_simple_stat(temp_df, "temperature", "avg_temp_c", round_digits=1)
    gdp_stats = build_simple_stat(gdp_df, "gdp", "gdp_per_capita_usd", round_digits=0)
    pop_stats = build_simple_stat(
        pop_df, "population", "population", exclude=("density", "growth", "share", "projection"), round_digits=0
    )

    merged = temp_stats.join([gdp_stats, pop_stats], how="outer")
    merged.index.name = "iso3"
    merged = merged.reset_index()

    merged.to_csv(OUT_CSV_PATH, index=False)
    print(f"Wrote {len(merged)} country stat rows to {OUT_CSV_PATH}")


if __name__ == "__main__":
    main()
