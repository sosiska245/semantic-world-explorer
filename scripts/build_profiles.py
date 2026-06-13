"""Build per-entity profile text (Wikipedia excerpt + structured-stat sentences)
for every country and curated city.

Caches raw Wikipedia content per entity under data/raw/wiki_cache/<id>.txt so
re-runs are resumable and don't re-hit Wikipedia.

Output: data/interim/profiles.parquet with columns:
  id, name, type, iso3, parent_country, lat, lon, profile_text, profile_excerpt
"""

import json
import os
import random
import time

import pandas as pd
import wikipediaapi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim")
WIKI_CACHE_DIR = os.path.join(RAW_DIR, "wiki_cache")

COUNTRIES_CSV = os.path.join(INTERIM_DIR, "countries_base.csv")
CITIES_CSV = os.path.join(INTERIM_DIR, "cities_base.csv")
STATS_CSV = os.path.join(INTERIM_DIR, "stats_by_country.csv")
OUT_PARQUET = os.path.join(INTERIM_DIR, "profiles.parquet")

USER_AGENT = "SemanticWorldExplorer/1.0 (contact: marcogrilocrocodiolo@gmail.com)"
MAX_PROFILE_CHARS = 6000
SUMMARY_TRUNCATE_CHARS = 1000
EXCERPT_CHARS = 400

COUNTRY_SECTION_TARGETS = {"demographics", "religion", "climate", "culture"}
CITY_SECTION_TARGETS = {"demographics", "climate", "culture", "cuisine"}

# Manual overrides for country/city names whose Wikipedia article title differs
# from the restcountries "common name". Expand this dict if a pipeline run
# reports an entity with no wiki text (see WARN logs below).
WIKI_TITLE_OVERRIDES = {
    "Georgia": "Georgia (country)",
    "Congo": "Republic of the Congo",
    "DR Congo": "Democratic Republic of the Congo",
    "Micronesia": "Federated States of Micronesia",
}

wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="en")


def _retry(fn, *args, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - network errors vary
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{retries} after error: {exc} (sleep {wait}s)")
            time.sleep(wait)


def _collect_sections(sections, targets, depth=0, max_depth=2):
    texts = []
    for s in sections:
        title_lower = s.title.strip().lower()
        if title_lower in targets and s.text.strip():
            texts.append(f"== {s.title} ==\n{s.text.strip()}")
        if depth < max_depth:
            texts.extend(_collect_sections(s.sections, targets, depth + 1, max_depth))
    return texts


def fetch_wiki(entity_id, title, alt_title, section_targets):
    """Returns dict {summary, sections_text, source_title} or None. Cached on disk."""
    cache_path = os.path.join(WIKI_CACHE_DIR, f"{entity_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _fetch(t):
        page = wiki.page(t)
        if not page.exists():
            return None
        summary = page.summary.strip()
        sections_text = "\n\n".join(_collect_sections(page.sections, section_targets))
        return {"summary": summary, "sections_text": sections_text, "source_title": t}

    result = _retry(_fetch, title)
    if result is None and alt_title and alt_title != title:
        result = _retry(_fetch, alt_title)

    if result is None:
        print(f"  WARN: no Wikipedia page found for '{title}' (id={entity_id})")
        result = {"summary": "", "sections_text": "", "source_title": None}

    os.makedirs(WIKI_CACHE_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    time.sleep(random.uniform(0.2, 0.5))
    return result


def format_population(pop):
    if pop is None or pd.isna(pop):
        return None
    pop = float(pop)
    if pop >= 1e9:
        return f"{pop / 1e9:.1f} billion"
    if pop >= 1e6:
        return f"{pop / 1e6:.0f} million"
    if pop >= 1e3:
        return f"{pop / 1e3:.0f} thousand"
    return f"{int(pop)}"


def stat_sentences(stats_row):
    if stats_row is None:
        return []

    sentences = []

    avg_temp = stats_row.get("avg_temp_c")
    if pd.notna(avg_temp):
        sentences.append(f"The average annual surface temperature is approximately {avg_temp}°C.")

    gdp = stats_row.get("gdp_per_capita_usd")
    if pd.notna(gdp):
        sentences.append(f"The GDP per capita is approximately ${gdp:,.0f} (current US dollars).")

    pop = stats_row.get("population")
    pop_str = format_population(pop)
    if pop_str:
        sentences.append(f"The population is approximately {pop_str} people.")

    return sentences


def build_profile_text(name, entity_type, wiki_result, stats_row):
    summary = wiki_result["summary"][:SUMMARY_TRUNCATE_CHARS]
    parts = [f"{name} ({entity_type})."]
    if summary:
        parts.append(summary)
    if wiki_result["sections_text"]:
        parts.append(wiki_result["sections_text"])
    sentences = stat_sentences(stats_row)
    if sentences:
        parts.append(" ".join(sentences))

    profile_text = "\n\n".join(parts)
    return profile_text[:MAX_PROFILE_CHARS]


def main():
    countries = pd.read_csv(COUNTRIES_CSV)
    cities = pd.read_csv(CITIES_CSV)
    stats = pd.read_csv(STATS_CSV).set_index("iso3")

    iso3_to_name = dict(zip(countries["iso3"], countries["name_common"]))

    rows = []

    for _, c in countries.iterrows():
        entity_id = f"country_{c['iso3']}"
        lat = c["country_lat"]
        lon = c["country_lng"]
        wiki_title = WIKI_TITLE_OVERRIDES.get(c["name_common"], c["name_common"])

        print(f"[country] {c['name_common']} -> '{wiki_title}'")
        wiki_result = fetch_wiki(entity_id, wiki_title, None, COUNTRY_SECTION_TARGETS)

        stats_row = stats.loc[c["iso3"]].to_dict() if c["iso3"] in stats.index else None
        profile_text = build_profile_text(c["name_common"], "country", wiki_result, stats_row)

        rows.append(
            {
                "id": entity_id,
                "name": c["name_common"],
                "type": "country",
                "iso3": c["iso3"],
                "parent_country": None,
                "lat": lat,
                "lon": lon,
                "profile_text": profile_text,
                "profile_excerpt": (wiki_result["summary"] or profile_text)[:EXCERPT_CHARS],
            }
        )

    for _, ci in cities.iterrows():
        entity_id = ci["city_id"]
        country_name = iso3_to_name.get(ci["country_iso3"])
        wiki_title = WIKI_TITLE_OVERRIDES.get(ci["city_name"], ci["city_name"])
        alt_title = f"{ci['city_name']}, {country_name}" if country_name else None

        print(f"[city] {ci['city_name']} ({country_name}) -> '{wiki_title}' / alt '{alt_title}'")
        wiki_result = fetch_wiki(entity_id, wiki_title, alt_title, CITY_SECTION_TARGETS)

        stats_row = stats.loc[ci["country_iso3"]].to_dict() if ci["country_iso3"] in stats.index else None
        profile_text = build_profile_text(ci["city_name"], "city", wiki_result, stats_row)

        rows.append(
            {
                "id": entity_id,
                "name": ci["city_name"],
                "type": "city",
                "iso3": ci["country_iso3"],
                "parent_country": country_name,
                "lat": ci["lat"],
                "lon": ci["lon"],
                "profile_text": profile_text,
                "profile_excerpt": (wiki_result["summary"] or profile_text)[:EXCERPT_CHARS],
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(INTERIM_DIR, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    n_empty = int((df["profile_excerpt"].str.len() == 0).sum())
    print(f"Wrote {len(df)} profiles to {OUT_PARQUET} ({n_empty} with empty excerpt)")


if __name__ == "__main__":
    main()
