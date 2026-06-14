"""Build per-entity profile text (Wikipedia excerpt + structured-stat sentences)
for every country.

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
STATS_CSV = os.path.join(INTERIM_DIR, "stats_by_country.csv")
FACTS_CSV = os.path.join(INTERIM_DIR, "facts_by_country.csv")
OUT_PARQUET = os.path.join(INTERIM_DIR, "profiles.parquet")

USER_AGENT = "SemanticWorldExplorer/1.0 (contact: marcogrilocrocodiolo@gmail.com)"
MAX_PROFILE_CHARS = 15000
SUMMARY_TRUNCATE_CHARS = 2000
EXCERPT_CHARS = 400
CATEGORY_CHAR_CAP = 600

# Defines both (a) which Wikipedia sections get pulled into a profile - any
# section whose lowercased title CONTAINS one of these keywords as a
# substring (e.g. "Economy and infrastructure", "Government and politics",
# "Science and technology" all match) - and (b) the order those sections
# appear in the assembled profile text. When a title matches more than one
# keyword, the earlier keyword in this list wins (so "Economy and
# infrastructure" -> "economy", "Government and politics" -> "government").
COUNTRY_SECTION_ORDER = [
    "economy",
    "agriculture",
    "culture",
    "religion",
    "climate",
    "energy",
    "geography",
    "government",
    "politics",
    "military",
    "education",
    "tourism",
    "history",
    "demographics",
    "cuisine",
    "sports",
    "infrastructure",
    "science",
    "technology",
    "transport",
]

MACRO_SECTION_MAP = {
    "economy":   ["economy", "agriculture", "energy", "infrastructure", "transport", "science", "technology"],
    "culture":   ["culture", "religion", "cuisine", "demographics", "sports", "tourism", "education"],
    "geography": ["geography", "climate"],
    "politics":  ["government", "politics", "military"],
    "history":   ["history"],
}
MACRO_SECTION_CHAR_CAP = 6000
# Bump when the cache format changes (e.g. cap increase) to invalidate old entries.
CACHE_VERSION = 2
# Raw sections stored in cache up to this limit, so future cap raises don't require re-fetching.
_RAW_SECTION_STORE_CAP = 15000

# Manual overrides for country names whose Wikipedia article title differs
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


def _section_category(title_lower, order):
    """Return the first keyword in `order` that is a substring of
    `title_lower`, or None if no keyword matches."""
    for keyword in order:
        if keyword in title_lower:
            return keyword
    return None


def _collect_sections(sections, order, depth=0, max_depth=2):
    """Group section text by category (first matching keyword in `order`),
    recursing into subsections up to max_depth. Returns dict category ->
    list of '== Title ==\\ntext' blocks, in traversal order within each
    category."""
    grouped = {}
    for s in sections:
        title_lower = s.title.strip().lower()
        category = _section_category(title_lower, order)
        if category and s.text.strip():
            grouped.setdefault(category, []).append(f"== {s.title} ==\n{s.text.strip()}")
        if depth < max_depth:
            for cat, texts in _collect_sections(s.sections, order, depth + 1, max_depth).items():
                grouped.setdefault(cat, []).extend(texts)
    return grouped


def fetch_wiki(entity_id, title, alt_title, section_order):
    """Returns dict {summary, sections_text, source_title, macro_sections}. Cached on disk.

    Cache v2 stores raw (uncapped) section text so that future MACRO_SECTION_CHAR_CAP
    changes apply without re-fetching Wikipedia.
    """
    cache_path = os.path.join(WIKI_CACHE_DIR, f"{entity_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("cache_version", 1) >= CACHE_VERSION and "macro_sections_raw" in cached:
            result = dict(cached)
            result["macro_sections"] = {
                k: v[:MACRO_SECTION_CHAR_CAP]
                for k, v in cached["macro_sections_raw"].items()
            }
            return result
        os.remove(cache_path)  # old format or old version: delete and re-fetch

    def _fetch(t):
        page = wiki.page(t)
        if not page.exists():
            return None
        summary = page.summary.strip()
        grouped = _collect_sections(page.sections, section_order)

        ordered_texts = []
        for category in section_order:
            blocks = grouped.get(category)
            if not blocks:
                continue
            ordered_texts.append("\n\n".join(blocks)[:CATEGORY_CHAR_CAP])
        sections_text = "\n\n".join(ordered_texts)

        macro_sections_raw = {}
        for macro_key, keywords in MACRO_SECTION_MAP.items():
            parts = []
            for kw in keywords:
                parts.extend(grouped.get(kw, []))
            if parts:
                macro_sections_raw[macro_key] = "\n\n".join(parts)[:_RAW_SECTION_STORE_CAP]

        macro_sections = {k: v[:MACRO_SECTION_CHAR_CAP] for k, v in macro_sections_raw.items()}

        return {
            "summary": summary,
            "sections_text": sections_text,
            "source_title": t,
            "macro_sections": macro_sections,
            "macro_sections_raw": macro_sections_raw,
            "cache_version": CACHE_VERSION,
        }

    result = _retry(_fetch, title)
    if result is None and alt_title and alt_title != title:
        result = _retry(_fetch, alt_title)

    if result is None:
        print(f"  WARN: no Wikipedia page found for '{title}' (id={entity_id})")
        result = {
            "summary": "", "sections_text": "", "source_title": None,
            "macro_sections": {}, "macro_sections_raw": {}, "cache_version": CACHE_VERSION,
        }

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


# Per-indicator sentence templates for data/interim/facts_by_country.csv rows.
# These render as plain sentences in the profile text (feeding the embedding,
# placed early so they survive MAX_PROFILE_CHARS truncation) and the same
# rows back the detail panel's "Evidence" block via src/facts_loader.py.
FACT_TEMPLATES = {
    "renewable_share": lambda name, v, year: f"Approximately {v:g}% of {name}'s primary energy consumption comes from renewable sources ({year}).",
    "agriculture_share_gdp": lambda name, v, year: f"Agriculture accounts for approximately {v:g}% of {name}'s GDP ({year}).",
    "agriculture_employment_share": lambda name, v, year: f"Approximately {v:g}% of {name}'s workforce is employed in agriculture ({year}).",
    "democracy_index": lambda name, v, year: f"{name}'s V-Dem electoral democracy index is approximately {v:.2f} on a 0-1 scale ({year}).",
    "regime_type": lambda name, v, year: f"{name} is classified as {'an' if v[0].lower() in 'aeiou' else 'a'} {v} ({year}).",
    "population_density": lambda name, v, year: f"{name} has a population density of approximately {v:g} people per square kilometer ({year}).",
    "land_area_km2": lambda name, v, year: f"{name} has a land area of approximately {v:,.0f} square kilometers ({year}).",
    "life_expectancy":        lambda name, v, year: f"Life expectancy in {name} is approximately {v:g} years ({year}).",
    "health_expenditure_gdp": lambda name, v, year: f"{name} spends approximately {v:g}% of its GDP on healthcare ({year}).",
    "internet_users_share":   lambda name, v, year: f"Approximately {v:g}% of {name}'s population uses the internet ({year}).",
    "mean_years_schooling":   lambda name, v, year: f"The average person in {name} has approximately {v:g} years of schooling ({year}).",
    "tourist_arrivals":       lambda name, v, year: f"{name} receives approximately {v:,.0f} international tourist arrivals per year ({year}).",
    "air_passengers":         lambda name, v, year: f"Air transport in {name}: approximately {v/1e6:.1f} million passengers ({year}).",
    "military_expenditure_gdp": lambda name, v, year: f"{name} spent approximately {v:.1f}% of its GDP on military expenditure ({year}).",
}


def fact_sentences(name, facts):
    sentences = []
    for f in facts:
        template = FACT_TEMPLATES.get(f["indicator"])
        if template is None:
            continue
        value = f["value"] if f["indicator"] == "regime_type" else float(f["value"])
        sentences.append(template(name, value, int(f["year"])))
    return sentences


def build_profile_text(name, entity_type, wiki_result, stats_row, facts=None):
    summary = wiki_result["summary"][:SUMMARY_TRUNCATE_CHARS]
    parts = [f"{name} ({entity_type})."]
    sentences = stat_sentences(stats_row)
    if sentences:
        parts.append(" ".join(sentences))
    fact_sents = fact_sentences(name, facts or [])
    if fact_sents:
        parts.append(" ".join(fact_sents))
    if summary:
        parts.append(summary)
    if wiki_result["sections_text"]:
        parts.append(wiki_result["sections_text"])

    profile_text = "\n\n".join(parts)
    return profile_text[:MAX_PROFILE_CHARS]


def main():
    countries = pd.read_csv(COUNTRIES_CSV)
    stats = pd.read_csv(STATS_CSV).set_index("iso3")

    facts_df = pd.read_csv(FACTS_CSV)
    facts_by_iso3 = {iso3: g.to_dict("records") for iso3, g in facts_df.groupby("iso3")}

    rows = []

    for _, c in countries.iterrows():
        entity_id = f"country_{c['iso3']}"
        lat = c["country_lat"]
        lon = c["country_lng"]
        wiki_title = WIKI_TITLE_OVERRIDES.get(c["name_common"], c["name_common"])

        print(f"[country] {c['name_common']} -> '{wiki_title}'")
        wiki_result = fetch_wiki(entity_id, wiki_title, None, COUNTRY_SECTION_ORDER)

        stats_row = stats.loc[c["iso3"]].to_dict() if c["iso3"] in stats.index else None
        facts = facts_by_iso3.get(c["iso3"], [])
        profile_text = build_profile_text(c["name_common"], "country", wiki_result, stats_row, facts)

        macro = wiki_result.get("macro_sections", {})
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
                "text_economy":   macro.get("economy", ""),
                "text_culture":   macro.get("culture", ""),
                "text_geography": macro.get("geography", ""),
                "text_politics":  macro.get("politics", ""),
                "text_history":   macro.get("history", ""),
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(INTERIM_DIR, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    n_empty = int((df["profile_excerpt"].str.len() == 0).sum())
    print(f"Wrote {len(df)} profiles to {OUT_PARQUET} ({n_empty} with empty excerpt)")


if __name__ == "__main__":
    main()
