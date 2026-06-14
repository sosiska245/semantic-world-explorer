# Semantic World Explorer — Project Knowledge File

> For the next developer. Everything you need to understand, run, and extend this project
> without reading the chat history. Last updated: June 2026.

---

## 1. What the project does

**Semantic World Explorer** is a Plotly Dash dashboard that lets users search 250 countries
by free-text concept rather than by name or filter. A user types something like
_"tropical beach resort culture"_ or _"military spending defence budget"_ and the app ranks
every country by how closely it matches — coloring a world map, a ranking table, and a
bar chart in real time.

Up to three concepts can be active simultaneously (Red / Green / Blue slots). With multiple
slots, the map blends each country's three similarity scores into a mixed color, so
geographic overlap between concepts becomes immediately visible. A Compare tab shows a
polarity scatter plot between any two concept slots.

The app is deployed on Render (free tier) and uses Voyage AI for embeddings.

---

## 2. How the data pipeline works

The offline pipeline runs once to produce the two files the app reads at runtime.
**Never modify `data/processed/` by hand** — always re-run the relevant script.

```
scripts/
  fetch_country_list.py        → data/interim/countries_base.csv       (250 countries, ISO3, lat/lon)
  fetch_structured_stats.py    → data/interim/stats_by_country.csv     (GDP, population, temp, etc.)
  fetch_facts.py               → data/interim/facts_by_country.csv     (15 OWID/WB indicators)
  fetch_domain_data.py         → data/interim/domain_facts.json        (WB API, NATO, nuclear, UNESCO)
  build_profiles.py            → data/interim/profiles.parquet         (Wikipedia text + stats sentences)
  build_embeddings.py          → data/processed/embeddings.parquet     (main + 5 section embeddings)
                                 data/processed/facts_by_country.csv   (copied from interim)
  build_domain_profiles.py     → updates embeddings.parquet            (adds 4 domain embedding columns)
```

### Key decisions in each step

**`fetch_facts.py`** — Downloads 15 structured indicators from Our World in Data (CSV) and
World Bank API (JSON). Stores them in long format: `(iso3, indicator, value, unit, year, source)`.
These facts serve two purposes: (a) evidence blocks in the detail panel, (b) direct ranking
in the Smart Query Router (see §5 Factual routing).

**`build_profiles.py`** — Fetches Wikipedia pages for each country via `wikipedia-api`.
Caches raw content under `data/raw/wiki_cache/<id>.txt` so re-runs skip already-fetched pages.
Assembles profile text: Wikipedia intro (≤2000 chars) + selected Wikipedia sections
(economy, culture, geography, politics, history, religion, climate, etc., each ≤6000 chars)
+ a few structured-stat sentences (population, GDP, life expectancy). Maximum profile length
is 15,000 chars. Also extracts per-section texts (`text_economy`, `text_culture`, etc.)
separately for section-level embedding.

**`build_embeddings.py`** — Embeds profile text using Voyage AI `voyage-4`, `input_type="document"`,
1024 dimensions. Batches of 10, resumable (skips already-embedded IDs). Also embeds the five
section texts per country, adding columns `emb_economy`, `emb_culture`, `emb_geography`,
`emb_politics`, `emb_history`. Copies `facts_by_country.csv` to `data/processed/` so the app
has everything it needs from that single directory.

**`build_domain_profiles.py`** — Adds four domain embedding columns to the existing parquet.
For each country, constructs short structured text for four domains (military, aviation,
technology, tourism/health) using World Bank data + static dicts from `domain_facts.json`.
Embeds them and stores as `emb_domain_{military,aviation,technology,tourism_health}`.

### Re-running after profile changes

```bash
# If Wikipedia text changes:
.venv/bin/python scripts/build_profiles.py    # uses wiki_cache, fast if content unchanged
.venv/bin/python scripts/build_embeddings.py  # re-embeds everything (--force to re-embed all)

# If only domain data changes:
.venv/bin/python scripts/fetch_domain_data.py
.venv/bin/python scripts/build_domain_profiles.py

# If only facts CSV changes (no embedding rebuild needed):
# Edit scripts/fetch_facts.py, then:
.venv/bin/python scripts/fetch_facts.py
# Then manually append/update data/processed/facts_by_country.csv
```

---

## 3. Important files

```
app.py                                    Entry point. Imports all layout + callbacks.
src/config.py                             EMBEDDING_MODEL, EMBEDDING_DIM, SLOT_COLORS
src/data_loader.py                        Loads parquet at startup → EMBEDDINGS, ENTITIES_DF, SECTION_EMBEDDINGS, DOMAIN_EMBEDDINGS
src/similarity.py                         Embeds user queries at runtime (Voyage, cached)
src/multivec.py                           Multi-section similarity + length penalty
src/bm25.py                               BM25 index over profile texts
src/domain_sim.py                         Domain embedding blend
src/factual_router.py                     Smart Query Router: keyword → indicator ranking
src/facts_loader.py                       Loads facts CSV for detail panel evidence blocks
src/callbacks/slots_callbacks.py          Main ranking callback — chooses which scoring path to use
src/callbacks/chart_callbacks.py          Bar chart + polarity scatter
src/callbacks/map_callbacks.py            Choropleth world map
src/callbacks/table_callbacks.py          Ranking table
src/callbacks/detail_callbacks.py         Country detail panel (facts evidence, similar countries)
src/layout/sidebar.py                     Slot inputs, ranking mode radio, info alerts
src/layout/explorer_tab.py               World map + ranking table
src/layout/compare_tab.py               Bar chart + polarity scatter
src/layout/about_tab.py                  Documentation tab (also describes ranking modes)

data/processed/embeddings.parquet         THE core data file — 250 rows × 24 columns
data/processed/facts_by_country.csv       15 indicators, 237 countries, long format
data/raw/wiki_cache/                      Raw Wikipedia JSON per country — do not delete

scripts/run_router_regression.py          Regression test: 12 factual routes + 5 edge cases + 20 semantic
scripts/run_final_audit.py               139-query quality audit across all modes
render.yaml                               Render.com deployment config
requirements.txt                          Python dependencies (no conda, pure pip)
```

---

## 4. Embeddings in `embeddings.parquet`

The parquet has 250 rows (one per country) and 24 columns. The embedding columns:

| Column | Dim | What it is | Built by |
|---|---|---|---|
| `embedding` | 1024 | Main embedding of full profile text (intro + all sections + stats). This is the primary similarity vector used in all modes. | `build_embeddings.py` |
| `emb_economy` | 1024 | Embedding of Wikipedia economy/infrastructure section text only (≤6000 chars). | `build_embeddings.py` |
| `emb_culture` | 1024 | Embedding of Wikipedia culture/religion/cuisine sections. | `build_embeddings.py` |
| `emb_geography` | 1024 | Embedding of Wikipedia geography/climate/environment sections. | `build_embeddings.py` |
| `emb_politics` | 1024 | Embedding of Wikipedia government/politics sections. | `build_embeddings.py` |
| `emb_history` | 1024 | Embedding of Wikipedia history section. | `build_embeddings.py` |
| `emb_domain_military` | 1024 | Embedding of structured military text: personnel count, NATO membership, nuclear status, military spending % GDP. 172 countries non-empty before domain rebuild, now 250. | `build_domain_profiles.py` |
| `emb_domain_aviation` | 1024 | Embedding of structured aviation text: air passengers per 1000 people (WB + PAX overrides), LPI logistics index, strategic geography notes. | `build_domain_profiles.py` |
| `emb_domain_technology` | 1024 | Embedding of structured technology text: high-tech exports % (WB), R&D spending % GDP (WB), internet users. Taiwan supplemented manually. | `build_domain_profiles.py` |
| `emb_domain_tourism_health` | 1024 | Embedding of structured tourism/health text: tourist arrivals, UNESCO site count, life expectancy, health expenditure % GDP. | `build_domain_profiles.py` |

All embeddings use Voyage AI `voyage-4`, `input_type="document"`, 1024 dimensions.
All vectors are pre-normalized (L2 norm ≈ 1), so dot product = cosine similarity.

**At runtime the app never re-embeds profile texts** — only short user queries are embedded
(~0.1 sec per query, cached in `src/cache.py`).

---

## 5. How each search mode works

### Auto (default)

The default mode. Checks the Smart Query Router first; if a factual route matches, uses
direct indicator ranking. Otherwise uses the Semantic path, with one exception: if the
query contains crop/commodity keywords (wheat, cocoa, viticulture, plantation, etc.), it
automatically uses Hybrid instead of pure Semantic — because BM25 keyword matching
outperforms cosine similarity for specific commodity names.

```
Auto path decision:
  detect_route(query) → factual route?  →  Factual ranking (indicator scores)
  no route + crop keywords?             →  Hybrid (cosine + 8% BM25)
  no route + no crop keywords           →  Semantic (cosine only)
```

### Semantic

Pure embedding similarity. Computes cosine similarity between query vector and each
country's section-blended score:

```
score = (1 - alpha) * main_cosine + alpha * best_section_cosine
```

where `best_section_cosine` is the max cosine across the 5 section embeddings (economy,
culture, geography, politics, history). `alpha = 0.10` for energy queries (to prevent
the energy section from dominating), `0.30` otherwise.

After scoring, a **length penalty** is applied:
```python
penalty = min(1.0, profile_len / 7000)
```
Countries with short profiles (< 7000 chars) have their scores multiplied by this fraction.
The top-5 raw-score countries are exempt, so genuinely top-ranked small islands still surface.
This suppresses small-territory "hub" countries that score moderately for everything due to
generic profiles.

### Hybrid (+ BM25)

Starts with the Semantic score, then blends in a BM25 keyword-overlap score:

```python
final = 0.92 * cosine_score + 0.08 * bm25_normalized_score
```

BM25 is built from all 250 profile texts at startup. Helps when the query contains
specific terms (crop names, country-specific words) that appear verbatim in profiles.
Can add noise for abstract conceptual queries — use when you have precise terminology.

### Domain (experimental)

Blends the main embedding with the best-matching domain embedding:

```python
score = (1 - 0.4) * main_cosine + 0.4 * best_domain_cosine
```

where `best_domain_cosine` is the max cosine across the 4 domain embeddings (military,
aviation, technology, tourism_health). Countries with no domain data use pure main score.

Domain mode is best for structured factual queries about alliance membership, aviation
hubs, semiconductor industry, or tourism infrastructure. It underperforms on numeric
magnitude queries because cosine similarity cannot distinguish "has $10B military budget"
from "has $800B military budget" — they both produce vectors about "military spending."

### Factual routing

Triggered automatically in Auto mode when the query matches a keyword phrase. Bypasses
embeddings entirely — ranks by a structured numeric indicator from `facts_by_country.csv`.

**Active routes (7 total):**

| Trigger phrases | Indicator | Countries | Hits@20 |
|---|---|---|---|
| "life expectancy", "longevity", "lifespan" | `life_expectancy` (UN/OWID) | 236 | 8/10 |
| "tourist arrivals", "most visited", "most tourists" | `tourist_arrivals` (UNWTO/OWID) | 206 | 8/10 |
| "electoral democracy", "political freedom", "most democratic" | `democracy_index` (V-Dem/OWID) | 176 | 9/10 |
| "internet users", "internet access", "digital society" | `internet_users_share` (ITU/OWID) | 212 | 7/10 |
| "renewable energy share", "most renewable", "renewable share" | `renewable_share` (OWID/EI) | 79 | 9/10 |
| "agriculture share", "agriculture dominates", "most agricultural" | `agriculture_share_gdp` (WB/OWID) | 204 | 7/10 |
| "military spending", "defence budget", "biggest military spender" | `military_spending_usd` (WB 2024) | 150 | 10/10 |

**Intentionally NOT routed:**
- Military spending % GDP (`military_expenditure_gdp`) — routes to % GDP, ranks Ukraine #1
  in wartime, Eritrea #2. Absolute USD (`military_spending_usd`) is used instead.
- Healthcare (`health_expenditure_gdp`) — Tuvalu and Nauru rank #1 due to foreign aid
  accounting artifacts in the WHO data. No reliable composite healthcare indicator available.
- Medical tourism — no structured ranking data from open sources.

---

## 6. Experiments tried

### Phase 1: Baseline semantic search

Simple cosine similarity of query vector against `embedding` (full profile text). No section
boost, no BM25, no penalty. Benchmarked at ~22% hits@20 on a 125-query set.

**Result:** Many small-territory hub countries (Saint Martin 665 chars, Anguilla, etc.) appeared
in top-20 for unrelated queries because their generic short profiles weakly matched everything.

### Phase 2: Multi-section similarity + length penalty + BM25

- Added section-level embeddings (economy/culture/geography/politics/history) and blended them
  as `(1-alpha) * main + alpha * best_section`.
- Added length penalty `min(1.0, profile_len/7000)` to suppress short-profile hubs.
- Added BM25 hybrid blend at alpha=0.08 (8% BM25).
- Added energy query alpha override (0.10 instead of 0.30) to prevent energy section from
  dominating all energy queries.
- Raised section text cap from 3000 → 6000 chars to reduce truncation on large countries.
- Added OWID structured indicators (14 total) to `facts_by_country.csv`.

**Result:** Full benchmark 43.5%. Hub appearances dropped from ~75 to 24 across 139-query set.
BM25 hybrid improved agriculture/food queries but added noise on abstract queries.

### Phase 3: Domain knowledge embeddings

Built 4 domain-specific embedding columns from structured World Bank data, NATO membership,
nuclear status, UNESCO counts, aviation passenger rates, high-tech exports, R&D spending.

- Domain mode blend (60/40 main/domain) at alpha=0.4.
- Visible as an optional "Domain (exp.)" mode in the UI.
- NATO membership, aviation hub density, semiconductor industry improved.

**Result:** Domain mode tied or slightly underperformed Hybrid on the full 139-query benchmark
(43.0% vs 43.8%). Strong gains for specific institutional queries, regressions on technology
category (hitech exports confused Thailand/Armenia with pure tech leaders). Kept as experimental
mode — not default.

### Phase 4: Smart Query Router

For queries that are fundamentally numeric/factual ranking problems, embedding similarity
cannot work well (it measures textual closeness, not numeric magnitude). Added keyword-triggered
routing to bypass embeddings entirely for 6 (later 7) indicator categories.

**Result:** Factual routes: 83% hits@20 vs 38% semantic on the same queries. Auto mode
overall: 45.9% vs 43.5% semantic baseline (+2.4pp).

---

## 7. What worked and what did not

### Worked well
- **Length penalty** — most impactful single change; eliminated most small-territory hub contamination
- **Section embeddings** — economy section helps financial hub queries, geography helps island/mountain queries
- **Factual routing** — 10/10 for military spending, 9/10 for democracy and renewable energy; far better than any embedding approach for direct numeric ranking queries
- **Energy alpha override** — fixed the energy category underperforming by reducing section weight
- **BM25 auto-hybrid for crop terms** — wheat/spice/tea queries gain 1-3 hits@20 from keyword matching

### Did not work / abandoned
- **Healthcare routing** — `health_expenditure_gdp` is distorted by foreign aid; Tuvalu and Nauru rank #1. No better open-source indicator available.
- **Military % GDP routing** — Ukraine at 39% GDP ranks #1 in wartime; not what users mean by "military power." Use absolute USD (World Bank `MS.MIL.XPND.CD`) instead.
- **Domain mode as default** — overall benchmark is slightly worse than Hybrid due to high-tech export proxy confusing Thailand/Armenia with semiconductor leaders. Kept as optional.
- **Fashion, anime, chess, arms exports** — semantic 0/10 across all modes. Wikipedia profiles don't state these rankings clearly enough for embedding similarity. Would need SIPRI, BOF fashion rankings, or similar structured sources.
- **Medical tourism** — 0/10 all modes. Wikipedia discusses healthcare and tourism separately; "medical tourism" as a concept is absent. Small Caribbean territories dominate the health + tourism signal.
- **Olympic medals** — 0/10. Medal counts are not in Wikipedia profile prose at the density needed.

---

## 8. Current benchmark results

**139 diverse queries, 1382 expected countries in top-20:**

| Mode | Hits@20 | % |
|---|---|---|
| Auto (current default) | 635/1382 | **45.9%** |
| Semantic only | 601/1382 | 43.5% |
| Hybrid (+ BM25) | 605/1382 | 43.8% |
| Domain (experimental) | 594/1382 | 43.0% |
| Oracle ceiling (best mode per query) | 724/1382 | **52.4%** |

**Auto mode breakdown:**
- 96% of queries use the semantic path (with hybrid for crop keywords)
- 4% (6 queries) use factual routing at 80%+ accuracy

**Factual router regression: 12 routes, ALL CHECKS PASSED**
- Factual hits@20: 83% (100/120)
- vs Semantic on same queries: 38% (45/120)

**Failure breakdown (748 missed expected countries):**
- SEMANTIC_MISS 55.9% — concept not encoded in any Wikipedia profile
- BROAD_QUERY 42.5% — scores too compressed to discriminate (query describes too many countries equally)
- INDICATOR_PROXY 1.6% — factual route fires but indicator is an imperfect proxy

**Category performance (Auto mode):**

| Category | Auto | Oracle gap |
|---|---|---|
| Culture (fashion, anime, jazz) | 19% | +6 hits |
| Aviation | 28% | +2 |
| Tourism | 32% | +3 |
| Military | 32% | +4 |
| Energy | 34% | +8 |
| Agriculture | 36% | +12 |
| Sports | 34% | +5 |
| Oil/gas | 78% | 0 |
| Factual (routed) | **80%** | 0 |
| Mountains, Desert | 70-74% | 0 |

---

## 9. Current best configuration

**Default: Auto mode** (hardcoded as initial store value in `app.py`, radio default in `sidebar.py`).

Key parameters (do not change without re-benchmarking):

```python
# multivec.py
PENALTY_TARGET = 7000.0   # chars at which penalty = 1.0 (no suppression)
PENALTY_TOP_EXEMPT = 5    # top-N raw-score countries exempt from penalty

# slots_callbacks.py
SECTION_ALPHA_DEFAULT = 0.30   # section embedding blend weight
SECTION_ALPHA_ENERGY  = 0.10   # override for energy queries

# bm25.py / slots_callbacks.py
BM25_ALPHA = 0.08   # 8% BM25, 92% cosine

# domain_sim.py
DOMAIN_ALPHA = 0.40   # 40% best domain, 60% main embedding

# factual_router.py — 7 routes (see §5)
```

---

## 10. Known limitations

1. **Semantic ceiling ~52% hits@20.** ~56% of all misses are SEMANTIC_MISS — the concept is simply
   absent from Wikipedia profiles. No amount of parameter tuning can fix this; it requires new data sources.

2. **Numeric magnitude blind.** Embedding similarity cannot distinguish "has $10B budget" from
   "$800B budget." Any query asking for the largest/highest/most in a numeric dimension needs
   a factual route (indicator ranking) — embedding will return countries that *mention* the
   concept, not those that *lead* it.

3. **Small territory residual contamination.** Length penalty reduced hub appearances from 75 to 24
   across the benchmark, but the top-5 exemption means legitimately top-ranking micro-states
   (like the Maldives for tropical islands) still appear alongside surprising small territories
   for niche queries.

4. **Coverage gaps in factual routes.** `military_spending_usd` covers 150 countries; 100 countries
   receive score 0 (ranked last). `renewable_share` covers only 79 countries. Countries with no
   data are invisible to those routes — they fall to the bottom even if they're genuinely relevant.

5. **Wikipedia text freshness.** Profiles are built from Wikipedia at the time `build_profiles.py`
   is run. Events after that point (conflicts, elections, economic shifts) are not reflected.
   Wiki cache under `data/raw/wiki_cache/` should be cleared periodically for a full refresh.

6. **English Wikipedia bias.** Countries with rich English Wikipedia articles (USA, UK, France)
   have much more profile text and more accurate section embeddings than countries with thin
   articles (many Pacific island states, some Central African countries).

7. **Domain mode technology regression.** Adding high-tech exports (WB `TX.VAL.TECH.MF.ZS`) to
   domain profiles caused Thailand and Armenia to rank unexpectedly high for AI/semiconductor
   queries — they have significant electronics assembly classified as high-tech. This is a
   known artifact; domain mode is therefore kept experimental.

---

## 11. Future improvement ideas

Ordered roughly by expected impact / implementation effort:

**High impact, low effort:**
- Add 2-3 more factual routes: `air_passengers` for aviation hub queries, `tourist_arrivals`
  secondary route for "tourism-dependent economy", absolute arms export value (SIPRI TIV
  database if parseable).
- Switch agriculture category to Hybrid by default in Auto mode (currently only crop-keyword
  queries auto-upgrade; agriculture-share queries should also route there).

**Medium impact, medium effort:**
- Add SIPRI TIV arms transfer data as a factual indicator for arms export queries.
- Add UN DPKO peacekeeping contributor data (major contributors: Ethiopia, Bangladesh,
  Pakistan, Rwanda, Ghana) — currently 1-2/10 for peacekeeping queries, zero structured data.
- Medical tourism indicator: could proxy via combining `tourist_arrivals` + `health_expenditure_gdp`
  with a whitelist of known medical tourism hubs.

**High impact, high effort (architectural change):**
- Query decomposition: parse multi-condition queries ("tourism-dependent AND island AND developing")
  into sub-queries, retrieve indicators for each, combine with explicit logic. This would require
  rewriting the scoring layer.
- Retrieval-augmented ranking: for each query, retrieve relevant OWID/WB indicators dynamically,
  then rank. Needs a query → indicator mapping model.
- Real-time Wikipedia profile refresh on demand — fetch latest revision per country via API when
  user requests a specific country's detail, blend with cached embedding.

**Current ceiling:** ~52% hits@20 with perfect mode selection. Architectural changes needed to
go meaningfully above 55%. The SEMANTIC_MISS category (56% of all failures) cannot be addressed
without new structured data sources for the topics that Wikipedia prose doesn't represent well
(fashion rankings, arms exports, Olympic medals, medical tourism, peacekeeping).

---

## 12. Deployment

### Environment variables

```bash
VOYAGE_API_KEY=voy-...    # Required. From https://www.voyageai.com
# No other secrets needed. All data is in data/processed/ (committed to git).
```

### Render.com (current)

Service is configured in `render.yaml`:
- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:server`
- Python 3.11.9, free plan
- Set `VOYAGE_API_KEY` as a secret env var in the Render dashboard

The app reads `VOYAGE_API_KEY` at runtime only when a user types a query — it embeds the
short query string via the Voyage API. All country embeddings are precomputed in the parquet.

### Local development

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in VOYAGE_API_KEY

# Run
python app.py                 # debug server at http://localhost:8050

# Or with gunicorn (production-like):
gunicorn app:server
```

### Running the offline pipeline (full rebuild)

Only needed if you change profile text or add countries:

```bash
# 1. Fetch base data (country list, stats, OWID facts)
python scripts/fetch_country_list.py
python scripts/fetch_structured_stats.py
python scripts/fetch_facts.py

# 2. Build Wikipedia profiles (uses wiki cache, ~5 min with cache warm)
python scripts/build_profiles.py

# 3. Embed profiles (Voyage API, ~10 min, ~$1-2)
python scripts/build_embeddings.py

# 4. Fetch and embed domain data
python scripts/fetch_domain_data.py
python scripts/build_domain_profiles.py

# 5. Verify
python scripts/run_router_regression.py    # all checks should pass
python scripts/run_final_audit.py          # ~5 min, prints quality report
```

### Pre-deployment checklist

Run before any deployment:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
import app  # imports all modules, registers all callbacks — should print nothing and exit cleanly
print('OK')
"
python scripts/run_router_regression.py    # should print ALL CHECKS PASSED
```

Key things to verify manually:
- `data/processed/embeddings.parquet` is 250 rows × 24 columns
- `data/processed/facts_by_country.csv` contains `military_spending_usd` indicator
- `store-ranking-mode` in `app.py` is initialized to `"auto"`
- `ranking-mode-radio` in `sidebar.py` has `value="auto"`
- `VOYAGE_API_KEY` is set in Render dashboard (not committed to git)
