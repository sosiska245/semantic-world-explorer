from dash import html


def about_layout():
    return html.Div(
        [
            # ── What is this ────────────────────────────────────────────────────
            html.Div(
                [
                    html.Div("Semantic World Explorer", className="swe-section-title"),
                    html.P(
                        "Type any concept — a cuisine, a sport, a political idea, a cultural "
                        "trait, a natural landmark — and every country on the map is instantly "
                        "ranked and colored by how closely it matches. The ranking is driven by "
                        "meaning, not keyword matching: 'thermal bath culture' finds Hungary "
                        "without Hungary appearing in your query."
                    ),
                    html.P(
                        "Up to three concept slots (Red, Green, Blue) can be active at once. "
                        "With multiple slots, each country's color blends all three scores — "
                        "so a country strong in two concepts shows a mixed color, and you can "
                        "visually spot where concepts overlap or diverge across the world."
                    ),
                ],
                className="swe-card",
            ),

            # ── How to use ──────────────────────────────────────────────────────
            html.Div(
                [
                    html.Div("How to Use", className="swe-section-title"),
                    html.Ol(
                        [
                            html.Li(
                                "Type a concept into Slot 1 and press Enter. "
                                "The map, ranking table, and bar chart update immediately."
                            ),
                            html.Li(
                                "Click '+ Add concept' to activate Slot 2 (Green) and/or Slot 3 (Blue). "
                                "Each slot is scored independently."
                            ),
                            html.Li(
                                "With two or more slots active, open the Compare tab to see a "
                                "polarity scatter plot — countries plotted by how they balance "
                                "two concepts against each other."
                            ),
                            html.Li(
                                "Click any map marker, table row, or scatter point — or use the "
                                "jump-to dropdown — to open the detail panel for that country, "
                                "which includes a profile excerpt, structured facts, and evidence cards."
                            ),
                            html.Li(
                                "Change the Ranking Mode to shift between Auto (default), "
                                "Brand / Associations, Semantic, Hybrid, or Domain — "
                                "each targets a different kind of query."
                            ),
                        ]
                    ),
                    html.P(
                        [
                            html.Strong("Tip: "),
                            "Watch the confidence badge below the search box. "
                            "HIGH (green) means one country scored clearly above the rest. "
                            "LOW (yellow) means the query is broad and many countries scored "
                            "similarly — try adding more specific keywords.",
                        ],
                        style={"fontSize": "0.9em"},
                        className="text-muted",
                    ),
                ],
                className="swe-card",
            ),

            # ── Ranking modes ───────────────────────────────────────────────────
            html.Div(
                [
                    html.Div("Ranking Modes", className="swe-section-title"),
                    html.P(
                        "Five modes control how similarity scores are computed. "
                        "Auto is the right choice for almost all queries — the others "
                        "exist for specific situations where they outperform Auto."
                    ),
                    html.Ul(
                        [
                            html.Li([
                                html.Strong("Auto (default): "),
                                "Detects what kind of query you typed and applies the "
                                "best strategy automatically. Factual ranking queries "
                                "(e.g. 'most visited country', 'highest military spending') are "
                                "intercepted and answered directly from structured World Bank / "
                                "SIPRI / V-Dem data — bypassing embeddings entirely for accurate "
                                "numeric rankings. All other queries use multi-section semantic "
                                "similarity with a Wikipedia-calibrated length penalty that "
                                "suppresses countries with very short, generic profiles.",
                            ]),
                            html.Li([
                                html.Strong("Brand / Associations: "),
                                "Scores countries against hand-curated brand profiles — "
                                "short, dense texts describing what each country is globally "
                                "associated with: cultural exports, iconic food and drink, "
                                "famous inventions, sports, music, landmarks. "
                                "Best for queries like 'reggae music', 'espresso culture', "
                                "'thermal baths', 'carnival parade', 'whale sharks'. "
                                "Applies a light length penalty so countries with thin profiles "
                                "do not crowd out richer matches. "
                                "Falls back to main embedding for countries without a brand profile.",
                            ]),
                            html.Li([
                                html.Strong("Semantic: "),
                                "Pure embedding similarity against each country's Wikipedia "
                                "profile text (summary + five thematic sections: economy, "
                                "culture, geography, politics, history). "
                                "Best for broad conceptual queries — geopolitical character, "
                                "environmental features, demographic structure, historical context. "
                                "Identical to Auto for non-factual queries.",
                            ]),
                            html.Li([
                                html.Strong("Hybrid (+ BM25): "),
                                "Blends Semantic scores with a BM25 keyword overlap score "
                                "(8% BM25, 92% semantic). Useful when your query contains "
                                "exact terms that appear verbatim in country profiles — "
                                "a language name, a named policy, a geographic term. "
                                "Can add noise for abstract or conceptual queries where "
                                "many profiles share the keywords incidentally.",
                            ]),
                            html.Li([
                                html.Strong("Domain (experimental): "),
                                "Blends Semantic scores with structured domain embeddings "
                                "built from World Bank indicators, NATO membership, nuclear "
                                "arsenals, UNESCO heritage site counts, and aviation data. "
                                "Most useful for categorical membership queries "
                                "('NATO ally', 'nuclear state', 'UNESCO heritage rich'). "
                                "Does not improve numeric magnitude queries — "
                                "cosine similarity cannot rank countries by how large a "
                                "number is, only by how related the description is.",
                            ]),
                        ]
                    ),
                ],
                className="swe-card",
            ),

            # ── Smart query router ──────────────────────────────────────────────
            html.Div(
                [
                    html.Div("Smart Query Router", className="swe-section-title"),
                    html.P(
                        "In Auto mode, a set of keyword triggers intercepts queries that "
                        "are inherently numeric ranking questions and routes them to direct "
                        "structured data instead of embeddings. These queries are answered "
                        "from World Bank, SIPRI, and V-Dem datasets and ranked by the actual "
                        "number, not by textual similarity."
                    ),
                    html.P("Currently routed to structured data automatically:"),
                    html.Ul(
                        [
                            html.Li("Life expectancy / longevity"),
                            html.Li("International tourist arrivals"),
                            html.Li("Democracy / electoral freedom index (V-Dem)"),
                            html.Li("Internet users (% of population)"),
                            html.Li("Renewable energy share"),
                            html.Li("Agriculture share of GDP"),
                            html.Li("Military spending (absolute USD)"),
                        ]
                    ),
                    html.P(
                        "When a query is routed, the mode tag in the confidence badge "
                        "reads 'factual:…' and the confidence badge is hidden — "
                        "factual rankings are not similarity scores and do not have "
                        "meaningful confidence gaps.",
                        style={"fontSize": "0.9em"},
                        className="text-muted",
                    ),
                ],
                className="swe-card",
            ),

            # ── How it works ───────────────────────────────────────────────────
            html.Div(
                [
                    html.Div("How It Works", className="swe-section-title"),
                    html.P(
                        "Each of the 250 countries and territories has two kinds of representation:"
                    ),
                    html.Ul(
                        [
                            html.Li([
                                html.Strong("Wikipedia profile embedding: "),
                                "The full Wikipedia article (summary + five thematic sections) "
                                "was embedded once, offline, using Voyage AI's voyage-4 model "
                                "(1024-dimensional vectors). Section embeddings (economy, culture, "
                                "geography, politics, history) allow the system to boost the "
                                "score when the relevant section of a profile matches your query "
                                "particularly well, even if the overall profile is broad.",
                            ]),
                            html.Li([
                                html.Strong("Brand profile embedding: "),
                                "A separate, curated short text describing what the country is "
                                "globally associated with — food, music, sport, inventions, "
                                "landmarks — embedded with the same model in document mode. "
                                "Used exclusively by Brand / Associations mode.",
                            ]),
                        ]
                    ),
                    html.P(
                        "When you type a query, only the query is embedded at runtime. "
                        "Its vector is compared to every precomputed country vector via "
                        "cosine similarity (a dot product, since all vectors are unit-normalized). "
                        "Higher similarity means the country's profile text is semantically "
                        "closer to your query."
                    ),
                    html.P(
                        "Map colors are rank-normalized per slot so the color gradient "
                        "always spans the full range of countries with meaningful signal — "
                        "countries below the noise floor are shown in gray."
                    ),
                ],
                className="swe-card",
            ),

            # ── Data sources ───────────────────────────────────────────────────
            html.Div(
                [
                    html.Div("Data Sources & Credits", className="swe-section-title"),
                    html.Ul(
                        [
                            html.Li(
                                "Country list, ISO codes, regions, and centroid coordinates: "
                                "mledoze/countries open dataset."
                            ),
                            html.Li(
                                "Profile text (summary, economy, culture, geography, politics, "
                                "history sections): Wikipedia (CC BY-SA)."
                            ),
                            html.Li(
                                "Brand / association profiles: hand-curated, describing globally "
                                "recognised cultural exports, cuisine, sport, music, inventions, "
                                "and landmarks per country."
                            ),
                            html.Li(
                                "Structured facts displayed in the Evidence panel — "
                                "life expectancy, GDP per capita, renewable energy share, "
                                "democracy index, internet users, tourist arrivals, "
                                "military expenditure, population, average temperature: "
                                "Our World in Data (ourworldindata.org), World Bank, and SIPRI."
                            ),
                            html.Li(
                                "Domain embeddings (Domain mode): World Bank indicators "
                                "(military personnel, R&D expenditure, high-technology exports, "
                                "logistics performance), NATO membership list, "
                                "UNESCO World Heritage Site counts (2024), nuclear arsenal data."
                            ),
                            html.Li(
                                "Factual router data — military spending, democracy index, "
                                "life expectancy, tourist arrivals, internet users, "
                                "renewable share, agriculture share: "
                                "World Bank, V-Dem, SIPRI, Our World in Data."
                            ),
                            html.Li(
                                "Embeddings: Voyage AI voyage-4, 1024-dimensional, "
                                "document and query modes."
                            ),
                        ]
                    ),
                ],
                className="swe-card",
            ),

            # ── Limitations ────────────────────────────────────────────────────
            html.Div(
                [
                    html.Div("Limitations", className="swe-section-title"),
                    html.Ul(
                        [
                            html.Li(
                                "Profiles are based on Wikipedia and may be incomplete, "
                                "outdated, or reflect that source's editorial biases. "
                                "Results are illustrative, not authoritative."
                            ),
                            html.Li(
                                "Cosine similarity measures how closely a country's profile "
                                "is described in terms related to your query — it does not "
                                "sort countries by a number. For 'biggest military', "
                                "'most tourists', or other numeric comparisons, use the "
                                "Smart Query Router (Auto mode) or check the Evidence panel."
                            ),
                            html.Li(
                                "Brand mode covers countries with explicit brand profiles. "
                                "Countries without a curated profile fall back to the main "
                                "embedding and may rank lower than expected on cultural queries."
                            ),
                            html.Li(
                                "Very small territories with short, generic Wikipedia articles "
                                "can appear as moderate matches across many unrelated topics. "
                                "The length penalty in Semantic and Brand modes reduces but "
                                "does not eliminate this effect."
                            ),
                            html.Li(
                                "Domain mode is experimental. It helps categorical queries "
                                "(NATO membership, nuclear state, aviation hub) but "
                                "does not reliably improve numeric magnitude queries "
                                "and can regress on general comparisons."
                            ),
                            html.Li(
                                "A small number of territories have no polygon in the base "
                                "map and are shown as a dot at their centroid. "
                                "They remain searchable and clickable."
                            ),
                            html.Li(
                                "This app runs on a free Render web service. "
                                "It sleeps after inactivity and the first request "
                                "after a cold start may take up to 60 seconds."
                            ),
                        ]
                    ),
                ],
                className="swe-card",
            ),
        ]
    )
