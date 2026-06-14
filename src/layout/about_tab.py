from dash import html


def about_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Div("About Semantic World Explorer", className="swe-section-title"),
                    html.P(
                        "This dashboard lets you explore the world through the lens of "
                        "free-text concepts. Type a description of an idea, culture, "
                        "climate, cuisine, or anything else into a 'concept slot', and "
                        "every country is recolored by how closely its "
                        "Wikipedia-derived profile matches that concept - based on "
                        "meaning, not keyword matching."
                    ),
                    html.P(
                        "Up to three slots can be active at once (Red, Green, Blue). "
                        "With multiple slots active, each country's marker color "
                        "blends the three similarity scores, so you can see where "
                        "concepts overlap (a country strong in two concepts shows a "
                        "mixed color) or diverge (strong in only one)."
                    ),
                ],
                className="swe-card",
            ),
            html.Div(
                [
                    html.Div("How to Use", className="swe-section-title"),
                    html.Ol(
                        [
                            html.Li("Type a concept into Slot 1 (e.g. 'tropical beaches and surfing culture') and press Enter."),
                            html.Li("Watch the world map, ranking table, and bar chart update with similarity scores."),
                            html.Li("Click '+ Add concept' to activate Slot 2 (Green) and/or Slot 3 (Blue) for up to three concepts at once."),
                            html.Li("With two or more slots active, switch to the Compare tab to see the Polarity scatter plot."),
                            html.Li("Click any map marker, table row, or scatter point - or use the jump-to dropdown - to open the detail panel for that country."),
                        ]
                    ),
                ],
                className="swe-card",
            ),
            html.Div(
                [
                    html.Div("How It Works", className="swe-section-title"),
                    html.P(
                        "Each country has a 'profile' built from its "
                        "Wikipedia summary plus relevant sections (demographics, "
                        "religion, climate, culture, economy) and structured statistics "
                        "(average temperature, population, GDP per capita). These "
                        "profiles were embedded once, offline, using "
                        "Voyage AI's voyage-4 model (1024-dimensional vectors) and "
                        "stored alongside each country's coordinates."
                    ),
                    html.P(
                        "When you type a concept, only that short query is embedded - "
                        "at runtime, with the same model. Its vector is compared to "
                        "every precomputed country vector via cosine similarity "
                        "(a dot product, since Voyage embeddings are pre-normalized). "
                        "Higher similarity means the country's profile text is more "
                        "semantically related to your concept."
                    ),
                    html.P(
                        "On the world map, each country with no positive similarity score "
                        "on every active concept is shown in gray ('low/no similarity'); "
                        "countries above that floor are rank-normalized per concept so "
                        "the color gradient always spans the full range among the "
                        "countries that actually have signal, instead of stretching a "
                        "tiny raw-similarity range across the whole ramp."
                    ),
                ],
                className="swe-card",
            ),
            html.Div(
                [
                    html.Div("Ranking Modes", className="swe-section-title"),
                    html.P(
                        "The ranking mode selector controls how similarity scores are computed:"
                    ),
                    html.Ul(
                        [
                            html.Li([
                                html.Strong("Semantic (default): "),
                                "Pure embedding similarity. Each country's full profile text "
                                "(Wikipedia summary + structured statistics + factual sentences) "
                                "is compared to your query using cosine similarity. Best for broad "
                                "conceptual queries — culture, geography, history, religion, "
                                "political character. Uses section-level embeddings (economy, "
                                "culture, geography, politics, history) to give an extra boost "
                                "when the relevant part of the profile matches well.",
                            ]),
                            html.Li([
                                html.Strong("Hybrid (+ BM25): "),
                                "Blends Semantic scores with a BM25 keyword overlap score "
                                "(alpha=0.08 — 8% BM25, 92% semantic). Helps when your query "
                                "contains specific country-related keywords that appear verbatim in "
                                "profile text. Can add noise for abstract conceptual queries: "
                                "if a keyword appears in many unrelated profiles, BM25 "
                                "promotes false positives. Use when you have precise terms "
                                "like a language name, a specific policy, or a named region.",
                            ]),
                            html.Li([
                                html.Strong("Domain (experimental): "),
                                "Blends Semantic scores with structured domain embeddings built "
                                "from World Bank indicators, NATO/nuclear membership lists, "
                                "passenger-per-capita aviation data, and UNESCO site counts. "
                                "Useful for domain-specific factual queries such as alliance "
                                "membership ('NATO member'), aviation hub density, or semiconductor "
                                "industry. Underperforms for numeric magnitude queries "
                                "('biggest military spender', 'highest R&D') because cosine "
                                "similarity measures textual closeness, not numeric ordering — "
                                "all countries with any military data score similarly regardless "
                                "of whether they spend $10B or $800B.",
                            ]),
                        ]
                    ),
                    html.P(
                        [
                            html.Strong("Important: "),
                            "None of the ranking modes answer numeric comparison queries "
                            "like 'largest economy', 'highest military spending', or "
                            "'most tourists visited' accurately. Embedding similarity "
                            "measures how closely a profile is discussed in terms related "
                            "to your query — it does not sort countries by a number. For "
                            "factual numeric comparisons, the Evidence panel (below the "
                            "detail view) shows direct structured data from Our World in Data "
                            "and the World Bank.",
                        ],
                        className="text-muted",
                        style={"fontSize": "0.9em"},
                    ),
                ],
                className="swe-card",
            ),
            html.Div(
                [
                    html.Div("Data Sources & Credits", className="swe-section-title"),
                    html.Ul(
                        [
                            html.Li("Country list, names, regions and centroid coordinates: the mledoze/countries open dataset."),
                            html.Li("Average temperature, population and GDP per capita: Our World in Data (ourworldindata.org), sourced from the World Bank and related datasets."),
                            html.Li("Structured facts (renewable energy share, democracy index, life expectancy, health spending, internet users, tourist arrivals, military expenditure, air passengers, and more): Our World in Data, SIPRI, and World Bank via Our World in Data."),
                            html.Li("Domain embeddings (Domain mode): World Bank API indicators for military personnel, logistics performance, R&D expenditure, and high-technology exports; NATO membership list; UNESCO World Heritage Site counts (2024)."),
                            html.Li("Profile text, including religion, culture and economy: Wikipedia article summaries and sections (CC BY-SA)."),
                            html.Li("Embeddings: Voyage AI voyage-4 (1024-dimensional)."),
                        ]
                    ),
                ],
                className="swe-card",
            ),
            html.Div(
                [
                    html.Div("Limitations", className="swe-section-title"),
                    html.Ul(
                        [
                            html.Li("Profiles summarize Wikipedia content and may be incomplete, outdated, or reflect that source's own biases - results are illustrative, not authoritative."),
                            html.Li("Cosine similarity reflects textual and semantic association, not a factual or quantitative measure. A high score means the country's profile is 'discussed in similar terms', not that it ranks first on any numeric indicator. For questions like 'which country has the biggest military' or 'most tourists', use the structured Evidence panel rather than the ranking."),
                            html.Li("A handful of very small territories have no outline in the base map and are shown as a small colored dot at their location instead of a filled shape, but remain clickable."),
                            html.Li("This app runs on a free Render web service, which sleeps when idle - the first request after a period of inactivity may take up to a minute to respond."),
                            html.Li("Some niche topics may score weaker than expected if the relevant information is missing or truncated from that country's profile - each profile is capped in length to keep embeddings focused."),
                            html.Li("Small territories with short, generic profiles can sometimes appear as moderate matches across many different topics ('weak universal hubs') due to their limited profile content."),
                            html.Li("Domain mode is experimental. It helps categorical queries (alliance membership, hub aviation, semiconductor industry) but does not improve numeric magnitude queries and can regress on general technology or spending comparisons."),
                        ]
                    ),
                ],
                className="swe-card",
            ),
        ]
    )
