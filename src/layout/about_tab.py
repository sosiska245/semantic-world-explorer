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
                        "every country and curated city is recolored by how closely its "
                        "Wikipedia-derived profile matches that concept - based on "
                        "meaning, not keyword matching."
                    ),
                    html.P(
                        "Up to three slots can be active at once (Red, Green, Blue). "
                        "With multiple slots active, each location's marker color "
                        "blends the three similarity scores, so you can see where "
                        "concepts overlap (a place strong in two concepts shows a "
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
                            html.Li("Click any map marker, table row, or scatter point - or use the jump-to dropdown - to open the detail panel and zoom the city drill-down map to that country."),
                        ]
                    ),
                ],
                className="swe-card",
            ),
            html.Div(
                [
                    html.Div("How It Works", className="swe-section-title"),
                    html.P(
                        "Each country and curated city has a 'profile' built from its "
                        "Wikipedia summary plus relevant sections (demographics, "
                        "religion, climate, culture, cuisine) and structured statistics "
                        "(average temperature, population, GDP per capita). These "
                        "profiles were embedded once, offline, using "
                        "Voyage AI's voyage-4-lite model (1024-dimensional vectors) and "
                        "stored alongside each location's coordinates."
                    ),
                    html.P(
                        "When you type a concept, only that short query is embedded - "
                        "at runtime, with the same model. Its vector is compared to "
                        "every precomputed location vector via cosine similarity "
                        "(a dot product, since Voyage embeddings are pre-normalized). "
                        "Higher similarity means the location's profile text is more "
                        "semantically related to your concept. Scores are then "
                        "min-max normalized across all locations so the color "
                        "differences are visible on the map."
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
                            html.Li("Capital and curated city coordinates: OpenStreetMap Nominatim, plus a hand-picked list of additional major cities."),
                            html.Li("Average temperature, population and GDP per capita: Our World in Data (ourworldindata.org), sourced from the World Bank and related datasets."),
                            html.Li("Profile text, including religion and culture: Wikipedia article summaries and sections (CC BY-SA)."),
                            html.Li("Embeddings: Voyage AI voyage-4-lite."),
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
                            html.Li("Cosine similarity reflects textual/semantic association, not a factual or quantitative measure - a high score means 'discussed in similar terms', not 'true' or 'most'."),
                            html.Li("City coverage is limited to capitals plus a small curated list, so the city drill-down map may be empty for some countries."),
                            html.Li("This app runs on a free Render web service, which sleeps when idle - the first request after a period of inactivity may take up to a minute to respond."),
                        ]
                    ),
                ],
                className="swe-card",
            ),
        ]
    )
