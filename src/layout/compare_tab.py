from dash import dash_table, dcc, html

from src.config import SLOT_OPTIONS
from src.data_loader import ENTITIES_DF

_TABLE_STYLE_HEADER = {
    "backgroundColor": "#ece9e2",
    "color": "#2d2a26",
    "fontFamily": "'IBM Plex Mono', monospace",
    "fontWeight": "600",
    "border": "none",
    "fontSize": "0.82em",
}
_TABLE_STYLE_CELL = {
    "backgroundColor": "#ffffff",
    "color": "#2d2a26",
    "fontFamily": "'IBM Plex Mono', monospace",
    "border": "1px solid #ddd9ce",
    "padding": "5px 8px",
    "fontSize": "0.82em",
}
_TABLE_STYLE_DATA_CONDITIONAL = [
    {"if": {"state": "active"}, "backgroundColor": "#f0ddd4", "border": "1px solid #d97757"},
    {"if": {"filter_query": "{rank} <= 5", "column_id": "rank"}, "color": "#d97757", "fontWeight": "700"},
]
_TABLE_COL_CONDITIONAL = [
    {"if": {"column_id": "score"},       "width": "58px", "textAlign": "right"},
    {"if": {"column_id": "rank"},        "width": "52px", "textAlign": "right"},
    {"if": {"column_id": "top_section"}, "width": "90px"},
] + [
    {"if": {"column_id": c}, "width": "72px", "textAlign": "right"}
    for c in ("economy", "culture", "geography", "politics", "history")
]


def _country_filter_options():
    opts = []
    for _, row in ENTITIES_DF.iterrows():
        iso = str(row["iso3"]) if "iso3" in ENTITIES_DF.columns else ""
        iso = "" if iso in ("nan", "None") else iso
        label = f"{row['name']} ({iso})" if iso else row["name"]
        opts.append({"label": label, "value": row["id"]})
    return opts


def compare_layout():
    return html.Div(
        [
            # ── Country Filter ──────────────────────────────────────────────
            html.Div(
                [
                    html.Div("Country Filter", className="swe-section-title"),
                    html.P(
                        "Select countries to compare. Type a name or ISO3 code, "
                        "or hold ⌘ (Cmd / Ctrl) and click any country on the map, "
                        "table, or chart to add it here.",
                        className="text-muted",
                        style={"fontSize": "0.82em", "marginBottom": "0.5rem"},
                    ),
                    dcc.Dropdown(
                        id="country-filter-dropdown",
                        options=_country_filter_options(),
                        multi=True,
                        searchable=True,
                        placeholder="Type country name or ISO3 code…",
                        clearable=True,
                        className="mb-1",
                    ),
                    html.Div(
                        id="country-filter-info",
                        className="text-muted",
                        style={"fontSize": "0.82em", "marginTop": "0.3rem"},
                    ),
                ],
                className="swe-card",
            ),

            # ── Top Matches bar chart ───────────────────────────────────────
            html.Div(
                [
                    html.Div("Top Matches", className="swe-section-title"),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="bar-slot-dropdown",
                                options=SLOT_OPTIONS,
                                value="R",
                                clearable=False,
                                style={"width": "240px", "display": "inline-block", "marginRight": "1.5rem"},
                            ),
                            html.Div(
                                [
                                    html.Label("Show top N:", className="text-muted me-2"),
                                    dcc.Slider(
                                        id="bar-topn-slider",
                                        min=5,
                                        max=30,
                                        step=5,
                                        value=10,
                                        marks={n: str(n) for n in range(5, 31, 5)},
                                    ),
                                ],
                                style={"display": "inline-block", "width": "300px", "verticalAlign": "middle"},
                            ),
                        ],
                        className="mb-2",
                    ),
                    dcc.Loading(
                        dcc.Graph(id="bar-chart", style={"height": "420px"}),
                        type="circle",
                    ),
                ],
                className="swe-card",
            ),

            # ── Polarity scatter ────────────────────────────────────────────
            html.Div(
                [
                    html.Div("Polarity", className="swe-section-title"),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="polarity-x-dropdown",
                                placeholder="X axis concept",
                                clearable=False,
                                style={"width": "240px", "display": "inline-block", "marginRight": "1rem"},
                            ),
                            dcc.Dropdown(
                                id="polarity-y-dropdown",
                                placeholder="Y axis concept",
                                clearable=False,
                                style={"width": "240px", "display": "inline-block"},
                            ),
                        ],
                        className="mb-2",
                    ),
                    dcc.Loading(
                        dcc.Graph(id="polarity-scatter", style={"height": "480px"}),
                        type="circle",
                    ),
                ],
                className="swe-card",
            ),

            # ── Country Details ─────────────────────────────────────────────
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Country Details", className="swe-section-title d-inline-block me-3"),
                            dcc.RadioItems(
                                id="compare-view-mode",
                                options=[
                                    {"label": "Cards", "value": "cards"},
                                    {"label": "Table", "value": "table"},
                                    {"label": "Compact", "value": "compact"},
                                ],
                                value="cards",
                                inline=True,
                                labelStyle={"marginRight": "1rem", "cursor": "pointer"},
                                className="compare-view-radio d-inline-flex",
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "0.5rem", "marginBottom": "0.5rem"},
                    ),
                    html.P(
                        "Select countries in the filter above (or ⌘-click on the map) to compare.",
                        id="compare-details-hint",
                        className="text-muted",
                        style={"fontSize": "0.85em"},
                    ),
                    # Cards / Compact output div
                    html.Div(id="compare-detail-panels"),
                    # Table (always in DOM — hidden until table mode active)
                    html.Div(
                        dash_table.DataTable(
                            id="compare-table",
                            columns=[
                                {"name": "Country",     "id": "name",        "type": "text"},
                                {"name": "Score",       "id": "score",       "type": "numeric"},
                                {"name": "Rank",        "id": "rank",        "type": "numeric"},
                                {"name": "Top Section", "id": "top_section", "type": "text"},
                                {"name": "Economy",     "id": "economy",     "type": "numeric"},
                                {"name": "Culture",     "id": "culture",     "type": "numeric"},
                                {"name": "Geography",   "id": "geography",   "type": "numeric"},
                                {"name": "Politics",    "id": "politics",    "type": "numeric"},
                                {"name": "History",     "id": "history",     "type": "numeric"},
                            ],
                            data=[],
                            sort_action="native",
                            page_size=50,
                            style_table={"overflowX": "auto", "marginTop": "0.5rem"},
                            style_header=_TABLE_STYLE_HEADER,
                            style_cell=_TABLE_STYLE_CELL,
                            style_cell_conditional=_TABLE_COL_CONDITIONAL,
                            style_data_conditional=_TABLE_STYLE_DATA_CONDITIONAL,
                        ),
                        id="compare-table-wrapper",
                        style={"display": "none"},
                    ),
                ],
                className="swe-card",
            ),
        ]
    )
