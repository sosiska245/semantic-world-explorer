import dash_bootstrap_components as dbc
from dash import dcc, html

from src.config import SLOT_COLORS, SLOT_DISPLAY
from src.data_loader import ENTITIES_DF

PRESET_QUERIES = [
    "renewable energy leaders",
    "Buddhist culture and meditation",
    "island nations",
    "alpine mountains and skiing",
    "oil and gas exporters",
    "tropical rainforest climate",
    "democracy and civil liberties",
    "coffee and tea culture",
    "space exploration technology",
    "football soccer culture",
]


def slot_card(index):
    color = SLOT_COLORS[index]
    return html.Div(
        [
            html.Label(SLOT_DISPLAY[color], className=f"slot-label slot-label-{color}"),
            dcc.Input(
                id={"type": "slot-input", "index": index},
                type="text",
                placeholder="e.g. blond people and Christianity",
                debounce=True,
                value="",
                className=f"slot-input slot-input-{color}",
            ),
        ],
        className="slot-card",
    )


def _jump_to_options():
    return [{"label": f"{row['name']} (Country)", "value": row["id"]} for _, row in ENTITIES_DF.iterrows()]


def sidebar():
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Concepts", className="swe-section-title"),
                    html.Div(id="slot-inputs-container", children=[slot_card(0)]),
                    html.Div(
                        [
                            dbc.Button("+ Add concept", id="btn-add-slot", size="sm", color="link", className="swe-btn me-2"),
                            dbc.Button("- Remove", id="btn-remove-slot", size="sm", color="link", className="swe-btn", disabled=True),
                        ],
                        className="mb-2",
                    ),
                    html.Div(
                        [html.Div("Try:", className="chips-label")]
                        + [
                            html.Button(q, id={"type": "query-chip", "query": q}, n_clicks=0, className="query-chip")
                            for q in PRESET_QUERIES
                        ],
                        id="chips-container",
                        className="chips-container",
                        style={"marginBottom": "0.75rem"},
                    ),
                    html.Div(
                        [
                            html.Div("Ranking mode", className="slot-label"),
                            dcc.RadioItems(
                                id="ranking-mode-radio",
                                options=[
                                    {"label": "Auto", "value": "auto"},
                                    {"label": "Semantic", "value": "semantic"},
                                    {"label": "Hybrid (+ BM25)", "value": "hybrid"},
                                    {"label": "Domain (exp.)", "value": "domain"},
                                    {"label": "Brand / Associations (exp.)", "value": "brand"},
                                ],
                                value="auto",
                                inline=True,
                                className="ranking-mode-radio",
                            ),
                        ],
                        className="mb-2",
                    ),
                    dbc.Alert(id="confidence-alert",     is_open=False,  className="mt-2", style={"fontSize": "0.82em", "padding": "0.4rem 0.75rem"}),
                    dbc.Alert(id="routing-info-alert",   color="info",   is_open=False, className="mt-1", style={"fontSize": "0.82em"}),
                    dbc.Alert(id="embedding-error-alert", color="danger", is_open=False, className="mt-1"),
                ],
                className="swe-card",
            ),
            html.Div(
                [
                    html.Div("Details", className="swe-section-title"),
                    dcc.Dropdown(
                        id="detail-jump-to",
                        options=_jump_to_options(),
                        placeholder="Jump to a country...",
                        className="mb-2",
                    ),
                    html.Div(
                        id="detail-panel-body",
                        children=html.P(
                            "Click a marker, table row, or scatter point - or use the "
                            "dropdown above - to see details here.",
                            className="text-muted",
                        ),
                    ),
                    html.Button(
                        "Unselect all",
                        id="clear-filter-detail-btn",
                        n_clicks=0,
                        className="swe-btn",
                        style={"display": "none", "marginTop": "0.6rem", "fontSize": "0.78em"},
                    ),
                ],
                id="sidebar-details-card",
                className="swe-card",
            ),
            html.Div(
                [
                    html.Button(
                        "Clear",
                        id="clear-filter-sidebar-btn",
                        n_clicks=0,
                        style={"display": "none"},
                    ),
                ],
                id="sidebar-selected-card",
                className="swe-card",
                style={"display": "none"},
            ),
        ],
        id="sidebar",
    )
