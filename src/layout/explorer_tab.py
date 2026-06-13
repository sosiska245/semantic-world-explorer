from dash import dash_table, dcc, html

from src.config import SLOT_OPTIONS


def explorer_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Div("World Map", className="swe-section-title"),
                    html.P(
                        "Marker color blends each active concept's similarity "
                        "(Slot 1 = Red, Slot 2 = Green, Slot 3 = Blue). Click a "
                        "marker to see details.",
                        className="text-muted",
                    ),
                    dcc.Graph(id="world-map", style={"height": "560px"}),
                ],
                className="swe-card",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Ranking", className="swe-section-title d-inline-block me-3"),
                            dcc.Dropdown(
                                id="ranking-sort-dropdown",
                                options=SLOT_OPTIONS,
                                value="R",
                                clearable=False,
                                style={"width": "240px", "display": "inline-block", "color": "#15151f"},
                            ),
                        ]
                    ),
                    dash_table.DataTable(
                        id="ranking-table",
                        columns=[
                            {"name": "Name", "id": "name"},
                            {"name": "Type", "id": "type"},
                        ],
                        data=[],
                        sort_action="native",
                        row_selectable="single",
                        page_size=12,
                        style_table={"overflowX": "auto", "marginTop": "0.75rem"},
                        style_header={
                            "backgroundColor": "#24243a",
                            "color": "#d5d5ea",
                            "fontWeight": "600",
                            "border": "none",
                        },
                        style_cell={
                            "backgroundColor": "#1e1e2e",
                            "color": "#f0f0f5",
                            "border": "1px solid #34344a",
                            "padding": "6px 10px",
                        },
                        style_data_conditional=[
                            {
                                "if": {"state": "selected"},
                                "backgroundColor": "#34344a",
                                "border": "1px solid #5c9aff",
                            },
                        ],
                    ),
                ],
                className="swe-card",
            ),
        ]
    )
