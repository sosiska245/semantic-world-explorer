from dash import dash_table, dcc, html

from src.config import SLOT_OPTIONS


def explorer_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Div("World Map", className="swe-section-title"),
                    html.Div(
                        [
                            dcc.Graph(id="world-map", style={"height": "560px"}),
                            html.Div(id="map-legend", className="map-legend-box"),
                        ],
                        className="map-card",
                    ),
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
                                style={"width": "240px", "display": "inline-block"},
                            ),
                        ]
                    ),
                    dash_table.DataTable(
                        id="ranking-table",
                        columns=[
                            {"name": "Name", "id": "name"},
                        ],
                        data=[],
                        sort_action="native",
                        row_selectable="single",
                        page_size=12,
                        style_table={"overflowX": "auto", "marginTop": "0.75rem"},
                        style_header={
                            "backgroundColor": "#ece9e2",
                            "color": "#2d2a26",
                            "fontFamily": "'IBM Plex Mono', monospace",
                            "fontWeight": "600",
                            "border": "none",
                        },
                        style_cell={
                            "backgroundColor": "#ffffff",
                            "color": "#2d2a26",
                            "fontFamily": "'IBM Plex Mono', monospace",
                            "border": "1px solid #ddd9ce",
                            "padding": "6px 10px",
                        },
                        style_cell_conditional=[
                            {"if": {"column_id": "rank"}, "width": "48px", "textAlign": "center"},
                            {"if": {"column_id": "relevance_bar"}, "width": "110px", "letterSpacing": "1px"},
                        ],
                        style_data_conditional=[
                            {
                                "if": {"filter_query": "{rank} <= 3"},
                                "backgroundColor": "#f7f0e3",
                            },
                            {
                                "if": {"filter_query": "{rank} = 1", "column_id": "rank"},
                                "fontWeight": "700",
                                "color": "#d97757",
                            },
                            {
                                "if": {"state": "selected"},
                                "backgroundColor": "#f0ddd4",
                                "border": "1px solid #d97757",
                            },
                        ],
                    ),
                ],
                className="swe-card",
            ),
        ]
    )
