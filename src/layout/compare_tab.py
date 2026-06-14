from dash import dcc, html

from src.config import SLOT_OPTIONS


def compare_layout():
    return html.Div(
        [
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
                    dcc.Graph(id="bar-chart", style={"height": "420px"}),
                ],
                className="swe-card",
            ),
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
                    dcc.Graph(id="polarity-scatter", style={"height": "480px"}),
                ],
                className="swe-card",
            ),
        ]
    )
