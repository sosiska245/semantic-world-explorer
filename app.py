import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from src.layout.about_tab import about_layout
from src.layout.compare_tab import compare_layout
from src.layout.explorer_tab import explorer_layout
from src.layout.navbar import navbar
from src.layout.sidebar import sidebar

# Importing these registers their @callback-decorated functions with Dash.
from src.callbacks import (  # noqa: F401
    chart_callbacks,
    detail_callbacks,
    map_callbacks,
    slots_callbacks,
    table_callbacks,
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Semantic World Explorer")
server = app.server

app.layout = html.Div(
    [
        dcc.Store(id="store-num-slots", data=1),
        dcc.Store(id="store-slots", data={"R": None, "G": None, "B": None}),
        dcc.Store(id="store-similarity", data={"R": None, "G": None, "B": None}),
        dcc.Store(id="store-selected-entity", data=None),
        navbar(),
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(sidebar(), width=12, lg=3),
                    dbc.Col(
                        dcc.Tabs(
                            id="tabs",
                            value="explorer",
                            className="swe-tabs",
                            children=[
                                dcc.Tab(
                                    label="Explorer",
                                    value="explorer",
                                    className="tab",
                                    selected_className="tab--selected",
                                    children=explorer_layout(),
                                ),
                                dcc.Tab(
                                    label="Compare",
                                    value="compare",
                                    className="tab",
                                    selected_className="tab--selected",
                                    children=compare_layout(),
                                ),
                                dcc.Tab(
                                    label="About / Help",
                                    value="about",
                                    className="tab",
                                    selected_className="tab--selected",
                                    children=about_layout(),
                                ),
                            ],
                        ),
                        width=12,
                        lg=9,
                    ),
                ],
            ),
            fluid=True,
            className="mt-3",
        ),
    ]
)


if __name__ == "__main__":
    app.run(debug=True)
