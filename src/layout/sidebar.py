import dash_bootstrap_components as dbc
from dash import dcc, html

from src.config import SLOT_COLORS, SLOT_DISPLAY
from src.data_loader import ENTITIES_DF


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
                    dbc.Alert(id="embedding-error-alert", color="danger", is_open=False, className="mt-2"),
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
                ],
                className="swe-card",
            ),
        ],
        id="sidebar",
    )
