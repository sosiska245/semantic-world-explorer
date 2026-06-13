import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback

from src.config import SLOT_COLOR_HEX, SLOT_COLORS, SLOT_DISPLAY
from src.data_loader import ENTITIES_DF, N_ENTITIES, get_entity_index
from src.similarity import sims_from_store_data

CARTESIAN_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#15151f",
    plot_bgcolor="#1e1e2e",
    margin=dict(l=10, r=10, t=30, b=10),
)


def _empty_figure(message):
    fig = go.Figure()
    fig.update_layout(
        **CARTESIAN_LAYOUT,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(text=message, showarrow=False, font=dict(size=14, color="#9a9ab0"), x=0.5, y=0.5, xref="paper", yref="paper")
        ],
    )
    return fig


@callback(
    Output("bar-chart", "figure"),
    Input("store-similarity", "data"),
    Input("bar-slot-dropdown", "value"),
    Input("bar-topn-slider", "value"),
)
def update_bar_chart(sim_data, color, top_n):
    sims = sims_from_store_data(sim_data, N_ENTITIES)
    sim = sims.get(color)
    if sim is None:
        return _empty_figure(f"Type a concept into {SLOT_DISPLAY.get(color, 'this slot')} to see top matches")

    order = np.argsort(sim)[::-1][: int(top_n)]
    names = ENTITIES_DF["name"].iloc[order].to_numpy()[::-1]
    values = sim[order][::-1]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=SLOT_COLOR_HEX.get(color, "#5c9aff"),
        )
    )
    fig.update_layout(
        **CARTESIAN_LAYOUT,
        title=f"Top {int(top_n)} - {SLOT_DISPLAY.get(color, '')}",
        xaxis_title="Cosine similarity",
    )
    return fig


@callback(
    Output("polarity-x-dropdown", "options"),
    Output("polarity-x-dropdown", "value"),
    Output("polarity-y-dropdown", "options"),
    Output("polarity-y-dropdown", "value"),
    Input("store-slots", "data"),
)
def update_polarity_dropdowns(slots_data):
    slots_data = slots_data or {}
    active = [c for c in SLOT_COLORS if slots_data.get(c)]
    options = [{"label": SLOT_DISPLAY[c], "value": c} for c in active]

    x_value = active[0] if len(active) >= 1 else None
    y_value = active[1] if len(active) >= 2 else (active[0] if active else None)

    return options, x_value, options, y_value


@callback(
    Output("polarity-scatter", "figure"),
    Input("store-similarity", "data"),
    Input("polarity-x-dropdown", "value"),
    Input("polarity-y-dropdown", "value"),
    Input("store-selected-entity", "data"),
)
def update_polarity_scatter(sim_data, x_color, y_color, selected_id):
    sims = sims_from_store_data(sim_data, N_ENTITIES)

    if not x_color or not y_color or sims.get(x_color) is None or sims.get(y_color) is None:
        return _empty_figure("Add at least two concepts to compare them here")

    x = sims[x_color]
    y = sims[y_color]

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            customdata=ENTITIES_DF["id"],
            text=ENTITIES_DF["name"],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=8, color="rgba(150,150,220,0.55)", line=dict(width=0)),
            showlegend=False,
        )
    )

    if selected_id:
        idx = get_entity_index(selected_id)
        if idx is not None:
            fig.add_trace(
                go.Scatter(
                    x=[x[idx]],
                    y=[y[idx]],
                    mode="markers",
                    marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=2, color="white")),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        **CARTESIAN_LAYOUT,
        xaxis_title=f"Similarity - {SLOT_DISPLAY.get(x_color, '')}",
        yaxis_title=f"Similarity - {SLOT_DISPLAY.get(y_color, '')}",
    )
    return fig
