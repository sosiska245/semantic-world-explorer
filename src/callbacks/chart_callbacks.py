import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, no_update

from src.config import SLOT_COLOR_HEX, SLOT_COLORS, SLOT_DISPLAY
from src.data_loader import ENTITIES_DF, N_ENTITIES, get_entity_index
from src.similarity import sims_from_store_data


def _filter_indices(selected_ids):
    """Return sorted ndarray of entity row indices for selected entity IDs, or None if no filter."""
    if not selected_ids:
        return None
    indices = [get_entity_index(eid) for eid in selected_ids]
    return np.array(sorted(i for i in indices if i is not None), dtype=int)

CARTESIAN_LAYOUT = dict(
    template="plotly",
    paper_bgcolor="#f4f3ee",
    plot_bgcolor="#ece9e2",
    font=dict(family="'IBM Plex Mono', monospace", color="#2d2a26"),
    margin=dict(l=10, r=10, t=30, b=10),
)


def _empty_figure(message):
    fig = go.Figure()
    fig.update_layout(
        **CARTESIAN_LAYOUT,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(text=message, showarrow=False, font=dict(size=14, color="#8c8579"), x=0.5, y=0.5, xref="paper", yref="paper")
        ],
    )
    return fig


@callback(
    Output("bar-chart", "figure"),
    Input("store-similarity", "data"),
    Input("bar-slot-dropdown", "value"),
    Input("bar-topn-slider", "value"),
    Input("country-filter-dropdown", "value"),
)
def update_bar_chart(sim_data, color, top_n, filter_ids):
    sims = sims_from_store_data(sim_data, N_ENTITIES)
    sim = sims.get(color)
    if sim is None:
        return _empty_figure(f"Type a concept into {SLOT_DISPLAY.get(color, 'this slot')} to see top matches")

    filter_idx = _filter_indices(filter_ids)
    if filter_idx is not None and len(filter_idx) > 0:
        sub_sim = sim[filter_idx]
        order = np.argsort(sub_sim)[::-1]
        sorted_indices = filter_idx[order]
        names      = ENTITIES_DF["name"].iloc[sorted_indices].to_numpy()[::-1]
        entity_ids = ENTITIES_DF["id"].iloc[sorted_indices].to_numpy()[::-1]
        values = sub_sim[order][::-1]
        title = f"Selected {len(filter_idx)} countries — {SLOT_DISPLAY.get(color, '')}"
    else:
        order = np.argsort(sim)[::-1][: int(top_n)]
        names      = ENTITIES_DF["name"].iloc[order].to_numpy()[::-1]
        entity_ids = ENTITIES_DF["id"].iloc[order].to_numpy()[::-1]
        values = sim[order][::-1]
        title = f"Top {int(top_n)} — {SLOT_DISPLAY.get(color, '')}"

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            customdata=entity_ids,
            orientation="h",
            marker_color=SLOT_COLOR_HEX.get(color, "#5c9aff"),
        )
    )
    fig.update_layout(
        **CARTESIAN_LAYOUT,
        title=title,
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
    Input("country-filter-dropdown", "value"),
)
def update_polarity_scatter(sim_data, x_color, y_color, selected_id, filter_ids):
    sims = sims_from_store_data(sim_data, N_ENTITIES)

    if not x_color or not y_color or sims.get(x_color) is None or sims.get(y_color) is None:
        return _empty_figure("Add at least two concepts to compare them here")

    x_all = sims[x_color]
    y_all = sims[y_color]

    filter_idx = _filter_indices(filter_ids)
    if filter_idx is not None and len(filter_idx) > 0:
        x_vals = x_all[filter_idx]
        y_vals = y_all[filter_idx]
        names  = ENTITIES_DF["name"].iloc[filter_idx].to_numpy()
        ids    = ENTITIES_DF["id"].iloc[filter_idx].to_numpy()
    else:
        x_vals = x_all
        y_vals = y_all
        names  = ENTITIES_DF["name"].to_numpy()
        ids    = ENTITIES_DF["id"].to_numpy()

    fig = go.Figure(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            customdata=ids,
            text=names,
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
                    x=[x_all[idx]],
                    y=[y_all[idx]],
                    mode="markers",
                    marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=2, color="#d97757")),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    suffix = f" (filtered to {len(filter_idx)})" if filter_idx is not None else ""
    fig.update_layout(
        **CARTESIAN_LAYOUT,
        xaxis_title=f"Similarity — {SLOT_DISPLAY.get(x_color, '')}{suffix}",
        yaxis_title=f"Similarity — {SLOT_DISPLAY.get(y_color, '')}",
    )
    return fig


@callback(
    Output("country-filter-info", "children"),
    Input("country-filter-dropdown", "value"),
)
def update_filter_info(filter_ids):
    if not filter_ids:
        return ""
    n = len(filter_ids)
    return f"Filtered to {n} {'country' if n == 1 else 'countries'} — charts show only these."


@callback(
    Output("country-filter-dropdown", "value", allow_duplicate=True),
    Input("polarity-scatter", "selectedData"),
    State("country-filter-dropdown", "value"),
    prevent_initial_call=True,
)
def scatter_box_select_to_filter(selected_data, current_filter):
    if not selected_data or not selected_data.get("points"):
        return no_update
    new_ids = [p["customdata"] for p in selected_data["points"] if p.get("customdata")]
    if not new_ids:
        return no_update
    current = list(current_filter or [])
    for eid in new_ids:
        if eid not in current:
            current.append(eid)
    return current
