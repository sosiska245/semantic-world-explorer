import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback

from src.data_loader import ENTITIES_DF, N_ENTITIES, entities_for_country, get_entity, get_entity_index
from src.similarity import NEUTRAL_GRAY, colors_from_normalized, normalize_all, sims_from_store_data

GEO_STYLE = dict(
    projection_type="natural earth",
    showland=True,
    landcolor="#2a2a3a",
    showcountries=True,
    countrycolor="#444455",
    showocean=True,
    oceancolor="#15151f",
    showframe=False,
    bgcolor="rgba(0,0,0,0)",
)


def _entity_colors(sim_data):
    sims = sims_from_store_data(sim_data, N_ENTITIES)
    if all(sims[c] is None for c in sims):
        return [NEUTRAL_GRAY] * N_ENTITIES
    normalized = normalize_all(sims, N_ENTITIES)
    return colors_from_normalized(normalized, "blend", N_ENTITIES)


@callback(
    Output("world-map", "figure"),
    Input("store-similarity", "data"),
    Input("store-selected-entity", "data"),
)
def update_world_map(sim_data, selected_id):
    colors = _entity_colors(sim_data)
    sizes = np.where(ENTITIES_DF["type"] == "country", 9, 5)

    fig = go.Figure(
        go.Scattergeo(
            lon=ENTITIES_DF["lon"],
            lat=ENTITIES_DF["lat"],
            mode="markers",
            marker=dict(color=colors, size=sizes, line=dict(width=0.5, color="rgba(255,255,255,0.3)")),
            customdata=ENTITIES_DF["id"],
            text=ENTITIES_DF["name"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    if selected_id:
        idx = get_entity_index(selected_id)
        if idx is not None:
            fig.add_trace(
                go.Scattergeo(
                    lon=[ENTITIES_DF["lon"].iloc[idx]],
                    lat=[ENTITIES_DF["lat"].iloc[idx]],
                    mode="markers",
                    marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(width=3, color="white")),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_geos(**GEO_STYLE)
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#15151f",
        plot_bgcolor="#15151f",
    )
    return fig


def _empty_geo_figure(message):
    fig = go.Figure()
    fig.update_geos(**GEO_STYLE)
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#15151f",
        plot_bgcolor="#15151f",
        annotations=[
            dict(text=message, showarrow=False, font=dict(size=14, color="#9a9ab0"), x=0.5, y=0.5, xref="paper", yref="paper")
        ],
    )
    return fig


@callback(
    Output("city-drilldown-map", "figure"),
    Output("drilldown-title", "children"),
    Input("store-selected-entity", "data"),
    Input("store-similarity", "data"),
    Input("drilldown-color-mode", "value"),
)
def update_drilldown(selected_id, sim_data, color_mode):
    if not selected_id:
        return _empty_geo_figure("Select a country or city to see its cities"), "City Drill-down"

    entity = get_entity(selected_id)
    if entity is None:
        return _empty_geo_figure("Select a country or city to see its cities"), "City Drill-down"

    iso3 = entity["iso3"]
    country_name = entity["name"] if entity["type"] == "country" else entity["parent_country"]
    title = f"Cities in {country_name}"

    cities_df = entities_for_country(iso3, include_country=False)
    if cities_df.empty:
        return _empty_geo_figure(f"No curated cities for {country_name}"), title

    sims = sims_from_store_data(sim_data, N_ENTITIES)
    normalized = normalize_all(sims, N_ENTITIES)
    idx = cities_df.index.to_numpy()
    sub_normalized = {c: arr[idx] for c, arr in normalized.items()}
    colors = colors_from_normalized(sub_normalized, color_mode, len(idx))

    fig = go.Figure(
        go.Scattergeo(
            lon=cities_df["lon"],
            lat=cities_df["lat"],
            mode="markers+text",
            text=cities_df["name"],
            textposition="top center",
            textfont=dict(color="#d5d5ea", size=10),
            marker=dict(color=colors, size=12, line=dict(width=0.5, color="rgba(255,255,255,0.3)")),
            customdata=cities_df["id"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    pad = 2.0
    fig.update_geos(
        **GEO_STYLE,
        lataxis_range=[cities_df["lat"].min() - pad, cities_df["lat"].max() + pad],
        lonaxis_range=[cities_df["lon"].min() - pad, cities_df["lon"].max() + pad],
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#15151f",
        plot_bgcolor="#15151f",
    )
    return fig, title
