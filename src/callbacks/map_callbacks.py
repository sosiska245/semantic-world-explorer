import plotly.graph_objects as go
from dash import Input, Output, callback, html

from src.choropleth import LOW_RELEVANCE_GRAY, country_blend_colors, discrete_colorscale
from src.config import SLOT_COLOR_HEX, SLOT_COLORS, SLOT_DISPLAY
from src.data_loader import ENTITIES_DF, N_ENTITIES, get_entity
from src.similarity import sims_from_store_data

# Light "cartography" theme for the world map, per the reference image: cream
# land, light-blue ocean, white country borders.
GEO_STYLE_WORLD = dict(
    projection_type="natural earth",
    showland=True,
    landcolor="#f3ecd8",
    showcountries=True,
    countrycolor="#ffffff",
    showocean=True,
    oceancolor="#cfe8f5",
    showframe=False,
    bgcolor="rgba(0,0,0,0)",
)

COUNTRY_MASK = (ENTITIES_DF["type"] == "country").to_numpy()
COUNTRIES_DF = ENTITIES_DF[ENTITIES_DF["type"] == "country"]


def _hover_texts(sims, country_mask):
    """List[str], one per country in COUNTRIES_DF order: name plus a
    'Slot X (Color): <sim>' line for each active slot, joined with <br> for
    a multi-line Plotly hover tooltip."""
    lines_per_country = [["<b>" + name + "</b>"] for name in COUNTRIES_DF["name"]]
    for color in SLOT_COLORS:
        sim = sims.get(color)
        if sim is None:
            continue
        hex_col = SLOT_COLOR_HEX.get(color, "#ffffff")
        for lines, v in zip(lines_per_country, sim[country_mask]):
            lines.append(f'<span style="color:{hex_col}">{SLOT_DISPLAY[color]}: {float(v):.3f}</span>')
    return ["<br>".join(lines) for lines in lines_per_country]


@callback(
    Output("world-map", "figure"),
    Input("store-similarity", "data"),
    Input("store-selected-entity", "data"),
)
def update_world_map(sim_data, selected_id):
    sims = sims_from_store_data(sim_data, N_ENTITIES)
    active = any(sims[c] is not None for c in sims)

    fig = go.Figure()
    hover_texts = _hover_texts(sims, COUNTRY_MASK)

    _HOVER_LABEL = dict(bgcolor="rgba(30,30,30,0.88)", font_color="white", bordercolor="rgba(0,0,0,0)")

    if active:
        colors = country_blend_colors(sims, COUNTRY_MASK)
        z, zmin, zmax, colorscale = discrete_colorscale(colors)
        fig.add_trace(
            go.Choropleth(
                locations=COUNTRIES_DF["iso3"],
                locationmode="ISO-3",
                z=z,
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                showscale=False,
                marker_line_color="#ffffff",
                marker_line_width=0.5,
                customdata=COUNTRIES_DF["id"],
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                hoverlabel=_HOVER_LABEL,
            )
        )
    else:
        colors = [LOW_RELEVANCE_GRAY] * len(COUNTRIES_DF)

    # Small centroid markers, same color as the choropleth fill, for every
    # country: keeps every country clickable, and acts as a fallback for any
    # territory the built-in choropleth atlas has no polygon for (it would
    # otherwise render with no fill at all).
    fig.add_trace(
        go.Scattergeo(
            lon=COUNTRIES_DF["lon"],
            lat=COUNTRIES_DF["lat"],
            mode="markers",
            marker=dict(color=colors, size=6, line=dict(width=0.5, color="rgba(255,255,255,0.6)")),
            customdata=COUNTRIES_DF["id"],
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            hoverlabel=_HOVER_LABEL,
            showlegend=False,
        )
    )

    if selected_id:
        highlight = get_entity(selected_id)
        if highlight is not None:
            fig.add_trace(
                go.Scattergeo(
                    lon=[highlight["lon"]],
                    lat=[highlight["lat"]],
                    mode="markers",
                    marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(width=3, color="#222222")),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_geos(**GEO_STYLE_WORLD)
    fig.update_layout(
        template="plotly",
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#f3ecd8",
        plot_bgcolor="#f3ecd8",
        uirevision="world-map-static",
    )
    return fig


@callback(
    Output("map-legend", "children"),
    Input("store-slots", "data"),
)
def update_map_legend(slots_data):
    slots_data = slots_data or {}
    rows = []
    for color in SLOT_COLORS:
        text = slots_data.get(color)
        if not text:
            continue
        rows.append(
            html.Div(
                [
                    html.Span(className=f"slot-swatch slot-{color}"),
                    html.Span(text, className="legend-label"),
                ],
                className="legend-row",
            )
        )

    if not rows:
        return html.Div("Type a concept in a slot to color the map.", className="legend-empty")

    rows.append(
        html.Div(
            [
                html.Span(className="slot-swatch legend-swatch-gray"),
                html.Span("Low/no similarity to active concept(s)", className="legend-label"),
            ],
            className="legend-row",
        )
    )
    return rows
