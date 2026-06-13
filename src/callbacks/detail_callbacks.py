from dash import Input, Output, State, callback, ctx, html, no_update

from src.config import SLOT_COLORS, SLOT_DISPLAY
from src.data_loader import N_ENTITIES, get_entity, get_entity_index
from src.similarity import sims_from_store_data


@callback(
    Output("store-selected-entity", "data"),
    Input("ranking-table", "selected_rows"),
    Input("world-map", "clickData"),
    Input("polarity-scatter", "clickData"),
    Input("city-drilldown-map", "clickData"),
    Input("detail-jump-to", "value"),
    State("ranking-table", "data"),
    prevent_initial_call=True,
)
def update_selected_entity(selected_rows, map_click, scatter_click, drilldown_click, jump_value, table_data):
    trigger = ctx.triggered_id

    if trigger == "ranking-table":
        if selected_rows:
            return table_data[selected_rows[0]]["id"]
        return no_update

    if trigger == "world-map":
        return map_click["points"][0]["customdata"]

    if trigger == "polarity-scatter":
        return scatter_click["points"][0]["customdata"]

    if trigger == "city-drilldown-map":
        return drilldown_click["points"][0]["customdata"]

    if trigger == "detail-jump-to":
        return jump_value

    return no_update


@callback(
    Output("detail-panel-body", "children"),
    Input("store-selected-entity", "data"),
    Input("store-similarity", "data"),
)
def update_detail_panel(selected_id, sim_data):
    if not selected_id:
        return html.P(
            "Click a marker, table row, or scatter point - or use the dropdown "
            "above - to see details here.",
            className="text-muted",
        )

    entity = get_entity(selected_id)
    if entity is None:
        return html.P("Entity not found.", className="text-muted")

    sims = sims_from_store_data(sim_data, N_ENTITIES)
    idx = get_entity_index(selected_id)

    sim_rows = []
    for color in SLOT_COLORS:
        sim = sims.get(color)
        if sim is None:
            continue
        sim_rows.append(
            html.Div(
                [
                    html.Span(className=f"slot-swatch slot-{color}"),
                    html.Span(f"{SLOT_DISPLAY[color]}: {float(sim[idx]):.3f}"),
                ],
                className="detail-sim-row",
            )
        )

    if entity["type"] == "country":
        type_label = "Country"
    else:
        type_label = f"City, {entity.get('parent_country', '')}"

    wiki_url = "https://en.wikipedia.org/wiki/" + entity["name"].replace(" ", "_")

    return [
        html.H5(entity["name"]),
        html.P(type_label, className="text-muted"),
        html.P(entity["profile_excerpt"]),
        html.Div(sim_rows),
        html.A("Wikipedia ->", href=wiki_url, target="_blank", className="d-block mt-2"),
    ]
