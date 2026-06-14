from dash import Input, Output, State, callback, ctx, html, no_update

from src.config import SLOT_COLORS, SLOT_DISPLAY
from src.data_loader import N_ENTITIES, get_entity, get_entity_index
from src.facts_loader import INDICATOR_DISPLAY, INDICATOR_LABELS, get_facts, match_category
from src.similarity import sims_from_store_data


@callback(
    Output("store-selected-entity", "data"),
    Input("ranking-table", "selected_rows"),
    Input("world-map", "clickData"),
    Input("polarity-scatter", "clickData"),
    Input("detail-jump-to", "value"),
    State("ranking-table", "data"),
    prevent_initial_call=True,
)
def update_selected_entity(selected_rows, map_click, scatter_click, jump_value, table_data):
    trigger = ctx.triggered_id

    if trigger == "ranking-table":
        if selected_rows:
            return table_data[selected_rows[0]]["id"]
        return no_update

    if trigger == "world-map":
        return map_click["points"][0]["customdata"]

    if trigger == "polarity-scatter":
        return scatter_click["points"][0]["customdata"]

    if trigger == "detail-jump-to":
        return jump_value

    return no_update


def _evidence_block(iso3, slots_data):
    """For each active slot whose query text matches a Phase-1 facts
    category (energy/agriculture/politics/geography), render either the
    country's factual indicators for that category ("factual evidence") or
    an explicit "no structured data" note ("missing data"). Slots whose
    query matches no category contribute nothing - those stay pure
    "semantic association", unchanged."""
    blocks = []
    for color in SLOT_COLORS:
        text = (slots_data or {}).get(color)
        category = match_category(text)
        if category is None:
            continue

        header = html.Div(
            [
                html.Span(className=f"slot-swatch slot-{color}"),
                html.Span(f"{SLOT_DISPLAY[color]} - {category} evidence:"),
            ],
            className="detail-sim-row",
        )

        facts = get_facts(iso3, category)
        if not facts:
            blocks.append(
                html.Div([header, html.P("No structured data available for this topic.", className="text-muted detail-evidence-empty")])
            )
            continue

        items = []
        for f in facts:
            label = INDICATOR_LABELS.get(f["indicator"], f["indicator"])
            display = INDICATOR_DISPLAY.get(f["indicator"], lambda v, u: f"{v} {u}")(f["value"], f["unit"])
            items.append(html.Li(f"{label}: {display} ({f['source']}, {f['year']})"))
        blocks.append(html.Div([header, html.Ul(items, className="detail-evidence-list")]))

    return blocks


@callback(
    Output("detail-panel-body", "children"),
    Input("store-selected-entity", "data"),
    Input("store-similarity", "data"),
    Input("store-slots", "data"),
)
def update_detail_panel(selected_id, sim_data, slots_data):
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
                    html.Span(f"{SLOT_DISPLAY[color]} - semantic similarity: {float(sim[idx]):.3f}"),
                ],
                className="detail-sim-row",
            )
        )

    evidence_blocks = _evidence_block(entity["iso3"], slots_data)

    wiki_url = "https://en.wikipedia.org/wiki/" + entity["name"].replace(" ", "_")

    children = [
        html.H5(entity["name"]),
        html.P("Country", className="text-muted"),
        html.P(entity["profile_excerpt"]),
        html.Div(sim_rows),
    ]
    if evidence_blocks:
        children.append(html.Div(evidence_blocks, className="detail-evidence"))
    children.append(html.A("Wikipedia ->", href=wiki_url, target="_blank", className="d-block mt-2"))
    return children
