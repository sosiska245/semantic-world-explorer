from dash import ALL, ClientsideFunction, Input, Output, State, callback, clientside_callback, ctx, html, no_update

from src.config import MAX_SLOTS, SLOT_COLORS
from src.data_loader import EMBEDDINGS, ENTITIES_DF, N_ENTITIES
from src.layout.sidebar import slot_card
from src.similarity import embed_query, sims_from_store_data


@callback(
    Output("store-num-slots", "data"),
    Input("btn-add-slot", "n_clicks"),
    Input("btn-remove-slot", "n_clicks"),
    State("store-num-slots", "data"),
    prevent_initial_call=True,
)
def update_num_slots(_add_clicks, _remove_clicks, num_slots):
    trigger = ctx.triggered_id
    if trigger == "btn-add-slot":
        return min(num_slots + 1, MAX_SLOTS)
    if trigger == "btn-remove-slot":
        return max(num_slots - 1, 1)
    return no_update


@callback(
    Output("slot-inputs-container", "children"),
    Output("btn-add-slot", "disabled"),
    Output("btn-remove-slot", "disabled"),
    Input("store-num-slots", "data"),
    State("store-slots", "data"),
)
def render_slot_inputs(num_slots, slots_data):
    slots_data = slots_data or {}
    children = []
    for i in range(num_slots):
        card = slot_card(i)
        existing_text = slots_data.get(SLOT_COLORS[i])
        if existing_text:
            card.children[1].value = existing_text
        children.append(card)
    return children, num_slots >= MAX_SLOTS, num_slots <= 1


@callback(
    Output("store-slots", "data"),
    Input({"type": "slot-input", "index": ALL}, "value"),
)
def update_store_slots(values):
    data = {color: None for color in SLOT_COLORS}
    for i, val in enumerate(values):
        if i < len(SLOT_COLORS):
            text = (val or "").strip()
            data[SLOT_COLORS[i]] = text or None
    return data


# Track which slot input was most recently typed in
clientside_callback(
    """
    function() {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) return window.dash_clientside.no_update;
        var prop_id = ctx.triggered[0].prop_id;
        var idStr = prop_id.split('.')[0];
        try {
            var idObj = JSON.parse(idStr);
            if (typeof idObj.index !== 'undefined') return idObj.index;
        } catch(e) {}
        return window.dash_clientside.no_update;
    }
    """,
    Output("store-last-focused-slot", "data"),
    Input({"type": "slot-input", "index": ALL}, "value"),
    prevent_initial_call=True,
)


@callback(
    Output({"type": "slot-input", "index": ALL}, "value"),
    Input({"type": "query-chip", "query": ALL}, "n_clicks"),
    State("store-last-focused-slot", "data"),
    State({"type": "slot-input", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def chip_clicked(n_clicks_list, last_focused_slot, current_values):
    if not any(n for n in n_clicks_list if n):
        return [no_update] * len(current_values)
    query = ctx.triggered_id["query"]
    result = [no_update] * len(current_values)
    # Priority 1: first empty slot (fill it)
    for i, v in enumerate(current_values):
        if not (v or "").strip():
            result[i] = query
            return result
    # Priority 2: all slots filled → target last-typed slot
    target = min(int(last_focused_slot or 0), len(result) - 1)
    result[target] = query
    return result


@callback(
    Output("store-ranking-mode", "data"),
    Input("ranking-mode-radio", "value"),
)
def update_ranking_mode(mode):
    return mode


# ── helpers (defined before the callback that uses them) ─────────────────────

_ENERGY_TERMS = {
    "energy", "solar", "renewable", "electricity", "nuclear",
    "hydroelectric", "photovoltaic", "clean energy", "fossil fuel",
}

# Commodity / crop keywords where BM25 keyword matching outperforms pure cosine.
# Applied in Auto mode only — forces hybrid path when these appear in the query.
_HYBRID_CROP_TERMS = {
    "wheat", "barley", "oats", "rye", "maize",
    "cocoa", "cacao",
    "viticulture", "paddy",
    "plantation",
    "livestock", "ranching",
    "rice cultivation", "coffee production", "wine production",
    "tea production", "spice farming", "cereal farming", "grain farming",
    "fishing industry", "seafood export", "farming export", "agriculture export",
}

_ISO3_LIST = ENTITIES_DF["iso3"].tolist()


def _section_alpha(query_text: str) -> float:
    t = query_text.lower()
    if any(term in t for term in _ENERGY_TERMS):
        return 0.10
    return 0.30


def _auto_use_hybrid(query_text: str) -> bool:
    """Return True when BM25 keyword boost helps (specific crops/commodities)."""
    t = query_text.lower()
    return any(term in t for term in _HYBRID_CROP_TERMS)


# ── main similarity callback ──────────────────────────────────────────────────

@callback(
    Output("store-similarity", "data"),
    Output("embedding-error-alert", "children"),
    Output("embedding-error-alert", "is_open"),
    Output("routing-info-alert", "children"),
    Output("routing-info-alert", "is_open"),
    Output("store-mode-info", "data"),
    Output("store-query-vecs", "data"),
    Input("store-slots", "data"),
    Input("store-ranking-mode", "data"),
)
def update_similarities(slots_data, ranking_mode):
    from src.multivec import apply_length_penalty, multi_sim

    slots_data   = slots_data or {}
    ranking_mode = ranking_mode or "auto"
    sims         = {}
    mode_info    = {}
    query_vecs   = {}
    errors       = []
    routing_msgs = []

    for color in SLOT_COLORS:
        text = slots_data.get(color)
        if not text:
            sims[color]       = None
            mode_info[color]  = None
            query_vecs[color] = None
            continue

        slot_num = SLOT_COLORS.index(color) + 1

        # ── Auto mode: check factual router first ─────────────────────────
        if ranking_mode == "auto":
            from src.factual_router import (
                detect_route, factual_coverage, factual_scores, get_source_info,
            )
            route = detect_route(text)
            if route is not None:
                indicator, ascending, label = route
                scores  = factual_scores(indicator, _ISO3_LIST, ascending)
                n_cov   = factual_coverage(indicator, _ISO3_LIST)
                src_inf = get_source_info(indicator, _ISO3_LIST)
                sims[color]       = scores.tolist()
                mode_info[color]  = f"auto→factual:{indicator}"
                query_vecs[color] = None
                routing_msgs.append(
                    f"Slot {slot_num}: Factual ranking — {label}"
                    f" · {n_cov} countries"
                    + (f" · {src_inf}" if src_inf else "")
                )
                continue
            # No factual route → fall through to embedding path below

        # ── Embedding path ────────────────────────────────────────────────
        try:
            vec = embed_query(text)
        except Exception as exc:  # noqa: BLE001
            sims[color]       = None
            mode_info[color]  = None
            query_vecs[color] = None
            errors.append(f"Couldn't embed Slot {slot_num}: {exc}")
            print(f"[embed_query error] {exc!r} status={getattr(exc, 'http_status', None)} headers={dict(getattr(exc, 'headers', {}) or {})}", flush=True)
            continue

        if ranking_mode == "brand":
            from src.brand_sim import brand_sim
            final = brand_sim(vec)
            mode_info[color] = "brand"
        elif ranking_mode == "domain":
            from src.domain_sim import domain_sim
            final = domain_sim(vec, alpha=0.4)
            final = apply_length_penalty(final)
            mode_info[color] = "domain"
        else:
            cosine = multi_sim(query_vec=vec, alpha=_section_alpha(text))
            cosine = apply_length_penalty(cosine)
            use_hybrid = (
                ranking_mode == "hybrid"
                or (ranking_mode == "auto" and _auto_use_hybrid(text))
            )
            if use_hybrid:
                from src.bm25 import blend as bm25_blend
                final = bm25_blend(cosine, text, alpha=0.08)
                mode_info[color] = "auto→hybrid" if ranking_mode == "auto" else "hybrid"
                if ranking_mode == "auto":
                    routing_msgs.append(f"Slot {slot_num}: Auto → Hybrid (BM25 boosted)")
            else:
                final = cosine
                mode_info[color] = "auto→semantic" if ranking_mode == "auto" else "semantic"
                if ranking_mode == "auto":
                    routing_msgs.append(f"Slot {slot_num}: Auto → Semantic embedding")

        query_vecs[color] = vec.tolist()
        sims[color]       = final.tolist()

    error_msg    = " | ".join(errors)
    routing_info = " | ".join(routing_msgs)
    return sims, error_msg, bool(error_msg), routing_info, bool(routing_info), mode_info, query_vecs


_CONF_HIGH   = 0.10
_CONF_MEDIUM = 0.05


@callback(
    Output("confidence-alert", "children"),
    Output("confidence-alert", "is_open"),
    Output("confidence-alert", "color"),
    Input("store-similarity", "data"),
    Input("ranking-sort-dropdown", "value"),
    State("store-mode-info", "data"),
)
def update_confidence_badge(sim_data, sort_color, mode_info):
    sort_color = sort_color or "R"
    mode_str   = (mode_info or {}).get(sort_color, "")

    # Factual routes: scores are ordinal, Δ is meaningless
    if mode_str and "factual" in mode_str:
        return "", False, "info"

    sims    = sims_from_store_data(sim_data, N_ENTITIES)
    sim_arr = sims.get(sort_color)
    if sim_arr is None:
        return "", False, "info"

    sorted_sims = sorted(sim_arr, reverse=True)
    if len(sorted_sims) < 20:
        return "", False, "info"

    delta = sorted_sims[0] - sorted_sims[19]

    if delta > _CONF_HIGH:
        return "Query confidence: HIGH — results are well-separated.", True, "success"
    if delta > _CONF_MEDIUM:
        return "Query confidence: MEDIUM — moderate score separation.", True, "warning"
    return (
        "Query confidence: LOW — this query is broad and many countries score similarly. "
        "Try adding more specific keywords.",
        True,
        "warning",
    )


@callback(
    Output("chips-container", "style"),
    Output("sidebar-details-card", "style"),
    Output("detail-jump-to", "style"),
    Input("tabs", "value"),
)
def toggle_tab_visibility(tab):
    if tab == "compare":
        return {"display": "none"}, {}, {"display": "none"}
    return {"marginBottom": "0.75rem"}, {}, {"marginBottom": "0.5rem"}


@callback(
    Output("compare-view-mode", "value"),
    Output("store-compare-count-prev", "data"),
    Input("country-filter-dropdown", "value"),
    State("compare-view-mode", "value"),
    State("store-compare-count-prev", "data"),
)
def auto_view_mode(filter_ids, current_mode, prev_count):
    n    = len(filter_ids or [])
    prev = prev_count or 0
    if prev <= 8 and n > 8:
        return "table", n
    if prev > 8 and n <= 8:
        return "cards", n
    return no_update, n


@callback(
    Output("sidebar-selected-card", "children"),
    Output("sidebar-selected-card", "style"),
    Input("tabs", "value"),
    Input("country-filter-dropdown", "value"),
)
def update_selected_card(tab, filter_ids):
    if tab != "explorer" or not filter_ids:
        return [], {"display": "none"}

    id_to_name = {row["id"]: row["name"] for _, row in ENTITIES_DF.iterrows()}
    items = [
        html.Button(
            id_to_name.get(eid, eid),
            id={"type": "selected-item-btn", "eid": eid},
            n_clicks=0,
            className="selected-item-btn",
            title="Click → details · ⌘-click → remove",
        )
        for eid in filter_ids
    ]
    return [
        html.Div(
            [
                html.Div("Selected", className="swe-section-title d-inline-block"),
                html.Button(
                    "Clear",
                    id="clear-filter-sidebar-btn",
                    n_clicks=0,
                    className="swe-btn",
                    style={"fontSize": "0.72em", "padding": "0.1rem 0.4rem", "marginLeft": "0.5rem", "verticalAlign": "middle"},
                ),
            ],
            style={"marginBottom": "0.3rem"},
        ),
        html.Div(items, style={"display": "flex", "flexDirection": "column", "gap": "0.15rem"}),
        html.P(
            "Click → details  ·  ⌘-click → remove",
            className="text-muted",
            style={"fontSize": "0.75em", "marginTop": "0.4rem"},
        ),
    ], {}
