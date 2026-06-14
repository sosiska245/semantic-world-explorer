from dash import ALL, Input, Output, State, callback, ctx, no_update

from src.config import MAX_SLOTS, SLOT_COLORS
from src.data_loader import EMBEDDINGS, ENTITIES_DF
from src.layout.sidebar import slot_card
from src.similarity import embed_query


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


@callback(
    Output({"type": "slot-input", "index": 0}, "value"),
    Input({"type": "query-chip", "query": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def chip_clicked(n_clicks_list):
    if not any(n_clicks_list):
        return no_update
    return ctx.triggered_id["query"]


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

_ISO3_LIST = ENTITIES_DF["iso3"].tolist()


def _section_alpha(query_text: str) -> float:
    t = query_text.lower()
    if any(term in t for term in _ENERGY_TERMS):
        return 0.10
    return 0.30


# ── main similarity callback ──────────────────────────────────────────────────

@callback(
    Output("store-similarity", "data"),
    Output("embedding-error-alert", "children"),
    Output("embedding-error-alert", "is_open"),
    Output("routing-info-alert", "children"),
    Output("routing-info-alert", "is_open"),
    Input("store-slots", "data"),
    Input("store-ranking-mode", "data"),
)
def update_similarities(slots_data, ranking_mode):
    from src.multivec import apply_length_penalty, multi_sim

    slots_data   = slots_data or {}
    ranking_mode = ranking_mode or "auto"
    sims         = {}
    errors       = []
    routing_msgs = []

    for color in SLOT_COLORS:
        text = slots_data.get(color)
        if not text:
            sims[color] = None
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
                sims[color] = scores.tolist()
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
            sims[color] = None
            errors.append(f"Couldn't embed Slot {slot_num}: {exc}")
            continue

        if ranking_mode == "domain":
            from src.domain_sim import domain_sim
            final = domain_sim(vec, alpha=0.4)
            final = apply_length_penalty(final)
        else:
            cosine = multi_sim(query_vec=vec, alpha=_section_alpha(text))
            cosine = apply_length_penalty(cosine)
            if ranking_mode == "hybrid":
                from src.bm25 import blend as bm25_blend
                final = bm25_blend(cosine, text, alpha=0.08)
            else:
                final = cosine

        sims[color] = final.tolist()

    error_msg    = " | ".join(errors)
    routing_info = " | ".join(routing_msgs)
    return sims, error_msg, bool(error_msg), routing_info, bool(routing_info)
