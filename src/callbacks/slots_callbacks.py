from dash import ALL, Input, Output, State, callback, ctx, no_update

from src.config import MAX_SLOTS, SLOT_COLORS
from src.data_loader import EMBEDDINGS
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
    Output("store-similarity", "data"),
    Output("embedding-error-alert", "children"),
    Output("embedding-error-alert", "is_open"),
    Input("store-slots", "data"),
)
def update_similarities(slots_data):
    slots_data = slots_data or {}
    sims = {}
    errors = []
    for color in SLOT_COLORS:
        text = slots_data.get(color)
        if not text:
            sims[color] = None
            continue
        try:
            vec = embed_query(text)
        except Exception as exc:  # noqa: BLE001 - surface any embedding error to the UI
            sims[color] = None
            errors.append(f"Couldn't embed Slot {SLOT_COLORS.index(color) + 1}: {exc}")
            continue
        sims[color] = (EMBEDDINGS @ vec).tolist()
    error_msg = " | ".join(errors)
    return sims, error_msg, bool(error_msg)
