"""Rich country detail card for the sidebar Detail panel and Compare tab."""

import numpy as np
from dash import (
    ALL, ClientsideFunction, Input, Output, State,
    callback, clientside_callback, ctx, html, no_update,
)

from src.config import SLOT_COLORS, SLOT_DISPLAY
from src.data_loader import (
    N_ENTITIES,
    SECTION_EMBEDDINGS,
    SECTION_KEYS,
    get_entity,
    get_entity_index,
)
from src.facts_loader import INDICATOR_DISPLAY, INDICATOR_LABELS, get_facts, match_category
from src.similarity import sims_from_store_data

# ── Clientside click router: map / scatter / bar / table → store-click-event ──
clientside_callback(
    ClientsideFunction(namespace="swe", function_name="routeClick"),
    Output("store-click-event", "data"),
    Input("world-map", "clickData"),
    Input("polarity-scatter", "clickData"),
    Input("bar-chart", "clickData"),
    Input("ranking-table", "active_cell"),
    State("ranking-table", "data"),
)

# ── Clientside click router: sidebar Selected list buttons → store-list-click ─
clientside_callback(
    ClientsideFunction(namespace="swe", function_name="routeSelectedItem"),
    Output("store-list-click", "data"),
    Input({"type": "selected-item-btn", "eid": ALL}, "n_clicks"),
    prevent_initial_call=True,
)

# ── ISO-3 → ISO-2 (for flag emoji) ───────────────────────────────────────────
_ISO3_TO_ISO2 = {
    "AFG": "AF", "ALB": "AL", "DZA": "DZ", "AND": "AD", "AGO": "AO",
    "ATG": "AG", "ARG": "AR", "ARM": "AM", "AUS": "AU", "AUT": "AT",
    "AZE": "AZ", "BHS": "BS", "BHR": "BH", "BGD": "BD", "BRB": "BB",
    "BLR": "BY", "BEL": "BE", "BLZ": "BZ", "BEN": "BJ", "BTN": "BT",
    "BOL": "BO", "BIH": "BA", "BWA": "BW", "BRA": "BR", "BRN": "BN",
    "BGR": "BG", "BFA": "BF", "BDI": "BI", "CPV": "CV", "KHM": "KH",
    "CMR": "CM", "CAN": "CA", "CAF": "CF", "TCD": "TD", "CHL": "CL",
    "CHN": "CN", "COL": "CO", "COM": "KM", "COD": "CD", "COG": "CG",
    "CRI": "CR", "CIV": "CI", "HRV": "HR", "CUB": "CU", "CYP": "CY",
    "CZE": "CZ", "DNK": "DK", "DJI": "DJ", "DOM": "DO", "ECU": "EC",
    "EGY": "EG", "SLV": "SV", "GNQ": "GQ", "ERI": "ER", "EST": "EE",
    "SWZ": "SZ", "ETH": "ET", "FJI": "FJ", "FIN": "FI", "FRA": "FR",
    "GAB": "GA", "GMB": "GM", "GEO": "GE", "DEU": "DE", "GHA": "GH",
    "GRC": "GR", "GRD": "GD", "GTM": "GT", "GIN": "GN", "GNB": "GW",
    "GUY": "GY", "HTI": "HT", "HND": "HN", "HUN": "HU", "ISL": "IS",
    "IND": "IN", "IDN": "ID", "IRN": "IR", "IRQ": "IQ", "IRL": "IE",
    "ISR": "IL", "ITA": "IT", "JAM": "JM", "JPN": "JP", "JOR": "JO",
    "KAZ": "KZ", "KEN": "KE", "KIR": "KI", "PRK": "KP", "KOR": "KR",
    "KWT": "KW", "KGZ": "KG", "LAO": "LA", "LVA": "LV", "LBN": "LB",
    "LSO": "LS", "LBR": "LR", "LBY": "LY", "LIE": "LI", "LTU": "LT",
    "LUX": "LU", "MDG": "MG", "MWI": "MW", "MYS": "MY", "MDV": "MV",
    "MLI": "ML", "MLT": "MT", "MHL": "MH", "MRT": "MR", "MUS": "MU",
    "MEX": "MX", "FSM": "FM", "MDA": "MD", "MCO": "MC", "MNG": "MN",
    "MNE": "ME", "MAR": "MA", "MOZ": "MZ", "MMR": "MM", "NAM": "NA",
    "NRU": "NR", "NPL": "NP", "NLD": "NL", "NZL": "NZ", "NIC": "NI",
    "NER": "NE", "NGA": "NG", "MKD": "MK", "NOR": "NO", "OMN": "OM",
    "PAK": "PK", "PLW": "PW", "PAN": "PA", "PNG": "PG", "PRY": "PY",
    "PER": "PE", "PHL": "PH", "POL": "PL", "PRT": "PT", "QAT": "QA",
    "ROU": "RO", "RUS": "RU", "RWA": "RW", "KNA": "KN", "LCA": "LC",
    "VCT": "VC", "WSM": "WS", "SMR": "SM", "STP": "ST", "SAU": "SA",
    "SEN": "SN", "SRB": "RS", "SYC": "SC", "SLE": "SL", "SGP": "SG",
    "SVK": "SK", "SVN": "SI", "SLB": "SB", "SOM": "SO", "ZAF": "ZA",
    "SSD": "SS", "ESP": "ES", "LKA": "LK", "SDN": "SD", "SUR": "SR",
    "SWE": "SE", "CHE": "CH", "SYR": "SY", "TWN": "TW", "TJK": "TJ",
    "TZA": "TZ", "THA": "TH", "TLS": "TL", "TGO": "TG", "TON": "TO",
    "TTO": "TT", "TUN": "TN", "TUR": "TR", "TKM": "TM", "TUV": "TV",
    "UGA": "UG", "UKR": "UA", "ARE": "AE", "GBR": "GB", "USA": "US",
    "URY": "UY", "UZB": "UZ", "VUT": "VU", "VEN": "VE", "VNM": "VN",
    "YEM": "YE", "ZMB": "ZM", "ZWE": "ZW", "VAT": "VA",
}

_SECTION_LABELS = {
    "economy": "Economy",
    "culture": "Culture",
    "geography": "Geography",
    "politics": "Politics",
    "history": "History",
}


def _flag(iso3: str) -> str:
    iso2 = _ISO3_TO_ISO2.get(iso3, "")
    if not iso2:
        return ""
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in iso2.upper())


def _mode_label(mode_str: str) -> str:
    if not mode_str:
        return "Semantic"
    if mode_str.startswith("auto→factual:"):
        ind = mode_str[len("auto→factual:"):]
        return f"Auto → {INDICATOR_LABELS.get(ind, ind)}"
    if mode_str.startswith("auto→"):
        resolved = mode_str[len("auto→"):]
        nice = {"semantic": "Semantic", "hybrid": "Hybrid (BM25)"}
        return f"Auto → {nice.get(resolved, resolved.title())}"
    if mode_str.startswith("factual:"):
        ind = mode_str[len("factual:"):]
        return f"Factual · {INDICATOR_LABELS.get(ind, ind)}"
    return {"semantic": "Semantic", "hybrid": "Hybrid (BM25)", "domain": "Domain"}.get(
        mode_str, mode_str.title()
    )


def _section_scores(country_idx: int, query_vec) -> dict:
    """Raw cosine similarity between query vec and each section embedding."""
    if query_vec is None or country_idx is None:
        return {}
    vec = np.array(query_vec, dtype=np.float32)
    norm_q = np.linalg.norm(vec)
    if norm_q < 1e-9:
        return {}
    scores = {}
    for key in SECTION_KEYS:
        vecs = SECTION_EMBEDDINGS.get(key, [])
        if country_idx < len(vecs) and vecs[country_idx] is not None:
            sv = vecs[country_idx]
            norm_s = np.linalg.norm(sv)
            if norm_s > 1e-9:
                scores[key] = float(np.dot(vec, sv) / (norm_q * norm_s))
    return scores


def _why_text(section_scores: dict) -> str:
    if not section_scores:
        return "Match based on overall country profile."
    sorted_secs = sorted(section_scores.items(), key=lambda x: -x[1])
    top = [_SECTION_LABELS.get(k, k.title()) for k, v in sorted_secs if v >= 0.60]
    mid = [_SECTION_LABELS.get(k, k.title()) for k, v in sorted_secs if 0.40 <= v < 0.60]
    if not top and not mid:
        top = [_SECTION_LABELS.get(sorted_secs[0][0], sorted_secs[0][0].title())]
    parts = []
    if top:
        if len(top) == 1:
            parts.append(f"Strong match on {top[0]}.")
        elif len(top) == 2:
            parts.append(f"Strong match on {top[0]} and {top[1]}.")
        else:
            parts.append(f"Strong match on {', '.join(top[:-1])} and {top[-1]}.")
    if mid:
        if len(mid) == 1:
            parts.append(f"Also relevant: {mid[0]}.")
        else:
            parts.append(f"Also relevant: {', '.join(mid[:-1])} and {mid[-1]}.")
    return " ".join(parts) or "Match across multiple profile areas."


def _section_breakdown(section_scores: dict, highlight: str) -> html.Div:
    sorted_secs = sorted(section_scores.items(), key=lambda x: -x[1])
    rows = []
    for key, score in sorted_secs:
        label = _SECTION_LABELS.get(key, key.title())
        bar_pct = int(round(score * 100))
        is_hl = highlight == key
        rows.append(
            html.Div(
                [
                    html.Span(label, className="section-bar-label"),
                    html.Div(
                        html.Div(
                            className="section-bar-fill",
                            style={"width": f"{bar_pct}%"},
                        ),
                        className="section-bar-track",
                    ),
                    html.Span(str(bar_pct), className="section-bar-score"),
                ],
                className="section-bar-row" + (" section-bar-row--hl" if is_hl else ""),
            )
        )
    return html.Div(rows, className="section-breakdown")


def _fact_pills(iso3: str, slots_data: dict) -> html.Div | None:
    pills = []
    seen = set()
    for color in SLOT_COLORS:
        text = (slots_data or {}).get(color)
        if not text:
            continue
        category = match_category(text)
        if category is None:
            continue
        facts = get_facts(iso3, category)
        if not facts:
            continue
        for f in facts[:3]:
            ind = f["indicator"]
            if ind in seen:
                continue
            seen.add(ind)
            label = INDICATOR_LABELS.get(ind, ind)
            display_fn = INDICATOR_DISPLAY.get(ind, lambda v, u: f"{v} {u}")
            try:
                value_str = display_fn(f["value"], f["unit"])
            except Exception:
                value_str = str(f["value"])
            pills.append(
                html.Span(
                    [
                        html.Span(label, className="pill-label"),
                        html.Span(value_str, className="pill-value"),
                    ],
                    className="fact-pill",
                )
            )
    if not pills:
        return None
    return html.Div(pills, className="fact-pills-row")


# ── section highlight callback ────────────────────────────────────────────────

@callback(
    Output("store-highlight-section", "data"),
    Input({"type": "section-btn", "section": ALL}, "n_clicks"),
    Input("store-selected-entity", "data"),
    State("store-highlight-section", "data"),
    prevent_initial_call=True,
)
def handle_section_highlight(n_clicks_list, _selected_entity, current_highlight):
    trigger = ctx.triggered_id
    if trigger == "store-selected-entity":
        return None
    if not n_clicks_list or all((n is None or n == 0) for n in n_clicks_list):
        return no_update
    if isinstance(trigger, dict) and trigger.get("type") == "section-btn":
        section = trigger["section"]
        return None if current_highlight == section else section
    return no_update


# ── Clear-all filter buttons ──────────────────────────────────────────────────

@callback(
    Output("country-filter-dropdown", "value", allow_duplicate=True),
    Input("clear-filter-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_filter(n_clicks):
    if not n_clicks:
        return no_update
    return []


@callback(
    Output("country-filter-dropdown", "value", allow_duplicate=True),
    Input("clear-filter-sidebar-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_filter_sidebar(n_clicks):
    if not n_clicks:
        return no_update
    return []


# ── entity selection / compare-filter routing ─────────────────────────────────

@callback(
    Output("store-selected-entity", "data"),
    Output("country-filter-dropdown", "value"),
    Input("store-click-event", "data"),
    Input("store-list-click", "data"),
    Input("detail-jump-to", "value"),
    State("country-filter-dropdown", "value"),
    prevent_initial_call=True,
)
def route_selection(click_event, list_click, jump_value, current_filter):
    trigger = ctx.triggered_id
    if trigger == "detail-jump-to":
        return jump_value, no_update
    event = click_event if trigger == "store-click-event" else list_click
    if event:
        entity_id = event.get("entity_id")
        if not entity_id:
            return no_update, no_update
        if event.get("meta"):
            current = list(current_filter or [])
            if entity_id in current:
                current.remove(entity_id)
            else:
                current.append(entity_id)
            return no_update, current
        return entity_id, no_update
    return no_update, no_update


# ── main detail card ──────────────────────────────────────────────────────────

@callback(
    Output("detail-panel-body", "children"),
    Input("store-selected-entity", "data"),
    Input("store-similarity", "data"),
    Input("store-slots", "data"),
    Input("store-mode-info", "data"),
    Input("store-query-vecs", "data"),
    Input("store-highlight-section", "data"),
    Input("ranking-sort-dropdown", "value"),
)
def update_detail_panel(
    selected_id, sim_data, slots_data, mode_info, query_vecs, highlight, sort_color
):
    if not selected_id:
        return html.P(
            "Click a marker, table row, or scatter point — or use the dropdown above — to see details here.",
            className="text-muted",
            style={"fontSize": "0.85rem"},
        )

    entity = get_entity(selected_id)
    if entity is None:
        return html.P("Entity not found.", className="text-muted")

    iso3 = entity["iso3"]
    name = entity["name"]
    idx  = get_entity_index(selected_id)
    flag = _flag(iso3)

    # ── Score and rank ────────────────────────────────────────────────────
    sims       = sims_from_store_data(sim_data, N_ENTITIES)
    sort_color = sort_color or "R"
    sim_arr    = sims.get(sort_color)
    score_pct  = None
    rank_n     = None
    if sim_arr is not None and idx is not None:
        arr     = np.array(sim_arr, dtype=np.float32)
        top_val = float(arr.max())
        if top_val > 0:
            score_pct = int(round(float(arr[idx]) / top_val * 100))
        rank_n = int(np.sum(arr > float(arr[idx]))) + 1

    # ── Mode label ────────────────────────────────────────────────────────
    mode_str  = (mode_info or {}).get(sort_color)
    mode_text = _mode_label(mode_str)

    # ── Section scores ────────────────────────────────────────────────────
    query_vec = (query_vecs or {}).get(sort_color)
    sec_scores = _section_scores(idx, query_vec)

    # ── Wikipedia link ────────────────────────────────────────────────────
    wiki_url = "https://en.wikipedia.org/wiki/" + name.replace(" ", "_")

    # ── Header: flag + name + wiki ────────────────────────────────────────
    header = html.Div(
        [
            html.Div(f"{flag} {name}" if flag else name, className="detail-country-name"),
            html.A("Wikipedia →", href=wiki_url, target="_blank", className="detail-wiki-link"),
        ],
        className="detail-header",
    )

    # ── Score block ───────────────────────────────────────────────────────
    if score_pct is not None:
        score_block = html.Div(
            [
                html.Span(f"Match Score {score_pct}/100", className="detail-score-value"),
                html.Span(f" · #{rank_n}", className="detail-score-rank"),
                html.Div(f"Mode: {mode_text}", className="detail-mode-label"),
            ],
            className="detail-score-block",
        )
    else:
        score_block = html.Div(
            html.Span("No query active", className="text-muted"),
            className="detail-score-block",
        )

    # ── Secondary slot scores (multi-slot, max 2) ─────────────────────────
    secondary_block = None
    secondary_spans = []
    for color in SLOT_COLORS:
        if color == sort_color:
            continue
        arr2 = sims.get(color)
        if arr2 is None or idx is None:
            continue
        a2 = np.array(arr2, dtype=np.float32)
        top2 = float(a2.max())
        if top2 > 0:
            p2 = int(round(float(a2[idx]) / top2 * 100))
            secondary_spans.append(
                html.Span(
                    f"{SLOT_DISPLAY[color]}: {p2}/100",
                    className=f"detail-slot-score slot-score-{color}",
                )
            )
    if secondary_spans:
        secondary_block = html.Div(secondary_spans[:2], className="detail-secondary-scores")

    # ── Why it matches ────────────────────────────────────────────────────
    why_section = None
    if sec_scores:
        sorted_secs = sorted(sec_scores.items(), key=lambda x: -x[1])
        why_text    = _why_text(sec_scores)
        btns = [
            html.Button(
                _SECTION_LABELS.get(k, k.title()),
                id={"type": "section-btn", "section": k},
                n_clicks=0,
                className=(
                    "section-link-btn section-link-btn--active"
                    if highlight == k
                    else "section-link-btn"
                ),
            )
            for k, _ in sorted_secs[:3]
        ]
        why_section = html.Div(
            [
                html.Div("Why it matches", className="detail-sub-label"),
                html.P(why_text, className="detail-why-text"),
                html.Div(btns, className="detail-section-btns"),
            ],
            className="detail-why-section",
        )
    elif mode_str and "factual" in mode_str:
        why_section = html.Div(
            html.P(
                "Ranked by factual indicator — no text similarity computed.",
                className="detail-why-text",
            ),
            className="detail-why-section",
        )

    # ── Section breakdown ─────────────────────────────────────────────────
    breakdown = None
    if sec_scores:
        breakdown = html.Div(
            [
                html.Div("Section breakdown", className="detail-sub-label"),
                _section_breakdown(sec_scores, highlight),
            ],
            className="detail-breakdown-section",
        )

    # ── Fact pills ────────────────────────────────────────────────────────
    pills_div = _fact_pills(iso3, slots_data)
    pills_section = None
    if pills_div:
        pills_section = html.Div(
            [html.Div("Fact evidence", className="detail-sub-label"), pills_div],
            className="detail-pills-section",
        )

    # ── Assemble ──────────────────────────────────────────────────────────
    parts = [header, score_block]
    if secondary_block:
        parts.append(secondary_block)
    if why_section:
        parts.append(why_section)
    if breakdown:
        parts.append(breakdown)
    if pills_section:
        parts.append(pills_section)

    return html.Div(parts, className="detail-card")


# ── Compare-tab helpers ───────────────────────────────────────────────────────

def _entity_score_rank(entity_id, sims, sort_color):
    """Return (score_pct, rank_n) for entity, or (None, None)."""
    idx = get_entity_index(entity_id)
    sim_arr = sims.get(sort_color or "R")
    if sim_arr is None or idx is None:
        return None, None
    arr = np.array(sim_arr, dtype=np.float32)
    top_val = float(arr.max())
    score_pct = int(round(float(arr[idx]) / top_val * 100)) if top_val > 0 else None
    rank_n = int(np.sum(arr > float(arr[idx]))) + 1
    return score_pct, rank_n


def _build_compare_card(entity_id, sims, mode_info, query_vecs, slots_data, sort_color):
    entity = get_entity(entity_id)
    if entity is None:
        return html.Div("Not found", className="text-muted")

    iso3 = entity["iso3"]
    name = entity["name"]
    idx  = get_entity_index(entity_id)
    flag = _flag(iso3)
    wiki_url = "https://en.wikipedia.org/wiki/" + name.replace(" ", "_")

    score_pct, rank_n = _entity_score_rank(entity_id, sims, sort_color)
    mode_str  = (mode_info or {}).get(sort_color or "R")
    mode_text = _mode_label(mode_str)
    query_vec = (query_vecs or {}).get(sort_color or "R")
    sec_scores = _section_scores(idx, query_vec)

    header = html.Div(
        [
            html.Div(f"{flag} {name}" if flag else name, className="detail-country-name"),
            html.A("Wikipedia →", href=wiki_url, target="_blank", className="detail-wiki-link"),
        ],
        className="detail-header",
    )
    score_block = html.Div(
        [
            html.Span(f"Match Score {score_pct}/100", className="detail-score-value")
            if score_pct is not None else html.Span("No query active", className="text-muted"),
            html.Span(f" · #{rank_n}", className="detail-score-rank") if rank_n else "",
            html.Div(f"Mode: {mode_text}", className="detail-mode-label"),
        ],
        className="detail-score-block",
    )
    breakdown = (
        html.Div(
            [html.Div("Section breakdown", className="detail-sub-label"), _section_breakdown(sec_scores, None)],
            className="detail-breakdown-section",
        )
        if sec_scores else None
    )
    pills_div = _fact_pills(iso3, slots_data)
    pills_section = (
        html.Div([html.Div("Fact evidence", className="detail-sub-label"), pills_div], className="detail-pills-section")
        if pills_div else None
    )
    parts = [header, score_block]
    if breakdown:
        parts.append(breakdown)
    if pills_section:
        parts.append(pills_section)
    return html.Div(parts, className="detail-card")


def _build_compact_row(entity_id, sims, query_vecs, sort_color):
    entity = get_entity(entity_id)
    if entity is None:
        return None
    name = entity["name"]
    idx  = get_entity_index(entity_id)
    flag = _flag(entity["iso3"])

    score_pct, rank_n = _entity_score_rank(entity_id, sims, sort_color)
    query_vec  = (query_vecs or {}).get(sort_color or "R")
    sec_scores = _section_scores(idx, query_vec)
    top_section = ""
    if sec_scores:
        top_key = max(sec_scores, key=sec_scores.get)
        top_section = _SECTION_LABELS.get(top_key, top_key.title())

    return html.Button(
        [
            html.Span(name, className="cmp-name"),
            html.Span(flag or "🌐", className="cmp-flag"),
            html.Span(str(score_pct) if score_pct is not None else "—", className="cmp-score"),
            html.Span(f"#{rank_n}" if rank_n else "—", className="cmp-rank"),
            html.Span(top_section, className="cmp-section"),
        ],
        id={"type": "compare-compact-btn", "eid": entity_id},
        n_clicks=0,
        className="cmp-row-btn",
        title="Click → details",
    )


def _build_table_row(entity_id, sims, query_vecs, sort_color):
    entity = get_entity(entity_id)
    if entity is None:
        return None
    name = entity["name"]
    idx  = get_entity_index(entity_id)

    score_pct, rank_n = _entity_score_rank(entity_id, sims, sort_color)
    query_vec  = (query_vecs or {}).get(sort_color or "R")
    sec_scores = _section_scores(idx, query_vec)
    top_section = ""
    if sec_scores:
        top_key = max(sec_scores, key=sec_scores.get)
        top_section = _SECTION_LABELS.get(top_key, top_key.title())

    row = {
        "id": entity_id,
        "name": name,
        "score": score_pct,
        "rank": rank_n,
        "top_section": top_section,
    }
    for key in ("economy", "culture", "geography", "politics", "history"):
        v = sec_scores.get(key)
        row[key] = int(round(v * 100)) if v is not None else None
    return row


# ── Compare-tab main callback ─────────────────────────────────────────────────

@callback(
    Output("compare-detail-panels", "children"),
    Output("compare-detail-panels", "style"),
    Output("compare-table", "data"),
    Output("compare-table-wrapper", "style"),
    Output("compare-details-hint", "style"),
    Input("country-filter-dropdown", "value"),
    Input("store-similarity", "data"),
    Input("store-slots", "data"),
    Input("store-mode-info", "data"),
    Input("store-query-vecs", "data"),
    Input("bar-slot-dropdown", "value"),
    Input("compare-view-mode", "value"),
)
def update_compare_details(filter_ids, sim_data, slots_data, mode_info, query_vecs, sort_color, view_mode):
    _HIDE = {"display": "none"}
    _SHOW = {}

    if not filter_ids:
        return [], _HIDE, [], _HIDE, _SHOW

    sims = sims_from_store_data(sim_data, N_ENTITIES)
    sc   = sort_color or "R"
    vm   = view_mode or "cards"

    if vm == "table":
        rows = [r for r in (_build_table_row(eid, sims, query_vecs, sc) for eid in filter_ids) if r]
        return [], _HIDE, rows, _SHOW, _HIDE

    if vm == "compact":
        btns = [b for b in (_build_compact_row(eid, sims, query_vecs, sc) for eid in filter_ids) if b]
        compact = html.Div(btns, className="cmp-list")
        return compact, _SHOW, [], _HIDE, _HIDE

    # cards (default)
    cards = [
        html.Div(_build_compare_card(eid, sims, mode_info, query_vecs, slots_data, sc), className="compare-detail-col")
        for eid in filter_ids
    ]
    return html.Div(cards, className="compare-detail-grid"), _SHOW, [], _HIDE, _HIDE


# ── Compare table click → details ─────────────────────────────────────────────

@callback(
    Output("store-selected-entity", "data", allow_duplicate=True),
    Input("compare-table", "active_cell"),
    State("compare-table", "data"),
    prevent_initial_call=True,
)
def compare_table_click(active_cell, table_data):
    if not active_cell or not table_data:
        return no_update
    row = active_cell.get("row")
    if row is not None and row < len(table_data):
        return table_data[row]["id"]
    return no_update


# ── Compact row: route via store-list-click (ctrl → remove filter, click → details) ──

clientside_callback(
    ClientsideFunction(namespace="swe", function_name="routeSelectedItem"),
    Output("store-list-click", "data", allow_duplicate=True),
    Input({"type": "compare-compact-btn", "eid": ALL}, "n_clicks"),
    prevent_initial_call=True,
)

# ── Scatter box-select (ctrl+drag) → add to compare filter ───────────────────

clientside_callback(
    ClientsideFunction(namespace="swe", function_name="routeSelectedData"),
    Output("country-filter-dropdown", "value", allow_duplicate=True),
    Input("polarity-scatter", "selectedData"),
    State("country-filter-dropdown", "value"),
    prevent_initial_call=True,
)
