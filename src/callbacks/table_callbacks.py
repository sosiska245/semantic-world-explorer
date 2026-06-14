from dash import Input, Output, callback
from dash.dash_table.Format import Format, Scheme

from src.config import SLOT_COLORS, SLOT_DISPLAY
from src.data_loader import ENTITIES_DF, N_ENTITIES
from src.similarity import sims_from_store_data

SIM_FORMAT = Format(precision=3, scheme=Scheme.fixed)

TOP_N = 50

BAR_LENGTH = 10
BAR_FILLED = "█"
BAR_EMPTY = "░"


def _relevance_bar(value, vmin, vmax, length=BAR_LENGTH):
    span = vmax - vmin
    frac = 0.0 if span < 1e-9 else (value - vmin) / span
    filled = round(frac * length)
    return BAR_FILLED * filled + BAR_EMPTY * (length - filled)


@callback(
    Output("ranking-table", "data"),
    Output("ranking-table", "columns"),
    Output("ranking-table", "sort_by"),
    Input("store-similarity", "data"),
    Input("ranking-sort-dropdown", "value"),
)
def update_ranking_table(sim_data, sort_color):
    sims = sims_from_store_data(sim_data, N_ENTITIES)

    df = ENTITIES_DF[["id", "name"]].copy()

    active_colors = [c for c in SLOT_COLORS if sims.get(c) is not None]
    for color in active_colors:
        df[f"sim_{color}"] = sims[color]

    sort_col = f"sim_{sort_color}"
    if sort_color in active_colors:
        sim_arr = sims[sort_color]
        vmin, vmax = float(sim_arr.min()), float(sim_arr.max())
        df["relevance_bar"] = [_relevance_bar(v, vmin, vmax) for v in sim_arr]
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df = df.head(TOP_N)
        sort_by = [{"column_id": sort_col, "direction": "desc"}]
        columns = [
            {"name": "#", "id": "rank"},
            {"name": "Relevance", "id": "relevance_bar"},
            {"name": "Name", "id": "name"},
        ]
    else:
        sort_by = []
        columns = [
            {"name": "Name", "id": "name"},
        ]

    for color in active_colors:
        col_id = f"sim_{color}"
        columns.append({"name": f"Sim - {SLOT_DISPLAY[color]}", "id": col_id, "type": "numeric", "format": SIM_FORMAT})

    return df.to_dict("records"), columns, sort_by
