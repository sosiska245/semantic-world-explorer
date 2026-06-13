from dash import Input, Output, callback
from dash.dash_table.Format import Format, Scheme

from src.config import SLOT_COLORS, SLOT_DISPLAY
from src.data_loader import ENTITIES_DF, N_ENTITIES
from src.similarity import sims_from_store_data

SIM_FORMAT = Format(precision=3, scheme=Scheme.fixed)


@callback(
    Output("ranking-table", "data"),
    Output("ranking-table", "columns"),
    Output("ranking-table", "sort_by"),
    Input("store-similarity", "data"),
    Input("ranking-sort-dropdown", "value"),
)
def update_ranking_table(sim_data, sort_color):
    sims = sims_from_store_data(sim_data, N_ENTITIES)

    df = ENTITIES_DF[["id", "name", "type"]].copy()
    df["type"] = df["type"].map({"country": "Country", "city": "City"})

    columns = [
        {"name": "Name", "id": "name"},
        {"name": "Type", "id": "type"},
    ]

    active_colors = [c for c in SLOT_COLORS if sims.get(c) is not None]
    for color in active_colors:
        col_id = f"sim_{color}"
        df[col_id] = sims[color]
        columns.append({"name": f"Sim - {SLOT_DISPLAY[color]}", "id": col_id, "type": "numeric", "format": SIM_FORMAT})

    sort_col = f"sim_{sort_color}"
    if sort_color in active_colors:
        df = df.sort_values(sort_col, ascending=False)
        sort_by = [{"column_id": sort_col, "direction": "desc"}]
    else:
        sort_by = []

    return df.to_dict("records"), columns, sort_by
