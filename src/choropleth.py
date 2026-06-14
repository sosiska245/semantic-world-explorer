"""Color computation for the world-map country choropleth.

Two problems the plain min-max normalization in src/similarity.py has for a
world map:

1. For niche queries, raw cosine similarity has almost no signal outside the
   top few entities (most values cluster near 0, some are negative). Min-max
   stretches that tiny range across the full ramp, making most countries look
   the same.
2. go.Choropleth's z/colorscale is a single continuous gradient - it can't
   natively represent an arbitrary per-country RGB blend for 1-3 active
   slots.

`country_blend_colors` fixes (1) with a relevance floor: a country with
similarity <= 0 on every active channel contributes nothing and renders as
LOW_RELEVANCE_GRAY, regardless of rank. Countries with positive similarity on
a channel get that channel's value rank-normalized over (0, 1] among the
positive entries, so the visible gradient always spans the full range among
entities that actually have signal.

`discrete_colorscale` fixes (2) with the standard step-function trick: each
location gets its own z value and a two-stop colorscale segment, so it maps
to its exact precomputed color with no interpolation.
"""

import numpy as np

from src.config import SLOT_COLORS

LOW_RELEVANCE_GRAY = "rgb(214,214,214)"


def _rank_normalize_positive(sim):
    """1D array -> array in [0,1]: 0 where sim <= 0, else rank/count_positive
    scaled to (0, 1] among the positive entries (order-preserving)."""
    out = np.zeros_like(sim, dtype=float)
    positive = sim > 0
    n_pos = int(positive.sum())
    if n_pos == 0:
        return out
    order = np.argsort(sim[positive])
    ranks = np.empty(n_pos, dtype=float)
    ranks[order] = np.arange(1, n_pos + 1)
    out[positive] = ranks / n_pos
    return out


def country_blend_colors(sims, country_mask):
    """sims: dict color -> raw similarity array (509,) or None, from
    sims_from_store_data. country_mask: boolean array selecting country rows.
    Returns list[str] 'rgb(r,g,b)' (or LOW_RELEVANCE_GRAY), one per country,
    in the same order as country_mask's True entries."""
    n = int(country_mask.sum())
    channels = {}
    for color in SLOT_COLORS:
        sim = sims.get(color)
        channels[color] = _rank_normalize_positive(sim[country_mask]) if sim is not None else np.zeros(n)

    r = (channels["R"] * 255).astype(int)
    g = (channels["G"] * 255).astype(int)
    b = (channels["B"] * 255).astype(int)

    colors = []
    for ri, gi, bi in zip(r, g, b):
        if ri == 0 and gi == 0 and bi == 0:
            colors.append(LOW_RELEVANCE_GRAY)
        else:
            colors.append(f"rgb({ri},{gi},{bi})")
    return colors


def discrete_colorscale(colors):
    """colors: list[str] 'rgb()', one per location in `locations` order.
    Returns (z, zmin, zmax, colorscale) for go.Choropleth such that location
    i's z value maps to exactly colors[i]."""
    n = len(colors)
    z = [i + 0.5 for i in range(n)]
    colorscale = []
    for i, c in enumerate(colors):
        colorscale.append([i / n, c])
        colorscale.append([(i + 1) / n, c])
    return z, 0, n, colorscale
