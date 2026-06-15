// Track Meta (Cmd on Mac) / Ctrl key for ⌘-click → add to compare filter.
window._swe_metaKey = false;

var _SWE_SELECT_CHARTS = ['bar-chart', 'polarity-scatter'];

// Per-element cached dragmode so we skip no-op relayout calls.
var _sweDragmodeCache = {};

window._sweSetDragmode = function (mode) {
    _SWE_SELECT_CHARTS.forEach(function (id) {
        var wrapper = document.getElementById(id);
        if (!wrapper || !window.Plotly) return;
        var el = wrapper.querySelector('.js-plotly-plot') || wrapper;
        if (_sweDragmodeCache[id] === mode) return;
        _sweDragmodeCache[id] = mode;
        Plotly.relayout(el, { dragmode: mode });
    });
};

// Sync dragmode whenever cursor enters or moves over a chart.
// This catches: re-renders resetting mode, Cmd already held when entering chart.
document.addEventListener('mousemove', function (e) {
    _SWE_SELECT_CHARTS.forEach(function (id) {
        var wrapper = document.getElementById(id);
        if (wrapper && wrapper.contains(e.target)) {
            var want = (e.metaKey || e.ctrlKey) ? 'select' : 'zoom';
            if (_sweDragmodeCache[id] !== want && window.Plotly) {
                _sweDragmodeCache[id] = want;
                var el = wrapper.querySelector('.js-plotly-plot') || wrapper;
                Plotly.relayout(el, { dragmode: want });
            }
        }
    });
});

document.addEventListener('keydown', function (e) {
    if (e.key === 'Meta' || e.key === 'Control') {
        window._swe_metaKey = true;
        window._sweSetDragmode('select');
    }
});
document.addEventListener('keyup', function (e) {
    if (e.key === 'Meta' || e.key === 'Control') {
        window._swe_metaKey = false;
        window._sweSetDragmode('zoom');
    }
});
window.addEventListener('blur', function () {
    window._swe_metaKey = false;
    window._sweSetDragmode('zoom');
});

// Capture modifier state on mouseup — used by box-select because during a
// Plotly drag the browser may suppress keyup events, making _swe_metaKey
// unreliable. mouseup fires at the exact moment drag ends with the true state.
window._swe_mouseupMeta = false;
document.addEventListener('mouseup', function (e) {
    window._swe_mouseupMeta = e.metaKey || e.ctrlKey || false;
});

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.swe = {
    // Fires when a button in the sidebar Selected list is clicked.
    // prop_id looks like: {"eid":"country_FRA","type":"selected-item-btn"}.n_clicks
    routeSelectedItem: function () {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) {
            return window.dash_clientside.no_update;
        }
        var prop_id = ctx.triggered[0].prop_id;
        var idStr = prop_id.split('.')[0];
        try {
            var idObj = JSON.parse(idStr);
            var entityId = idObj.eid;
            if (entityId) {
                return { entity_id: entityId, meta: window._swe_metaKey || false };
            }
        } catch (e) {}
        return window.dash_clientside.no_update;
    },

    // Fires when user box-selects on polarity scatter (requires Cmd/Ctrl).
    // REPLACES the current filter with the new selection.
    routeSelectedData: function (selectedData, currentFilter) {
        if (!window._swe_mouseupMeta) return window.dash_clientside.no_update;
        if (!selectedData || !selectedData.points || !selectedData.points.length) {
            return window.dash_clientside.no_update;
        }
        var ids = [];
        selectedData.points.forEach(function (p) {
            if (p.customdata && ids.indexOf(p.customdata) === -1) ids.push(p.customdata);
        });
        return ids;
    },

    // Fires when user box-selects on bar chart.
    // REPLACES the current filter with the new selection.
    routeBarSelectedData: function (selectedData, currentFilter) {
        if (!selectedData || !selectedData.points || !selectedData.points.length) {
            return window.dash_clientside.no_update;
        }
        var ids = [];
        selectedData.points.forEach(function (p) {
            // customdata can be a scalar string or wrapped in an array by Plotly
            var raw = p.customdata;
            var id = (raw === null || raw === undefined) ? null
                   : Array.isArray(raw) ? raw[0]
                   : raw;
            if (id && typeof id === 'string' && ids.indexOf(id) === -1) ids.push(id);
        });
        if (!ids.length) return window.dash_clientside.no_update;
        return ids;
    },

    routeClick: function (mapClick, scatterClick, barClick) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || !ctx.triggered.length) {
            return window.dash_clientside.no_update;
        }
        var prop = ctx.triggered[0].prop_id;
        var entityId = null;

        if (prop === 'world-map.clickData' && mapClick && mapClick.points && mapClick.points.length) {
            entityId = mapClick.points[0].customdata;
        } else if (prop === 'polarity-scatter.clickData' && scatterClick && scatterClick.points && scatterClick.points.length) {
            entityId = scatterClick.points[0].customdata;
        } else if (prop === 'bar-chart.clickData' && barClick && barClick.points && barClick.points.length) {
            entityId = barClick.points[0].customdata;
        }

        if (!entityId) return window.dash_clientside.no_update;
        return { entity_id: entityId, meta: window._swe_metaKey || false };
    }
};
