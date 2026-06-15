// Track Meta (Cmd on Mac) / Ctrl key for ⌘-click → add to compare filter.
window._swe_metaKey = false;
document.addEventListener('keydown', function (e) {
    if (e.key === 'Meta' || e.key === 'Control') window._swe_metaKey = true;
});
document.addEventListener('keyup', function (e) {
    if (e.key === 'Meta' || e.key === 'Control') window._swe_metaKey = false;
});
window.addEventListener('blur', function () { window._swe_metaKey = false; });

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

    // Fires when user box-selects on polarity scatter.
    // Only acts when Ctrl/Cmd is held — merges selected entity IDs into current filter.
    routeSelectedData: function (selectedData, currentFilter) {
        if (!window._swe_metaKey) return window.dash_clientside.no_update;
        if (!selectedData || !selectedData.points || !selectedData.points.length) {
            return window.dash_clientside.no_update;
        }
        var current = currentFilter ? currentFilter.slice() : [];
        selectedData.points.forEach(function (p) {
            if (p.customdata && current.indexOf(p.customdata) === -1) {
                current.push(p.customdata);
            }
        });
        return current;
    },

    routeClick: function (mapClick, scatterClick, barClick, activeCell, tableData) {
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
        } else if (prop === 'ranking-table.active_cell' && activeCell && tableData) {
            var row = activeCell.row;
            if (row !== undefined && tableData && row < tableData.length) {
                entityId = tableData[row].id;
            }
        }

        if (!entityId) return window.dash_clientside.no_update;
        return { entity_id: entityId, meta: window._swe_metaKey || false };
    }
};
