from __future__ import annotations

import base64
import html
import json
import logging
from pathlib import Path

import branca.element
import folium
import shapely
from folium import raster_layers

from tn_dollar_general.data_fetcher import StoreRecord
from tn_dollar_general.geometry import FurthestPointResult, geodesic_circle_coords
from tn_dollar_general.hexgrid import (
    HexRanking,
    build_click_table,
    encode_data_block,
    raster_image_bounds,
)

logger = logging.getLogger(__name__)

# Rail reserves its own column: map encoded at 100% - 350px, rail fixed at
# 340px on the right + 10px gap, so no part of the map hides under the rail.
_RAIL_WIDTH_PX = 340
_RAIL_RESERVED_PX = 350


def _safe_tooltip(text: str) -> str:
    """Sanitize text for use as a folium Tooltip label.

    folium's Tooltip template interpolates the text raw into a JS template
    literal inside a <script> block (and Leaflet inserts the rendered span
    via innerHTML), so HTML payloads, backticks, and ${...} interpolation
    must all be neutralized. Entities are decoded when Leaflet hydrates the
    tooltip, so the visible text is unaffected. Also the shared implementation
    behind :func:`_rail_escape`.
    """
    escaped = html.escape(text, quote=True)
    escaped = escaped.replace("`", "&#96;").replace("$", "&#36;")
    return escaped


def _rail_escape(text: str) -> str:
    """Sanitize text for the rail's JSON payload / innerHTML rendering.

    Single shared escaping implementation with :func:`_safe_tooltip`: both
    contexts inject text into inline HTML (folium tooltips and the rail's
    ``innerHTML``), so HTML payloads, backticks and ``${...}`` interpolation
    must all be neutralized. Entities are decoded back to visible text at the
    call sites, so the display is unaffected.
    """
    return _safe_tooltip(text)


def _bearing_to_compass(bearing: float) -> str:
    """Convert bearing degrees (0-360) to 16-point compass heading."""
    points = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW",
    ]
    idx = round(bearing / 22.5) % 16
    return points[idx]


def _build_store_popup(
    store: StoreRecord,
    is_constraining: bool = False,
    distance_miles: float | None = None,
    distance_km: float | None = None,
    bearing: float | None = None,
) -> str:
    name_esc = html.escape(store.name or "Dollar General")
    street_esc = html.escape(store.street) if store.street else None
    city_esc = html.escape(store.city) if store.city else None
    state_esc = html.escape(store.state) if store.state else ("TN" if store.in_tn else "Border")
    id_esc = html.escape(store.id)

    addr_parts = []
    if street_esc:
        addr_parts.append(street_esc)
    if city_esc:
        addr_parts.append(city_esc)
    addr_parts.append(state_esc)
    addr_line = ", ".join(addr_parts)

    osm_url = (
        f"https://www.openstreetmap.org/{id_esc.replace(':', '/')}"
        if ":" in id_esc
        else f"https://www.openstreetmap.org/{id_esc}"
    )

    badge_html = ""
    if is_constraining and distance_miles is not None:
        bearing_str = f" ({_bearing_to_compass(bearing)})" if bearing is not None else ""
        badge_html = f"""
        <div style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 6px 8px; margin: 6px 0; border-radius: 2px;">
            <span style="font-weight: 700; color: #b91c1c; font-size: 11px; text-transform: uppercase;">Constraining Store</span><br>
            <span style="font-size: 12px; color: #7f1d1d;">Distance: <b>{distance_miles:.2f} mi</b> ({distance_km:.2f} km){bearing_str}</span>
        </div>
        """

    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; line-height: 1.4; color: #1e293b; min-width: 200px;">
        <h4 style="margin: 0 0 4px 0; font-size: 14px; font-weight: 700; color: #0f172a;">{name_esc}</h4>
        <div style="color: #475569; margin-bottom: 4px;">{addr_line}</div>
        {badge_html}
        <div style="font-size: 11px; color: #64748b; margin-top: 4px;">
            <b>Coordinates:</b> {store.lat:.5f}, {store.lon:.5f}<br>
            <a href="{osm_url}" target="_blank" rel="noopener noreferrer" style="color: #2563eb; text-decoration: underline;">View on OpenStreetMap</a>
        </div>
    </div>
    """


def _build_furthest_popup(result: FurthestPointResult) -> str:
    nearest_html = ""
    for ns in result.nearest_stores[:3]:
        s = ns.store
        loc = f"{s.city}, {s.state}" if s.city and s.state else (s.city or s.state or "Store")
        compass = _bearing_to_compass(ns.bearing_degrees)
        nearest_html += f"""
        <li style="margin-bottom: 4px;">
            <b>{html.escape(s.name)}</b> ({html.escape(loc)})<br>
            <span style="color: #64748b; font-size: 11px;">{ns.distance_miles:.2f} mi ({ns.distance_km:.2f} km) &bull; Bearing: {ns.bearing_degrees:.1f}° ({compass})</span>
        </li>
        """

    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; line-height: 1.4; color: #1e293b; min-width: 260px;">
        <h3 style="margin: 0 0 4px 0; font-size: 15px; font-weight: 700; color: #dc2626;">
            Dollar General Pole of Inaccessibility
        </h3>
        <p style="margin: 0 0 8px 0; color: #475569; font-size: 11px;">
            The exact geographic point in Tennessee furthest from any Dollar General.
        </p>
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; margin-bottom: 8px;">
            <div style="font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 2px;">
                {result.distance_miles:.2f} miles ({result.distance_km:.2f} km)
            </div>
            <div style="font-size: 11px; color: #64748b;">
                Radius of largest empty circle: {result.distance_meters:,.0f} meters
            </div>
            <div style="font-size: 11px; color: #475569; margin-top: 4px;">
                <b>Coordinates:</b> {result.lat:.6f}° N, {abs(result.lon):.6f}° W
            </div>
        </div>
        <div style="font-weight: 600; font-size: 11px; text-transform: uppercase; color: #64748b; margin-bottom: 4px;">
            Constraining Nearest Stores
        </div>
        <ul style="margin: 0 0 0 16px; padding: 0; font-size: 11px;">
            {nearest_html}
        </ul>
    </div>
    """


def _build_summary_card_html(total_tn: int, total_border: int, result: FurthestPointResult) -> str:
    nearest_items = ""
    for idx, ns in enumerate(result.nearest_stores[:3], 1):
        s = ns.store
        loc = f"{s.city}, {s.state}" if s.city and s.state else (s.city or s.state or "")
        compass = _bearing_to_compass(ns.bearing_degrees)
        nearest_items += f"""
        <div style="margin-bottom: 6px; font-size: 11px; line-height: 1.3;">
            <div style="font-weight: 600; color: #1e293b;">
                {idx}. {html.escape(s.name)} {f'<span style=\"color: #64748b; font-weight: 400;\">({html.escape(loc)})</span>' if loc else ''}
            </div>
            <div style="color: #64748b;">
                {ns.distance_miles:.2f} mi ({ns.distance_km:.2f} km) &bull; {ns.bearing_degrees:.0f}° {compass}
            </div>
        </div>
        """

    return f"""
    <div id="dg-summary-card" style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.96);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 16px 18px;
        border-radius: 12px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        max-width: 340px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        color: #1e293b;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <div>
                <h3 style="margin: 0; font-size: 14px; font-weight: 800; color: #0f172a; text-transform: uppercase; letter-spacing: 0.5px;">
                    Tennessee Dollar General
                </h3>
                <div style="font-size: 11px; font-weight: 700; color: #dc2626; text-transform: uppercase;">
                    Pole of Inaccessibility
                </div>
            </div>
            <span style="background: #fef08a; color: #854d0e; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 9999px; white-space: nowrap; border: 1px solid #facc15;">
                {total_tn:,} Stores in TN
            </span>
        </div>

        <div style="background: #fef2f2; border: 1px solid #fee2e2; border-radius: 8px; padding: 10px; margin-bottom: 10px;">
            <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #991b1b; margin-bottom: 2px;">
                Furthest Distance From Any DG
            </div>
            <div style="font-size: 22px; font-weight: 800; color: #b91c1c; line-height: 1.1;">
                {result.distance_miles:.2f} <span style="font-size: 14px; font-weight: 600;">miles</span>
            </div>
            <div style="font-size: 12px; color: #7f1d1d; margin-top: 2px;">
                {result.distance_km:.2f} km ({result.distance_meters:,.0f} meters)
            </div>
        </div>

        <div style="font-size: 11px; color: #475569; margin-bottom: 10px; line-height: 1.4;">
            <div><b>Coordinates:</b> {result.lat:.5f}° N, {abs(result.lon):.5f}° W</div>
            {f'<div><b>Border buffer stores analyzed:</b> {total_border}</div>' if total_border else ''}
        </div>

        <div style="border-top: 1px solid #e2e8f0; padding-top: 8px;">
            <div style="font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;">
                Nearest Constraining Stores
            </div>
            {nearest_items}
        </div>
    </div>
    """


def _store_display_string(store: StoreRecord, address: str | None) -> str:
    """Rail display string for one store, in the pinned composition order.

    ``address`` (the Task 2 enrichment value) wins when not None/empty; else
    the store's own ``housenumber+street, city, state`` triple ("TN" for
    in-TN stores without a state); else the plain lat/lon coordinate pair.
    A store with no street part AND no city falls straight to the coordinate
    pair even when a (possibly defaulted) state is present — the triple would
    render a bare "TN" and spec §5.2 wants an addressless card to show
    coordinates. The returned string is NOT escaped yet — callers route it
    through :func:`_rail_escape` before it reaches the payload.
    """
    if address:
        return address
    first_part = " ".join(part for part in (store.housenumber, store.street) if part)
    if not first_part and not store.city:
        # Bare state-only composition ("TN") is not a display address.
        return f"{store.lat:.5f}, {store.lon:.5f}"
    state = store.state or ("TN" if store.in_tn else None)
    display = ", ".join(part for part in (first_part, store.city, state) if part)
    if not display:
        display = f"{store.lat:.5f}, {store.lon:.5f}"
    return display


def _add_raster_layer(m: folium.Map, hex_ranking: HexRanking, raster_png: Path | str) -> None:
    """Bake exactly ONE raster: the remoteness PNG as a data: URI overlay.

    Added first, before every other layer, so all later layers render above
    it. Bounds come from :func:`tn_dollar_general.hexgrid.raster_image_bounds`
    — the same registration math :func:`render_remoteness_png` uses — in
    Leaflet order ``[[lat_min, lon_min], [lat_max, lon_max]]``.
    """
    png_bytes = Path(raster_png).read_bytes()
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    lat_min, lon_min, lat_max, lon_max = raster_image_bounds(
        hex_ranking.centers_m, hex_ranking.hex_across_m
    )
    marker = (
        "<!-- Remoteness hex field baked as exactly one folium.raster_layers."
        "ImageOverlay (the pinned anti-goal: never per-hex vector layers) -->\n"
    )
    m.get_root().html.add_child(branca.element.Element(marker))
    raster_layers.ImageOverlay(
        image=data_uri,
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        name="Remoteness Raster",
        interactive=False,
    ).add_to(m)


_RAIL_CSS_TEMPLATE = """
<style>
/* Side rail reserves its own column beside the map: the map container is
   narrowed so the fixed RAIL (%(rail_px)s px) sits in reserved space and no
   map area is hidden under it. The floating summary card (bottom-left) is
   untouched and stays visible. */
#%(map_id)s {
    width: calc(100%% - %(reserved_px)spx) !important;
    margin-right: %(reserved_px)spx;
}
#dg-rail {
    position: fixed;
    top: 0;
    right: 0;
    width: %(rail_px)spx;
    height: 100%%;
    z-index: 9000;
    background: #f8fafc;
    border-left: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: -4px 0 12px -6px rgba(0, 0, 0, 0.25);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #1e293b;
    overflow-y: auto;
}
#dg-rail-header {
    padding: 14px 16px 10px 16px;
    border-bottom: 1px solid #e2e8f0;
}
#dg-rail-header h2 {
    margin: 0;
    font-size: 14px;
    font-weight: 800;
    color: #0f172a;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
#dg-rail-header p {
    margin: 2px 0 0 0;
    font-size: 11px;
    color: #64748b;
}
#dg-rail-tabs {
    display: flex;
    border-bottom: 1px solid #e2e8f0;
    background: #fff;
}
.dg-tab-btn {
    flex: 1;
    border: none;
    background: transparent;
    padding: 9px 6px;
    font-size: 11.5px;
    font-weight: 700;
    color: #64748b;
    cursor: pointer;
    border-bottom: 2px solid transparent;
}
.dg-tab-btn-active {
    color: #b91c1c;
    border-bottom-color: #dc2626;
    background: #fef2f2;
}
#dg-tab-podium, #dg-tab-top10 {
    padding: 12px 16px;
    font-size: 12px;
}
.dg-card-hint { color: #64748b; font-size: 11.5px; line-height: 1.4; }
.dg-rail-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #dc2626;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 10px;
}
.dg-card-rank { font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; }
.dg-card-miles { font-size: 22px; font-weight: 800; color: #b91c1c; line-height: 1.2; }
.dg-card-miles .dg-miles-unit { font-size: 13px; font-weight: 600; color: #64748b; }
.dg-card-store { font-size: 12.5px; font-weight: 600; color: #0f172a; margin-top: 4px; }
.dg-card-addr { font-size: 11px; color: #475569; margin-top: 2px; line-height: 1.35; }
/* Selected hexagon card: emphasized against its rank-neighbors. */
.dg-rail-card.dg-rail-card-selected {
    border-left-width: 7px;
    border-color: #dc2626;
    background: #fff7f7;
    box-shadow: 0 6px 14px -6px rgba(220, 38, 38, 0.55);
}
.dg-top10-list { list-style: none; margin: 0; padding: 0; }
.dg-top10-item {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 6px;
    cursor: pointer;
    font-size: 12px;
    color: #1e293b;
}
.dg-top10-item:hover { border-color: #dc2626; background: #fef2f2; }
.dg-top10-item b { color: #b91c1c; }
</style>
"""

_RAIL_HTML_TEMPLATE = """
<div id="dg-rail">
    <div id="dg-rail-header">
        <h2>Tennessee Hex Remoteness</h2>
        <p>Distance from every TN hexagon center to the nearest Dollar General.</p>
    </div>
    <div id="dg-rail-tabs">
        <button id="dg-tab-btn-podium" class="dg-tab-btn dg-tab-btn-active" type="button">Remoteness Podium</button>
        <button id="dg-tab-btn-top10" class="dg-tab-btn" type="button">Top 10 Most Remote</button>
    </div>
    <div id="dg-tab-podium">
        <div class="dg-card-hint">Click any hexagon on the map to inspect its remoteness rank and nearest store.</div>
    </div>
    <div id="dg-tab-top10" style="display:none;"></div>
</div>
"""

_RAIL_JS_TEMPLATE = """
<script>
(function () {
    'use strict';
    window.addEventListener('load', function () {
        var blockEl = document.getElementById('dg-hex-data');
        if (!blockEl) { return; }
        var data;
        try { data = JSON.parse(blockEl.textContent); }
        catch (err) { return; }
        var map = window['__MAP_ID__'];
        if (!map || typeof L === 'undefined') { return; }

        var NHexes = data.totals.hexes;
        var SENTINEL = 4294967295;

        function b64ToU8(b64) {
            var bin = atob(b64);
            var out = new Uint8Array(bin.length);
            for (var p = 0; p < bin.length; p++) { out[p] = bin.charCodeAt(p); }
            return out;
        }
        function u32List(b64) {
            var dv = new DataView(b64ToU8(b64).buffer);
            var arr = new Array(dv.byteLength / 4);
            for (var p = 0; p < arr.length; p++) { arr[p] = dv.getUint32(p * 4, true); }
            return arr;
        }
        function u16List(b64) {
            var dv = new DataView(b64ToU8(b64).buffer);
            var arr = new Array(dv.byteLength / 2);
            for (var p = 0; p < arr.length; p++) { arr[p] = dv.getUint16(p * 2, true); }
            return arr;
        }

        var rankOrder = u32List(data.rank_order);
        var rankOf = new Array(NHexes);
        for (var r = 0; r < NHexes; r++) { rankOf[rankOrder[r]] = r + 1; }
        var storeIdx = u32List(data.store_idx);
        var distMiQ = u16List(data.dist_mi_q);
        var centersQ = u16List(data.centers);
        var click = data.click;
        var clip = data.clip;
        // Decoded once at load: the click handler must not re-decode per click.
        var clickIdx = u32List(click.idx);

        function centerLatLon(h) {
            return [
                clip.lat[0] + centersQ[h * 2] / 65535.0 * (clip.lat[1] - clip.lat[0]),
                clip.lon[0] + centersQ[h * 2 + 1] / 65535.0 * (clip.lon[1] - clip.lon[0]),
            ];
        }

        var elPodium = document.getElementById('dg-tab-podium');
        var elTop10 = document.getElementById('dg-tab-top10');
        var btnPodium = document.getElementById('dg-tab-btn-podium');
        var btnTop10 = document.getElementById('dg-tab-btn-top10');
        if (!elPodium || !elTop10 || !btnPodium || !btnTop10) { return; }

        function switchTab(which) {
            var isPodium = which === 'podium';
            elPodium.style.display = isPodium ? '' : 'none';
            elTop10.style.display = isPodium ? 'none' : '';
            btnPodium.className = isPodium ? 'dg-tab-btn dg-tab-btn-active' : 'dg-tab-btn';
            btnTop10.className = isPodium ? 'dg-tab-btn' : 'dg-tab-btn dg-tab-btn-active';
        }
        btnPodium.addEventListener('click', function () { switchTab('podium'); });
        btnTop10.addEventListener('click', function () { switchTab('top10'); });

        // One ephemeral click highlight: a single dashed circle at the hex
        // center, replaced on every click (never accumulated).
        var highlight = null;
        function pulse(latlon, doFly) {
            var radius = (data.hex_across_m || 1000) / 2 + 100;
            if (highlight) { map.removeLayer(highlight); }
            highlight = L.circle(latlon, {
                radius: radius,
                color: '#ef4444',
                weight: 2.5,
                dashArray: '6, 6',
                fill: false,
                interactive: false,
            });
            highlight.addTo(map);
            if (doFly) { map.flyTo(latlon, Math.max(map.getZoom(), 11)); }
        }

        // Store text arrives in the data block already escaped by the builder;
        // it is concatenated (never through a template literal) and inserted
        // via innerHTML so the browser decodes entities to visible text.
        function cardHtml(h, isSelected) {
            var s = data.stores[storeIdx[h]];
            var nm = s && s.name ? s.name : 'Dollar General';
            var addr = s && s.address ? s.address : '';
            var addrLine = addr ? addr : 'no address on record';
            return '<div class="dg-rail-card'
                + (isSelected ? ' dg-rail-card-selected' : '') + '">'
                + '<div class="dg-card-rank">#' + rankOf[h] + ' of ' + NHexes + ' hexes</div>'
                + '<div class="dg-card-miles">' + (distMiQ[h] / 1000).toFixed(2)
                + ' <span class="dg-miles-unit">mi</span></div>'
                + '<div class="dg-card-store">DG — ' + nm + '</div>'
                + '<div class="dg-card-addr">' + addrLine + '</div>'
                + '</div>';
        }

        function showHexCard(h, doFly) {
            switchTab('podium');
            // Podium: the selected hexagon plus its one-better and one-worse
            // rank-neighbors (spec §2). rankOrder[r] is the hex at rank r+1,
            // so rank k-1 sits at index k-2 and rank k+1 at index k. End ranks
            // omit the missing neighbor entirely — never #0 or #N+1.
            var k = rankOf[h];
            var cards = [];
            if (k > 1) { cards.push(cardHtml(rankOrder[k - 2], false)); }
            cards.push(cardHtml(h, true));
            if (k < NHexes) { cards.push(cardHtml(rankOrder[k], false)); }
            elPodium.innerHTML = cards.join('');
            pulse(centerLatLon(h), doFly);
        }

        map.on('click', function (e) {
            var i = Math.floor((e.latlng.lat - click.lat_min) / click.cell_deg);
            var j = Math.floor((e.latlng.lng - click.lon_min) / click.cell_deg);
            if (i < 0 || i >= click.lat_cells || j < 0 || j >= click.lon_cells) { return; }
            var h = clickIdx[i * click.lon_cells + j];
            if (h === SENTINEL) { return; }  // off-state: leave the rail untouched
            showHexCard(h, false);
        });

        var listHtml = '';
        for (var t = 0; t < data.top10.length; t++) {
            var hex = data.top10[t];
            if (hex === SENTINEL) { continue; }
            var s = data.stores[storeIdx[hex]];
            var nm = s && s.name ? s.name : 'Dollar General';
            listHtml += '<li class="dg-top10-item" data-hex="' + hex + '"><b>#' + (t + 1)
                + '</b> ' + (distMiQ[hex] / 1000).toFixed(2) + ' mi — DG — ' + nm + '</li>';
        }
        elTop10.innerHTML = '<ul class="dg-top10-list">' + listHtml + '</ul>';

        var items = elTop10.querySelectorAll('.dg-top10-item');
        for (var a = 0; a < items.length; a++) {
            (function (li) {
                li.addEventListener('click', function () {
                    var hex = parseInt(li.getAttribute('data-hex'), 10);
                    showHexCard(hex, true);  // fly to the quantized center, then pulse
                });
            })(items[a]);
        }

        // The CSS narrows the map container after Leaflet sized itself on
        // load; re-measure once so tiles/controls match the reserved column.
        map.invalidateSize();
    });
})();
</script>
"""


def _add_rail_and_data_block(
    m: folium.Map,
    hex_ranking: HexRanking,
    stores: list[StoreRecord],
    store_display: dict[str, str] | None,
    tn_geom: shapely.Geometry,
) -> None:
    """Bake the pinned JSON data block + side rail UI/click JS onto the map.

    ``tn_geom`` (the same state polygon the ranking was built on) is passed
    to :func:`build_click_table` so bucket cells off-state carry the
    sentinel: a click inside the state snaps to the nearest ranked hex
    (border-gap snap), a click entirely off-state leaves the rail untouched
    (spec §5.3).

    All store text (name, id, display address) passes through
    :func:`_rail_escape` before serialization: the payload is the only source,
    and the JS re-renders it via innerHTML, so escaping here is the XSS gate.
    """
    map_id = m.get_name()
    stores = list(stores)

    click_table = build_click_table(hex_ranking, state_geom=tn_geom)
    display_map = {
        s.id: _rail_escape(_store_display_string(s, (store_display or {}).get(s.id)))
        for s in stores
    }
    block = encode_data_block(hex_ranking, click_table, display_map, stores)
    for s, entry in zip(stores, block["stores"]):
        entry["id"] = _rail_escape(str(s.id))
        entry["name"] = _rail_escape(s.name or "Dollar General")
        entry["address"] = display_map[s.id]
    # Meta extension consumed by the JS highlight radius; adding keys is
    # within the pinned schema (the Task 4 base64 layout is untouched).
    block["hex_across_m"] = float(hex_ranking.hex_across_m)
    payload = json.dumps(block, separators=(",", ":"), sort_keys=True)

    data_script = (
        '<script type="application/json" id="dg-hex-data">' + payload + "</script>"
    )

    # Format ONLY the templates. The JSON payload is data, not a format
    # string: building it into the "%"-formatted chunk would crash (or corrupt)
    # on any store name/address containing a percent sign.
    rail_html = (
        (_RAIL_CSS_TEMPLATE + _RAIL_HTML_TEMPLATE)
        % {
            "map_id": map_id,
            "rail_px": _RAIL_WIDTH_PX,
            "reserved_px": _RAIL_RESERVED_PX,
        }
        + data_script
    )

    rail_js = _RAIL_JS_TEMPLATE.replace("__MAP_ID__", map_id)

    m.get_root().html.add_child(branca.element.Element(rail_html))
    m.get_root().html.add_child(branca.element.Element(rail_js))


def build_map(
    tn_geom: shapely.Geometry,
    stores: list[StoreRecord],
    result: FurthestPointResult,
    output_path: Path | str,
    hex_ranking: HexRanking | None = None,
    raster_png: Path | str | None = None,
    store_display: dict[str, str] | None = None,
) -> Path:
    """Build a self-contained interactive Folium map visualizing Dollar Generals and the furthest point.

    When ``hex_ranking`` is given (with the corresponding ``raster_png`` from
    :func:`tn_dollar_general.hexgrid.render_remoteness_png`), the map also
    carries: one baked remoteness raster overlay, the pinned side-rail UI
    (podium/tab cards + top-10 jump list), and the click-handling data block.
    When ``hex_ranking`` is None the output is exactly the pre-ranking map:
    no raster, no rail, no data block.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Base Map centered on Tennessee (requires no API key)
    m = folium.Map(
        location=[35.86, -86.35],
        zoom_start=7,
        tiles="OpenStreetMap",
        prefer_canvas=True,
    )

    # 0. Single baked raster overlay, added FIRST so every later layer stays
    # above it in the same Leaflet pane.
    if hex_ranking is not None:
        if raster_png is None:
            raise ValueError("raster_png is required when hex_ranking is provided")
        _add_raster_layer(m, hex_ranking, raster_png)

    # Alternative Basemaps
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri.WorldImagery",
        overlay=False,
        control=True,
    ).add_to(m)

    # 2. Tennessee State Boundary
    folium.GeoJson(
        tn_geom,
        name="Tennessee State Boundary",
        style_function=lambda x: {
            "color": "#1e293b",
            "weight": 2.5,
            "fillColor": "#334155",
            "fillOpacity": 0.04,
        },
    ).add_to(m)

    # Fast lookup for constraining stores
    constraining_lookup = {
        ns.store.id: ns for ns in result.nearest_stores[:3]
    }

    # 3. Tennessee Dollar Generals Layer
    fg_tn = folium.FeatureGroup(name="Tennessee Dollar Generals (All Markers)", show=True)
    fg_border = folium.FeatureGroup(name="Border State Dollar Generals (Buffer)", show=False)

    total_tn = 0
    total_border = 0

    for s in stores:
        ns_info = constraining_lookup.get(s.id)
        is_constraining = ns_info is not None
        dist_mi = ns_info.distance_miles if ns_info else None
        dist_km = ns_info.distance_km if ns_info else None
        bearing = ns_info.bearing_degrees if ns_info else None

        popup_html = _build_store_popup(
            s,
            is_constraining=is_constraining,
            distance_miles=dist_mi,
            distance_km=dist_km,
            bearing=bearing,
        )
        popup = folium.Popup(popup_html, max_width=320)

        if s.in_tn:
            total_tn += 1
            folium.CircleMarker(
                location=[s.lat, s.lon],
                radius=4,
                color="#111111",
                weight=1,
                fill=True,
                fill_color="#f5c518",
                fill_opacity=0.9,
                popup=popup,
                tooltip=_safe_tooltip(s.name),
            ).add_to(fg_tn)
        else:
            total_border += 1
            folium.CircleMarker(
                location=[s.lat, s.lon],
                radius=4,
                color="#475569",
                weight=1,
                fill=True,
                fill_color="#94a3b8",
                fill_opacity=0.7,
                popup=popup,
                tooltip=_safe_tooltip(f"{s.name} (Border)"),
            ).add_to(fg_border)

    fg_tn.add_to(m)
    fg_border.add_to(m)

    # 4. Furthest Point & Empty Circle Feature Group
    fg_furthest = folium.FeatureGroup(name="Furthest Point from a Dollar General", show=True)

    # 4a. Geodesic empty circle
    circle_coords = geodesic_circle_coords(result.lat, result.lon, result.distance_meters)
    folium.Polygon(
        locations=circle_coords,
        color="#dc2626",
        weight=2.5,
        fill=True,
        fill_color="#ef4444",
        fill_opacity=0.15,
        popup=f"Empty Circle (radius: {result.distance_miles:.2f} mi / {result.distance_km:.2f} km)",
    ).add_to(fg_furthest)

    # 4b. Dashed lines from furthest point to nearest constraining stores + highlight markers
    for ns in result.nearest_stores[:3]:
        s = ns.store
        folium.PolyLine(
            locations=[[result.lat, result.lon], [s.lat, s.lon]],
            color="#dc2626",
            weight=2.5,
            dash_array="6, 6",
            opacity=0.9,
            popup=f"Distance to {html.escape(s.name)}: {ns.distance_miles:.2f} mi ({ns.distance_km:.2f} km)",
        ).add_to(fg_furthest)

        # Highlight ring around constraining store
        folium.CircleMarker(
            location=[s.lat, s.lon],
            radius=8,
            color="#dc2626",
            weight=3,
            fill=True,
            fill_color="#f5c518",
            fill_opacity=1.0,
            popup=folium.Popup(
                _build_store_popup(
                    s,
                    is_constraining=True,
                    distance_miles=ns.distance_miles,
                    distance_km=ns.distance_km,
                    bearing=ns.bearing_degrees,
                ),
                max_width=320,
            ),
            tooltip=_safe_tooltip(f"Constraining Store: {s.name} ({ns.distance_miles:.2f} mi)"),
        ).add_to(fg_furthest)

    # 4c. Prominent target marker at furthest point
    furthest_popup = folium.Popup(_build_furthest_popup(result), max_width=340)
    folium.Marker(
        location=[result.lat, result.lon],
        icon=folium.Icon(color="red", icon="bullseye", prefix="fa"),
        popup=furthest_popup,
        tooltip=f"Dollar General Pole of Inaccessibility ({result.distance_miles:.2f} mi)",
    ).add_to(fg_furthest)

    fg_furthest.add_to(m)

    # 5. Layer Control
    folium.LayerControl(collapsed=False).add_to(m)

    # 6. Floating summary card
    card_html = _build_summary_card_html(total_tn, total_border, result)
    m.get_root().html.add_child(branca.element.Element(card_html))

    # 7. Remoteness side rail + baked data block + click JS (ranked builds only)
    if hex_ranking is not None:
        _add_rail_and_data_block(m, hex_ranking, stores, store_display, tn_geom)

    # 8. Save to HTML
    m.save(str(output_path))
    logger.info("Saved interactive map to %s", output_path)

    return output_path
