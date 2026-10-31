from __future__ import annotations

import html
import logging
from pathlib import Path
import branca.element
import folium
import shapely

from tn_dollar_general.data_fetcher import StoreRecord
from tn_dollar_general.geometry import FurthestPointResult, geodesic_circle_coords

logger = logging.getLogger(__name__)


def _safe_tooltip(text: str) -> str:
    """Sanitize text for use as a folium Tooltip label.

    folium's Tooltip template interpolates the text raw into a JS template
    literal inside a <script> block (and Leaflet inserts the rendered span
    via innerHTML), so HTML payloads, backticks, and ${...} interpolation
    must all be neutralized. Entities are decoded when Leaflet hydrates the
    tooltip, so the visible text is unaffected.
    """
    escaped = html.escape(text, quote=True)
    escaped = escaped.replace("`", "&#96;").replace("$", "&#36;")
    return escaped


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


def build_map(
    tn_geom: shapely.Geometry,
    stores: list[StoreRecord],
    result: FurthestPointResult,
    output_path: Path | str,
) -> Path:
    """Build a self-contained interactive Folium map visualizing Dollar Generals and the furthest point."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Base Map centered on Tennessee (requires no API key)
    m = folium.Map(
        location=[35.86, -86.35],
        zoom_start=7,
        tiles="OpenStreetMap",
        prefer_canvas=True,
    )

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

    # 7. Save to HTML
    m.save(str(output_path))
    logger.info("Saved interactive map to %s", output_path)

    return output_path
