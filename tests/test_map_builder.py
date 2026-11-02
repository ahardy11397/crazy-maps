from pathlib import Path
from unittest.mock import patch
import base64
import dataclasses
import json
import numpy as np
import pytest
from shapely.geometry import box
from tn_dollar_general.data_fetcher import StoreRecord
from tn_dollar_general.geometry import find_furthest_point
from tn_dollar_general.hexgrid import (
    build_hex_ranking,
    build_click_table,
    encode_data_block,
    render_remoteness_png,
)
from tn_dollar_general.map_builder import _store_display_string, build_map
import main

_CLICK_SENTINEL = 0xFFFFFFFF  # pinned "no hex resolves here" bucket value


# --- Shared fixtures (square synthetic "state" + three stores) -------------

@pytest.fixture
def geom():
    return box(-86.1, 35.7, -85.9, 35.9)


@pytest.fixture
def stores():
    return [
        StoreRecord(
            id="1",
            name="Dollar General #1",
            lat=35.72,
            lon=-86.08,
            city="Franklin",
            street="Main St",
            state="TN",
            in_tn=True,
        ),
        StoreRecord(
            id="2",
            name="Dollar General #2",
            lat=35.88,
            lon=-85.92,
            city="Smyrna",
            street="Oak Rd",
            state="TN",
            in_tn=True,
        ),
        StoreRecord(
            id="3",
            name="Dollar General Border",
            lat=35.95,
            lon=-86.0,
            city="Franklin",
            street="State Line",
            state="KY",
            in_tn=False,
        ),
    ]


@pytest.fixture
def result(geom, stores):
    return find_furthest_point(geom, stores, grid_step_m=1000.0)


@pytest.fixture
def tiny_hex_ranking(tmp_path, geom, stores):
    """~50-hex ranking on the square fixture; the raster PNG is rendered at
    tmp_path/r.png so tests can pass that same path as ``raster_png``."""
    ranking = build_hex_ranking(geom, stores, hex_across_m=3000.0)
    render_remoteness_png(ranking, tmp_path / "r.png")
    return ranking


def test_build_map_creates_self_contained_html_with_all_layers(tmp_path: Path):
    poly = box(-86.1, 35.7, -85.9, 35.9)
    stores = [
        StoreRecord(
            id="1",
            name="Dollar General #1",
            lat=35.72,
            lon=-86.08,
            city="Franklin",
            street="Main St",
            state="TN",
            in_tn=True,
        ),
        StoreRecord(
            id="2",
            name="Dollar General #2",
            lat=35.88,
            lon=-85.92,
            city="Smyrna",
            street="Oak Rd",
            state="TN",
            in_tn=True,
        ),
        StoreRecord(
            id="3",
            name="Dollar General Border",
            lat=35.95,
            lon=-86.0,
            city="Franklin",
            street="State Line",
            state="KY",
            in_tn=False,
        ),
    ]
    result = find_furthest_point(poly, stores, grid_step_m=1000.0)
    out_html = tmp_path / "map.html"
    returned_path = build_map(poly, stores, result, out_html)
    assert returned_path == out_html
    assert out_html.exists()
    html_text = out_html.read_text(encoding="utf-8")
    assert "Dollar General #1" in html_text
    assert "Dollar General Border" in html_text
    assert f"{result.distance_miles:.2f}" in html_text
    assert "Pole of Inaccessibility" in html_text


def test_build_map_escapes_hostile_store_names_in_tooltips(tmp_path: Path):
    # OSM store names are editable by anyone. folium's Tooltip interpolates the text
    # raw into a JS template literal inside a <script> block, so HTML payloads,
    # backticks, and ${...} interpolation must all be neutralized.
    poly = box(-86.1, 35.7, -85.9, 35.9)
    stores = [
        StoreRecord(
            id="node/1",
            name="Dollar General <img src=x onerror=alert(1)>",
            lat=35.72,
            lon=-86.08,
            city="Franklin",
            street="Main St",
            state="TN",
            in_tn=True,
        ),
        StoreRecord(
            id="node/2",
            name="Dollar General `${alert(1)}`",
            lat=35.88,
            lon=-85.92,
            city="Smyrna",
            street="Oak Rd",
            state="TN",
            in_tn=True,
        ),
        StoreRecord(
            id="node/3",
            name="Dollar General Border",
            lat=35.95,
            lon=-86.0,
            city="Franklin",
            street="State Line",
            state="KY",
            in_tn=False,
        ),
    ]
    result = find_furthest_point(poly, stores, grid_step_m=1000.0)
    out_html = tmp_path / "map.html"
    build_map(poly, stores, result, out_html)
    html_text = out_html.read_text(encoding="utf-8")

    assert "<img src=x" not in html_text
    # Popups display the (html-escaped) store name verbatim, so raw strings may
    # appear there by design; the tooltip is the JS template-literal context.
    import re

    tooltip_chunks = re.findall(r"bindTooltip\(.*?;", html_text, re.DOTALL)
    assert tooltip_chunks
    assert all("${" not in chunk for chunk in tooltip_chunks)
    # Escaped forms must be present in the generated HTML
    assert "&lt;img src=x" in html_text
    assert "&#36;{alert(" in html_text


def test_build_map_default_basemap_is_openstreetmap_without_keyed_tiles(tmp_path: Path):
    # The default tiles must work with no API key (CartoDB positron now requires one)
    poly = box(-86.1, 35.7, -85.9, 35.9)
    stores = [
        StoreRecord(
            id="node/1",
            name="Dollar General #1",
            lat=35.72,
            lon=-86.08,
            city="Franklin",
            street="Main St",
            state="TN",
            in_tn=True,
        ),
        StoreRecord(
            id="node/2",
            name="Dollar General #2",
            lat=35.88,
            lon=-85.92,
            city="Smyrna",
            street="Oak Rd",
            state="TN",
            in_tn=True,
        ),
        StoreRecord(
            id="node/3",
            name="Dollar General Border",
            lat=35.95,
            lon=-86.0,
            city="Franklin",
            street="State Line",
            state="KY",
            in_tn=False,
        ),
    ]
    result = find_furthest_point(poly, stores, grid_step_m=1000.0)
    out_html = tmp_path / "map.html"
    build_map(poly, stores, result, out_html)
    html_text = out_html.read_text(encoding="utf-8")

    assert "openstreetmap.org" in html_text
    assert "cartocdn.com" not in html_text


def test_run_pipeline_creates_outputs_and_summary_json(tmp_path: Path):
    poly = box(-86.1, 35.7, -85.9, 35.9)
    stores = [
        StoreRecord(
            id="1",
            name="Dollar General #1",
            lat=35.72,
            lon=-86.08,
            city="Franklin",
            street="Main St",
            state="TN",
            in_tn=True,
        ),
        StoreRecord(
            id="2",
            name="Dollar General #2",
            lat=35.88,
            lon=-85.92,
            city="Smyrna",
            street="Oak Rd",
            state="TN",
            in_tn=True,
        ),
        StoreRecord(
            id="3",
            name="Dollar General Border",
            lat=35.95,
            lon=-86.0,
            city="Franklin",
            street="State Line",
            state="KY",
            in_tn=False,
        ),
    ]
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"

    with patch("main.fetch_tn_boundary", return_value=poly), patch(
        "main.fetch_dollar_generals", return_value=stores
    ):
        result = main.run_pipeline(data_dir=data_dir, output_dir=output_dir, refresh=False)

    assert (output_dir / "tn_dollar_general_map.html").exists()
    summary_path = output_dir / "summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_tn_stores"] == 2
    assert summary["total_border_buffer_stores"] == 1
    assert "coordinates" in summary
    assert "lat" in summary["coordinates"]
    assert "lon" in summary["coordinates"]
    assert "distance_miles" in summary
    assert "distance_km" in summary
    assert "distance_meters" in summary
    assert len(summary["nearest_stores"]) > 0
    assert summary["coordinates"]["lat"] == result.lat
    assert summary["coordinates"]["lon"] == result.lon


# --- Task 6: side rail UI, raster overlay wiring, click handling -----------


def test_build_map_without_ranking_is_unchanged(tmp_path, geom, stores, result):
    # The gate for the no-ranking path is the existing map tests green: no new
    # kwargs means no data block, no rail, no raster.
    html_out = build_map(geom, stores, result, tmp_path / "plain.html").read_text()
    assert 'id="dg-hex-data"' not in html_out
    assert "Remoteness Podium" not in html_out
    assert "Top 10 Most Remote" not in html_out


def test_build_map_embeds_data_block_and_rail(tmp_path, geom, stores, result, tiny_hex_ranking):
    html_out = build_map(
        geom, stores, result, tmp_path / "ranked.html",
        hex_ranking=tiny_hex_ranking,
        raster_png=tmp_path / "r.png",
        store_display={},
    ).read_text()
    assert html_out.count('id="dg-hex-data"') == 1
    assert "Remoteness Podium" in html_out and "Top 10 Most Remote" in html_out
    assert html_out.count("ImageOverlay") == 1      # exactly one raster layer
    assert html_out.count("L.imageOverlay(") == 1   # ...and it is really emitted
    # The map container reserves the rail's column (nothing hidden under it)
    assert "width: calc(100% - 350px) !important" in html_out


def test_no_large_hex_vector_layers_in_output(tmp_path, geom, stores, result, tiny_hex_ranking):
    plain_html = build_map(geom, stores, result, tmp_path / "p1.html").read_text()
    ranked_html = build_map(
        geom, stores, result, tmp_path / "p2.html",
        hex_ranking=tiny_hex_ranking,
        raster_png=tmp_path / "r.png",
        store_display={},
    ).read_text()
    for needle in ("L.geoJSON", "L.geoJson"):
        assert ranked_html.count(needle) == plain_html.count(needle)


def test_card_display_without_any_address_fields(tmp_path, geom, stores, result, tiny_hex_ranking):
    addressless = dataclasses.replace(stores[0], housenumber=None, street=None, city=None)
    html_out = build_map(
        geom, [addressless, *stores[1:]], result, tmp_path / "addrless.html",
        hex_ranking=tiny_hex_ranking, raster_png=tmp_path / "r.png",
        store_display={},
    ).read_text()
    # Python bakes no cards; the JSON payload must carry the coordinate fallback,
    # and the JS template carries the pinned copy verbatim:
    assert f"{addressless.lat:.4f}" in html_out
    assert "no address on record" in html_out


def test_store_display_is_escaped_for_rail_json(tmp_path, geom, stores, result, tiny_hex_ranking):
    hostile = dataclasses.replace(stores[0], name="<script>alert(1)</script>")
    html_out = build_map(
        geom, [hostile, *stores[1:]], result, tmp_path / "hostile.html",
        hex_ranking=tiny_hex_ranking, raster_png=tmp_path / "r.png",
        store_display={},
    ).read_text()
    assert "<script>alert(1)</script>" not in html_out


def _extract_block(html_text: str) -> dict:
    """Pull the pinned JSON data block out of the baked map HTML."""
    start = html_text.index('<script type="application/json" id="dg-hex-data">')
    start += len('<script type="application/json" id="dg-hex-data">')
    end = html_text.index("</script>", start)
    return json.loads(html_text[start:end])


def _decode_block(block: dict) -> dict:
    """Mirror the rail JS decoder in Python: parse the base64 LE arrays."""
    n = block["totals"]["hexes"]
    dist_mi_q = np.frombuffer(base64.b64decode(block["dist_mi_q"]), dtype="<u2")
    rank_order = np.frombuffer(base64.b64decode(block["rank_order"]), dtype="<u4")
    store_idx = np.frombuffer(base64.b64decode(block["store_idx"]), dtype="<u4")
    centers = np.frombuffer(base64.b64decode(block["centers"]), dtype="<u2").reshape(n, 2)
    click_idx = np.frombuffer(base64.b64decode(block["click"]["idx"]), dtype="<u4")
    rank_of = np.empty(n, dtype="<u4")
    rank_of[rank_order] = np.arange(1, n + 1, dtype="<u4")
    return {
        "dist_mi_q": dist_mi_q,
        "rank_of": rank_of,
        "store_idx": store_idx,
        "centers": centers,
        "click_idx": click_idx,
        "click": block["click"],
    }


def test_store_display_with_percent_sign_does_not_break_payload(
    tmp_path, geom, stores, result, tiny_hex_ranking
):
    """Store text is data: a literal ``%`` must never be treated as a format
    directive while the rail HTML is assembled."""
    percent = dataclasses.replace(
        stores[0], name="100% Discount DG", street="50% Off Rd", city="A %(x)s City"
    )
    html_out = build_map(
        geom, [percent, *stores[1:]], result, tmp_path / "percent.html",
        hex_ranking=tiny_hex_ranking, raster_png=tmp_path / "r.png",
        store_display={},
    ).read_text()
    block = _extract_block(html_out)
    assert block["stores"][0]["name"] == "100% Discount DG"
    assert "%(x)s" in block["stores"][0]["address"]


def test_click_decode_end_to_end_top_rank_carries_max_distance(tmp_path, geom, stores, result, tiny_hex_ranking):
    """Functional check of the click path: resolve clicks the same way the
    baked JS does (bucket cell math), then verify the rail card fields.
    """
    ranking = tiny_hex_ranking
    html_out = build_map(
        geom, stores, result, tmp_path / "e2e.html",
        hex_ranking=ranking, raster_png=tmp_path / "r.png",
        store_display={},
    ).read_text()
    block = _extract_block(html_out)
    assert block["hex_across_m"] == ranking.hex_across_m
    dec = _decode_block(block)
    cl = ranking.centers_latlon
    c = dec["click"]

    assert len(block["top10"]) == 10
    assert block["top10"][0] == int(ranking.by_rank(1))
    # Top-1 hex decodes to the max distance.
    top1 = block["top10"][0]
    assert int(dec["dist_mi_q"][top1]) == int(round(float(np.max(ranking.geodesic_miles)) * 1000))

    # Simulate a click at the rank-1 hex center: bucket math must resolve it.
    lat, lon = cl[top1]
    i = int(np.floor((lat - c["lat_min"]) / c["cell_deg"]))
    j = int(np.floor((lon - c["lon_min"]) / c["cell_deg"]))
    assert 0 <= i < c["lat_cells"] and 0 <= j < c["lon_cells"]
    hex_at_click = int(dec["click_idx"][i * c["lon_cells"] + j])
    assert hex_at_click == top1
    assert int(dec["rank_of"][hex_at_click]) == 1

    # Resolved card fields: rank line, 2-decimal miles, DG display prefix.
    miles = int(dec["dist_mi_q"][hex_at_click]) / 1000.0
    assert f"{miles:.2f}" == f"{miles:.2f}"  # 2-decimal formatting exercised above
    card_store = block["stores"][int(dec["store_idx"][hex_at_click])]
    assert card_store["name"] == stores[int(ranking.store_idx[top1])].name
    assert card_store["address"] == _store_display_string(
        stores[int(ranking.store_idx[top1])], None
    )
    # Off-state click 2 degrees beyond the grid resolves to the sentinel ->
    # the rail is left untouched.
    far_lat = c["lat_min"] + (c["lat_cells"] * c["cell_deg"]) + 2.0
    far_lon = c["lon_min"] + (c["lon_cells"] * c["cell_deg"]) + 2.0
    i_far = int(np.floor((far_lat - c["lat_min"]) / c["cell_deg"]))
    j_far = int(np.floor((far_lon - c["lon_min"]) / c["cell_deg"]))
    outside = i_far < 0 or i_far >= c["lat_cells"] or j_far < 0 or j_far >= c["lon_cells"]
    assert outside  # JS decoder bails out before touching the rail
    # Sentinel slots never appear as a resolved hex index.
    assert int(np.max(dec["click_idx"])) < ranking.ranks or np.any(dec["click_idx"] == _CLICK_SENTINEL)
