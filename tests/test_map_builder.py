from pathlib import Path
from unittest.mock import patch
import json
from shapely.geometry import box
from tn_dollar_general.data_fetcher import StoreRecord
from tn_dollar_general.geometry import find_furthest_point
from tn_dollar_general.map_builder import build_map
import main


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
