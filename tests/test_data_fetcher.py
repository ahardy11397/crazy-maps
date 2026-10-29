from pathlib import Path
from unittest.mock import patch, MagicMock
import json
from shapely.geometry import box, MultiPolygon, Polygon
from tn_dollar_general.data_fetcher import (
    StoreRecord,
    parse_boundary_geojson,
    parse_and_deduplicate_overpass,
    fetch_dollar_generals,
    fetch_tn_boundary,
)


def test_parse_boundary_handles_feature_collection_and_multipolygon():
    poly1 = box(-86.0, 35.0, -85.0, 36.0)
    poly2 = box(-87.0, 35.0, -86.5, 35.5)
    fc = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"NAME": "Tennessee"},
            "geometry": MultiPolygon([poly1, poly2]).__geo_interface__,
        }],
    }
    geom = parse_boundary_geojson(fc)
    assert geom.contains(box(-85.6, 35.4, -85.4, 35.6))


def test_parse_overpass_extracts_nodes_ways_and_filters_warehouses():
    tn_box = box(-87.0, 35.0, -85.0, 36.5)
    elements = [
        # Node inside TN
        {"type": "node", "id": 1, "lat": 35.5, "lon": -86.0, "tags": {"name": "Dollar General", "addr:city": "Murfreesboro"}},
        # Duplicate way centroid ~15m away from node 1 (should be deduplicated)
        {"type": "way", "id": 2, "center": {"lat": 35.5001, "lon": -86.0001}, "tags": {"name": "Dollar General", "building": "retail"}},
        # Store outside TN (in KY buffer)
        {"type": "node", "id": 3, "lat": 36.8, "lon": -86.0, "tags": {"name": "Dollar General Market", "addr:state": "KY"}},
        # Distribution center (must be excluded)
        {"type": "way", "id": 4, "center": {"lat": 35.8, "lon": -86.2}, "tags": {"name": "Dollar General Distribution Center", "building": "warehouse"}},
        # Industrial warehouse with brand tag (must be excluded)
        {"type": "node", "id": 5, "lat": 35.9, "lon": -86.3, "tags": {"brand": "Dollar General", "industrial": "warehouse"}},
    ]
    stores = parse_and_deduplicate_overpass(elements, tn_box, dedup_distance_m=50.0)
    assert len(stores) == 2
    in_tn_stores = [s for s in stores if s.in_tn]
    out_tn_stores = [s for s in stores if not s.in_tn]
    assert len(in_tn_stores) == 1
    assert in_tn_stores[0].city == "Murfreesboro"
    assert len(out_tn_stores) == 1
    assert out_tn_stores[0].name == "Dollar General Market"


def test_parse_overpass_empty_and_invalid():
    tn_box = box(-87.0, 35.0, -85.0, 36.5)
    assert parse_and_deduplicate_overpass([], tn_box) == []
    bad_elements = [
        {"type": "node", "id": 1, "tags": {"name": "Dollar General"}},
        {"type": "node", "id": 2, "lat": "not_a_num", "lon": -86.0, "tags": {"name": "Dollar General"}},
        {"type": "node", "id": 3, "lat": 35.5, "lon": -86.0, "tags": {"name": "Walmart"}},
    ]
    assert parse_and_deduplicate_overpass(bad_elements, tn_box) == []


def test_parse_overpass_relation_and_wikidata():
    tn_box = box(-87.0, 35.0, -85.0, 36.5)
    elements = [
        {
            "type": "relation",
            "id": 99,
            "center": {"lat": 35.5, "lon": -86.0},
            "tags": {"brand:wikidata": "Q145168", "name": "DG Store"},
        }
    ]
    stores = parse_and_deduplicate_overpass(elements, tn_box)
    assert len(stores) == 1
    assert stores[0].name == "DG Store"
    assert stores[0].in_tn is True


def test_fetch_dollar_generals_falls_back_Across_overpass_endpoints(tmp_path: Path):
    tn_box = box(-87.0, 35.0, -85.0, 36.5)
    cache_file = tmp_path / "dg.geojson"
    fail_resp = MagicMock(status_code=504)
    fail_resp.raise_for_status.side_effect = Exception("Gateway Timeout")
    ok_resp = MagicMock(status_code=200)
    ok_resp.json.return_value = {
        "elements": [{"type": "node", "id": 10, "lat": 35.5, "lon": -86.0, "tags": {"name": "Dollar General"}}]
    }
    with patch("tn_dollar_general.data_fetcher.requests.post", side_effect=[fail_resp, ok_resp]) as mock_post:
        stores = fetch_dollar_generals(tn_box, cache_file, refresh=True)
        assert len(stores) == 1
        assert mock_post.call_count == 2
        assert cache_file.exists()


def test_fetch_dollar_generals_cache_hit(tmp_path: Path):
    tn_box = box(-87.0, 35.0, -85.0, 36.5)
    cache_file = tmp_path / "cached_dg.geojson"
    cache_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-86.0, 35.5]},
                "properties": {
                    "id": "123",
                    "name": "Dollar General Cached",
                    "lat": 35.5,
                    "lon": -86.0,
                    "city": "Nashville",
                    "street": "Broadway",
                    "state": "TN",
                    "in_tn": True,
                },
            }
        ],
    }
    cache_file.write_text(json.dumps(cache_data), encoding="utf-8")
    with patch("tn_dollar_general.data_fetcher.requests.post") as mock_post:
        stores = fetch_dollar_generals(tn_box, cache_file, refresh=False)
        assert len(stores) == 1
        assert stores[0].name == "Dollar General Cached"
        assert stores[0].city == "Nashville"
        assert mock_post.call_count == 0


def test_fetch_tn_boundary_caching_and_download(tmp_path: Path):
    cache_file = tmp_path / "tn_boundary.geojson"
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Kentucky"},
                "geometry": Polygon([(-86, 36.5), (-85, 36.5), (-85, 37.5), (-86, 37.5)]).__geo_interface__,
            },
            {
                "type": "Feature",
                "properties": {"name": "Tennessee"},
                "geometry": Polygon([(-86, 35), (-85, 35), (-85, 36), (-86, 36)]).__geo_interface__,
            },
        ],
    }
    with patch("tn_dollar_general.data_fetcher.requests.get", return_value=mock_resp) as mock_get:
        geom = fetch_tn_boundary(cache_file, refresh=True)
        assert geom.is_valid
        assert mock_get.call_count == 1
        assert cache_file.exists()

    # Second call should use cache
    with patch("tn_dollar_general.data_fetcher.requests.get") as mock_get2:
        cached_geom = fetch_tn_boundary(cache_file, refresh=False)
        assert cached_geom.is_valid
        assert mock_get2.call_count == 0
