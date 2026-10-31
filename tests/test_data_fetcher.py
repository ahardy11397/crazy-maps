from pathlib import Path
from unittest.mock import patch, MagicMock
import json
from shapely.geometry import box, MultiPolygon, Polygon
from tn_dollar_general.data_fetcher import (
    StoreRecord,
    TN_BOUNDARY_URLS,
    parse_boundary_geojson,
    parse_and_deduplicate_overpass,
    fetch_dollar_generals,
    fetch_tn_boundary,
)


def test_tn_boundary_urls_lead_with_census_500k():
    # Census 1:500k boundary is the high-resolution source required by the spec;
    # it must be tried first, and the dead 'wiki/files' URL path must not remain.
    assert "eric.clst.org" in TN_BOUNDARY_URLS[0]
    assert "gz_2010_us_040_00_500k.json" in TN_BOUNDARY_URLS[0]
    assert all("wiki/files" not in url for url in TN_BOUNDARY_URLS)


def square_geom_fixture():
    # Same tiny square TN geometry used inline by the neighboring parse tests.
    return box(-87.0, 35.0, -85.0, 36.5)


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


def test_parse_assigns_typed_osm_ids_for_links():
    # IDs must carry their OSM element type so popup links resolve on openstreetmap.org
    tn_box = box(-87.0, 35.0, -85.0, 36.5)
    elements = [
        {"type": "node", "id": 7, "lat": 35.6, "lon": -86.1, "tags": {"name": "Dollar General"}},
        {"type": "way", "id": 42, "center": {"lat": 35.2, "lon": -85.7}, "tags": {"brand": "Dollar General"}},
        {
            "type": "relation",
            "id": 99,
            "center": {"lat": 36.0, "lon": -85.4},
            "tags": {"brand:wikidata": "Q145168", "name": "Dollar General"},
        },
    ]
    stores = parse_and_deduplicate_overpass(elements, tn_box)
    assert len(stores) == 3
    assert {s.id for s in stores} == {"node/7", "way/42", "relation/99"}


def test_dedup_radius_is_50_meters_not_feet():
    # Spec says deduplication within 50 m; EPSG:6576 was US feet (50 ft = 15 m),
    # so two representations ~30 m apart must collapse into one store.
    tn_box = box(-87.0, 35.0, -85.0, 36.5)
    elements = [
        {"type": "node", "id": 1, "lat": 35.5, "lon": -86.0, "tags": {"name": "Dollar General", "addr:city": "M"}},
        {"type": "way", "id": 2, "center": {"lat": 35.5, "lon": -86.00035}, "tags": {"name": "Dollar General"}},
    ]
    stores = parse_and_deduplicate_overpass(elements, tn_box, dedup_distance_m=50.0)
    assert len(stores) == 1


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


def test_parse_captures_housenumber_and_postcode():
    elements = [{
        "type": "node", "id": 10, "lat": 35.0, "lon": -86.0,
        "tags": {"name": "Dollar General", "addr:street": "Main St",
                 "addr:housenumber": "707", "addr:postcode": "38104"},
    }]
    rec = parse_and_deduplicate_overpass(elements, square_geom_fixture())[0]
    assert rec.housenumber == "707"
    assert rec.postcode == "38104"


def test_cache_roundtrip_persists_housenumber_and_postcode(tmp_path, monkeypatch):
    tn_geom = square_geom_fixture()
    cache_file = tmp_path / "cache.geojson"
    resp = MagicMock(status_code=200)
    resp.json.return_value = {
        "elements": [{
            "type": "node", "id": 10, "lat": 35.5, "lon": -86.0,
            "tags": {"name": "Dollar General", "addr:street": "Main St",
                     "addr:housenumber": "707", "addr:postcode": "38104"},
        }]
    }
    monkeypatch.setattr("tn_dollar_general.data_fetcher.requests.post", lambda *args, **kwargs: resp)
    stores = fetch_dollar_generals(tn_geom, cache_file, refresh=True)
    assert len(stores) == 1
    # Second call loads from the GeoJSON cache on disk.
    reloaded = fetch_dollar_generals(tn_geom, cache_file, refresh=False)
    assert reloaded[0].housenumber == stores[0].housenumber
    assert reloaded[0].postcode == stores[0].postcode
    assert reloaded[0].housenumber == "707"
    assert reloaded[0].postcode == "38104"


def test_dedup_merge_fills_housenumber_from_colocated_duplicate():
    # node + way for the same store within 50 m: node has addr:street, the
    # way also has addr:housenumber; kept record must gain the housenumber.
    tn_geom = square_geom_fixture()
    elements = [
        {"type": "node", "id": 1, "lat": 35.5, "lon": -86.0,
         "tags": {"name": "Dollar General", "addr:street": "Main St", "addr:state": "TN"}},
        {"type": "way", "id": 2, "center": {"lat": 35.5, "lon": -86.00035},
         "tags": {"name": "Dollar General", "building": "retail", "addr:housenumber": "707"}},
    ]
    stores = parse_and_deduplicate_overpass(elements, tn_geom, dedup_distance_m=50.0)
    assert len(stores) == 1
    assert stores[0].street == "Main St"      # kept from the node
    assert stores[0].housenumber == "707"     # filled from the suppressed way


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
