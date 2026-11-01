from pathlib import Path
from unittest.mock import patch, MagicMock
import json
from shapely.geometry import box, MultiPolygon, Polygon
from tn_dollar_general.data_fetcher import (
    DEFAULT_USER_AGENT,
    StoreRecord,
    TN_BOUNDARY_URLS,
    HTTPNominatimClient,
    parse_boundary_geojson,
    parse_and_deduplicate_overpass,
    enrich_store_addresses,
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


# ---------------------------------------------------------------------------
# Address enrichment via Nominatim (id-keyed separate cache)
# ---------------------------------------------------------------------------

def _store(store_id, lat, lon, city=None, street=None, state=None,
           housenumber=None, in_tn=True):
    return StoreRecord(
        id=store_id, name="Dollar General", lat=lat, lon=lon,
        city=city, street=street, state=state, in_tn=in_tn,
        housenumber=housenumber,
    )


class _FakeNominatimClient:
    """Same button as HTTPNominatimClient: reverse(lat, lon) -> dict | None.

    Records every (lat, lon) call; never touches the network.
    """

    def __init__(self, responses=None, raise_for=(), raw_for=()):
        self.calls = []
        self.responses = responses or {}
        self.raise_for = set(raise_for)
        self.raw_for = dict(raw_for)  # (lat, lon) -> raw truthy non-dict payload

    def reverse(self, lat, lon):
        self.calls.append((lat, lon))
        if (lat, lon) in self.raise_for:
            raise RuntimeError("nominatim unreachable")
        if (lat, lon) in self.raw_for:
            return self.raw_for[(lat, lon)]
        address = self.responses.get((lat, lon))
        return {"address": address} if address is not None else None


def test_enrich_geocodes_only_incomplete_and_uses_cache(tmp_path):
    # 3 stores: A complete (housenumber+street+city), B complete but
    # pre-seeded in the cache file, C incomplete. Only C may be geocoded
    # live. The fake records (lat, lon) calls; C is the only store whose
    # coords get looked up.
    cache_file = tmp_path / "store_addresses.json"
    cache_file.write_text(
        json.dumps({"version": 1, "addresses": {"node/2": "3 Old Hwy, Pulaski, TN"}}),
        encoding="utf-8",
    )
    store_a = _store("way/1", 35.5, -86.0, city="Cedar Grove",
                     street="New Life Ln", state="TN", housenumber="707")
    store_b = _store("node/2", 35.2, -86.2, city="Pulaski",
                     street="Old Hwy", state="TN", housenumber="3")
    store_c = _store("node/3", 35.1, -86.1)
    fake = _FakeNominatimClient(
        responses={(35.1, -86.1): {"road": "Mock Rd", "house_number": "12",
                                   "city": "Mock City", "state": "TN"}},
    )

    result = enrich_store_addresses([store_a, store_b, store_c], cache_file, client=fake)

    assert fake.calls == [(store_c.lat, store_c.lon)]      # only C geocoded live
    assert result["way/1"] == "707 New Life Ln, Cedar Grove, TN"  # composed without any call
    assert result["node/2"] == "3 Old Hwy, Pulaski, TN"    # cache honored
    assert result["node/3"] == "12 Mock Rd, Mock City, TN"
    cache = json.loads(cache_file.read_text())
    assert cache["version"] == 1
    # Amended per controller ruling (merge semantics on all write paths):
    # the seeded prior entry (node/2) is retained, the newly geocoded
    # node/3 is added, nothing is lost.
    assert cache["addresses"] == {
        "node/2": "3 Old Hwy, Pulaski, TN",
        "node/3": "12 Mock Rd, Mock City, TN",
    }


def test_enrich_failure_is_graceful_and_not_cached(tmp_path):
    # Store C raises inside the client, store D's reverse returns None,
    # store E returns a truthy non-dict payload: all absent from the
    # result and the cache, no exception, the complete store still enriches.
    cache_file = tmp_path / "store_addresses.json"
    complete = _store("way/1", 35.5, -86.0, city="Cedar Grove",
                      street="New Life Ln", state="TN", housenumber="707")
    raising = _store("node/2", 35.1, -86.1)
    empty = _store("node/3", 35.3, -86.3)
    junk_payload = _store("node/4", 35.4, -86.4)
    fake = _FakeNominatimClient(
        responses={(35.3, -86.3): None},
        raise_for={(35.1, -86.1)},
        raw_for={(35.4, -86.4): ["not", "a", "dict"]},
    )

    result = enrich_store_addresses([complete, raising, empty, junk_payload], cache_file, client=fake)

    assert "node/2" not in result
    assert "node/3" not in result
    assert "node/4" not in result
    assert result["way/1"] == "707 New Life Ln, Cedar Grove, TN"
    assert not cache_file.exists()  # nothing successful to write -> no cache


def test_enrich_refresh_regeocodes_cached_incomplete_store(tmp_path):
    # refresh=True ignores the existing cache: an incomplete store already
    # present in the cache file is geocoded again and its fresh value wins.
    cache_file = tmp_path / "store_addresses.json"
    cache_file.write_text(
        json.dumps({"version": 1, "addresses": {"node/3": "12 Stale Rd, Stale City, TN"}}),
        encoding="utf-8",
    )
    incomplete = _store("node/3", 35.1, -86.1)
    fake = _FakeNominatimClient(
        responses={(35.1, -86.1): {"road": "Fresh Rd", "house_number": "45",
                                   "city": "Fresh City", "state": "TN"}},
    )

    result = enrich_store_addresses([incomplete], cache_file, refresh=True, client=fake)

    assert fake.calls == [(incomplete.lat, incomplete.lon)]   # cached entry was NOT reused
    assert result["node/3"] == "45 Fresh Rd, Fresh City, TN"  # fresh geocode displayed
    cache = json.loads(cache_file.read_text())
    assert cache["addresses"]["node/3"] == "45 Fresh Rd, Fresh City, TN"  # new wins on file


def test_enrich_refresh_write_preserves_prior_entries_and_new_wins(tmp_path):
    # Refresh write-back must not discard previously geocoded addresses
    # (spec 5.4): a prior entry for a store not looked up this run survives,
    # and a re-geocoded id overwrites its stale value.
    cache_file = tmp_path / "store_addresses.json"
    cache_file.write_text(
        json.dumps({"version": 1, "addresses": {"node/3": "12 Stale Rd, Stale City, TN",
                                                "node/7": "42 Old Ave, Kept City, TN"}}),
        encoding="utf-8",
    )
    regcoded = _store("node/3", 35.1, -86.1)
    prior_entry_store = _store("node/7", 35.8, -86.8, city="Kept City",
                               street="Kept Ln", state="TN", housenumber="9")
    fake = _FakeNominatimClient(
        responses={(35.1, -86.1): {"road": "Fresh Rd", "house_number": "45",
                                   "city": "Fresh City", "state": "TN"}},
    )

    result = enrich_store_addresses([regcoded, prior_entry_store], cache_file, refresh=True, client=fake)

    assert fake.calls == [(regcoded.lat, regcoded.lon)]          # only the incomplete store looked up
    assert result["node/7"] == "9 Kept Ln, Kept City, TN"            # complete -> own fields, cache ignored
    cache = json.loads(cache_file.read_text())
    assert cache["addresses"] == {
        "node/3": "45 Fresh Rd, Fresh City, TN",   # fresh re-geocode persisted
        "node/7": "42 Old Ave, Kept City, TN",     # prior entry survived untouched
    }


def test_enrich_display_composition_order(tmp_path):
    # Geocoded parts compose in order house_number+road, city, state,
    # ", "-joined, dropping empties -> "707 New Life Ln, Cedar Grove, TN".
    cache_file = tmp_path / "store_addresses.json"
    incomplete = _store("node/9", 35.0, -86.0)
    fake = _FakeNominatimClient(
        responses={(35.0, -86.0): {"road": "New Life Ln", "house_number": "707",
                                   "city": "Cedar Grove", "state": "TN"}},
    )

    result = enrich_store_addresses([incomplete], cache_file, client=fake)

    assert result == {"node/9": "707 New Life Ln, Cedar Grove, TN"}


def test_http_nominatim_client_sends_one_request_then_sleeps(monkeypatch):
    # Polite client: exactly one GET per reverse() call with a
    # DEFAULT_USER_AGENT-suffixed UA, then time.sleep(delay_seconds).
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"address": {"road": "X"}}
    get_calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        get_calls.append({"url": url, "params": params,
                          "headers": headers, "timeout": timeout})
        return resp

    monkeypatch.setattr("tn_dollar_general.data_fetcher.requests.get", fake_get)

    client = HTTPNominatimClient()
    result = client.reverse(35.5, -86.0)
    client.reverse(35.6, -86.1)

    assert result == {"address": {"road": "X"}}
    assert [c["url"] for c in get_calls] == [client.endpoint] * 2
    assert get_calls[0]["params"]["lat"] == 35.5
    assert get_calls[0]["params"]["lon"] == -86.0
    assert get_calls[0]["headers"]["User-Agent"].startswith(DEFAULT_USER_AGENT)
    assert sleeps == [client.delay_seconds] * 2
