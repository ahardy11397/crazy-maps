import math
import pytest
from shapely.geometry import box, MultiPolygon, Polygon
from pyproj import Geod
from tn_dollar_general.data_fetcher import StoreRecord
from tn_dollar_general.geometry import (
    find_furthest_point,
    geodesic_circle_coords,
    FurthestPointResult,
    NearestStoreInfo,
)

GEOD = Geod(ellps="WGS84")


def _make_store(idx: int, lat: float, lon: float, in_tn: bool = True) -> StoreRecord:
    return StoreRecord(
        id=str(idx),
        name=f"DG #{idx}",
        lat=lat,
        lon=lon,
        city=None,
        street=None,
        state="TN" if in_tn else "KY",
        in_tn=in_tn,
    )


def test_interior_voronoi_vertex_exact_center():
    # Symmetric square around (-86.0, 35.8) with 4 stores at the corners
    poly = box(-86.1, 35.7, -85.9, 35.9)
    stores = [
        _make_store(1, 35.7, -86.1),
        _make_store(2, 35.7, -85.9),
        _make_store(3, 35.9, -86.1),
        _make_store(4, 35.9, -85.9),
    ]
    res = find_furthest_point(poly, stores, grid_step_m=1000.0)
    assert abs(res.lat - 35.8) < 1e-3
    assert abs(res.lon - (-86.0)) < 1e-3
    assert len(res.nearest_stores) >= 3


def test_boundary_constrained_with_infinite_voronoi_ridges():
    # Only 2 stores placed near the west edge of the polygon -> furthest point must be on the east corners
    poly = box(-86.1, 35.7, -85.9, 35.9)
    stores = [
        _make_store(1, 35.75, -86.09),
        _make_store(2, 35.85, -86.09),
    ]
    res = find_furthest_point(poly, stores, grid_step_m=1000.0)
    assert abs(res.lon - (-85.9)) < 1e-4
    assert abs(res.lat - 35.7) < 1e-3 or abs(res.lat - 35.9) < 1e-3


def test_out_of_state_border_store_constrains_result():
    # Polygon has 1 store on the west edge; without out-of-state store, east edge (-85.9) would win.
    # Place an out-of-state store right outside the east edge at (-85.89, 35.8).
    poly = box(-86.1, 35.7, -85.9, 35.9)
    stores = [
        _make_store(1, 35.8, -86.1, in_tn=True),
        _make_store(2, 35.8, -85.89, in_tn=False),
    ]
    res = find_furthest_point(poly, stores, grid_step_m=1000.0)
    # Furthest point must be near the midpoint longitude (~ -85.995) on top or bottom edge, NOT on the east edge (-85.9)
    assert abs(res.lon - (-85.995)) < 0.01
    assert abs(res.lat - 35.7) < 1e-3 or abs(res.lat - 35.9) < 1e-3


def test_multipolygon_and_geodesic_circle():
    mp = MultiPolygon([box(-86.1, 35.7, -86.0, 35.8), box(-85.5, 35.7, -85.3, 35.9)])
    stores = [_make_store(1, 35.75, -86.05), _make_store(2, 35.7, -85.5)]
    res = find_furthest_point(mp, stores, grid_step_m=1000.0)
    assert abs(res.lon - (-85.3)) < 1e-3 and abs(res.lat - 35.9) < 1e-3
    circle = geodesic_circle_coords(res.lat, res.lon, res.distance_meters, num_points=36)
    assert len(circle) == 37  # closed ring


def test_empty_stores_or_geom_raises_value_error():
    poly = box(-86.1, 35.7, -85.9, 35.9)
    with pytest.raises(ValueError, match="stores"):
        find_furthest_point(poly, [])

    empty_poly = Polygon()
    with pytest.raises(ValueError, match="geom"):
        find_furthest_point(empty_poly, [_make_store(1, 35.8, -86.0)])


def test_result_units_consistency():
    poly = box(-86.1, 35.7, -85.9, 35.9)
    stores = [_make_store(1, 35.7, -86.1), _make_store(2, 35.9, -85.9)]
    res = find_furthest_point(poly, stores, grid_step_m=1000.0)
    assert abs(res.distance_km - res.distance_meters / 1000.0) < 1e-6
    assert abs(res.distance_miles - res.distance_meters / 1609.344) < 1e-6
    for s in res.nearest_stores:
        assert abs(s.distance_km - s.distance_meters / 1000.0) < 1e-6
        assert abs(s.distance_miles - s.distance_meters / 1609.344) < 1e-6
        assert 0.0 <= s.bearing_degrees < 360.0
