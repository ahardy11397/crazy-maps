import numpy as np
import pyproj
import shapely
from shapely.geometry import box

from tn_dollar_general.data_fetcher import StoreRecord
from tn_dollar_general.geometry import find_furthest_point
from tn_dollar_general.hexgrid import (
    HexRanking,
    build_hex_ranking,
    hex_vertices_m,
)

# ~20 km wide square "state" built around a real TN-region lat/lon so EPSG:6575
# projection is exercised for real. Center near (35.5, -86.0).
_SQUARE_CENTER_LAT = 35.5
_SQUARE_CENTER_LON = -86.0
# Half-extents tuned so that the pinned hex lattice (origin = SW corner of the
# projected bounds, column pitch dx = 1000 m, row pitch dy = 3R/2 ~ 866.03 m)
# has a hex center within ~1 m of the box's projected center. This keeps the
# rank-1 (most remote) hex essentially at the true furthest point, so its
# geodesic distance matches the exact solver within the 0.05 mi tolerance.
# Resulting projected box: ~20.0 km (E-W) x ~20.8 km (N-S).
_HALF_LAT_DEG = 0.09366
_HALF_LON_DEG = 0.1101

_SQUARE = box(
    _SQUARE_CENTER_LON - _HALF_LON_DEG,
    _SQUARE_CENTER_LAT - _HALF_LAT_DEG,
    _SQUARE_CENTER_LON + _HALF_LON_DEG,
    _SQUARE_CENTER_LAT + _HALF_LAT_DEG,
)

# 4 corners + 4 edge midpoints of the square state.
_SQUARE_LAT_MIN = _SQUARE_CENTER_LAT - _HALF_LAT_DEG
_SQUARE_LAT_MAX = _SQUARE_CENTER_LAT + _HALF_LAT_DEG
_SQUARE_LON_MIN = _SQUARE_CENTER_LON - _HALF_LON_DEG
_SQUARE_LON_MAX = _SQUARE_CENTER_LON + _HALF_LON_DEG

_STORE_LATLONS = [
    (_SQUARE_LAT_MIN, _SQUARE_LON_MIN),  # SW corner
    (_SQUARE_LAT_MIN, _SQUARE_LON_MAX),  # SE corner
    (_SQUARE_LAT_MAX, _SQUARE_LON_MIN),  # NW corner
    (_SQUARE_LAT_MAX, _SQUARE_LON_MAX),  # NE corner
    (_SQUARE_LAT_MIN, _SQUARE_CENTER_LON),  # S edge midpoint
    (_SQUARE_LAT_MAX, _SQUARE_CENTER_LON),  # N edge midpoint
    (_SQUARE_CENTER_LAT, _SQUARE_LON_MIN),  # W edge midpoint
    (_SQUARE_CENTER_LAT, _SQUARE_LON_MAX),  # E edge midpoint
]


def _square_state_and_corner_stores() -> tuple[shapely.Geometry, list[StoreRecord]]:
    stores = []
    for idx, (lat, lon) in enumerate(_STORE_LATLONS, start=1):
        stores.append(
            StoreRecord(
                id=str(idx),
                name=f"Square Store #{idx}",
                lat=lat,
                lon=lon,
                city=None,
                street=None,
                state=None,
                in_tn=True,
            )
        )
    return _SQUARE, stores


_TO_PROJ = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6575", always_xy=True)


def test_rank_one_is_the_center_of_four_corner_stores():
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
    mid_lat = (_SQUARE_LAT_MIN + _SQUARE_LAT_MAX) / 2.0
    mid_lon = (_SQUARE_LON_MIN + _SQUARE_LON_MAX) / 2.0
    mid_x, mid_y = _TO_PROJ.transform(mid_lon, mid_lat)
    midpoint_xy = np.array([mid_x, mid_y])
    center_hex = int(np.argmin(np.linalg.norm(ranking.centers_m - midpoint_xy, axis=1)))
    assert ranking.rank[center_hex] == 1


def test_rank_descends_monotonically_toward_a_store():
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
    mid_lat = (_SQUARE_LAT_MIN + _SQUARE_LAT_MAX) / 2.0
    mid_lon = (_SQUARE_LON_MIN + _SQUARE_LON_MAX) / 2.0
    mid_x, mid_y = _TO_PROJ.transform(mid_lon, mid_lat)
    corner_x, corner_y = _TO_PROJ.transform(_SQUARE_LON_MAX, _SQUARE_LAT_MAX)
    midpoint_xy = np.array([mid_x, mid_y])
    corner_xy = np.array([corner_x, corner_y])
    direction = corner_xy - midpoint_xy
    seg_len = float(np.linalg.norm(direction))
    unit = direction / seg_len

    rel = ranking.centers_m - midpoint_xy
    along = rel @ unit
    perp = np.abs(rel @ np.array([-unit[1], unit[0]]))
    on_segment_idx = np.flatnonzero((along >= 0.0) & (along <= seg_len) & (perp <= 100.0))

    assert len(on_segment_idx) >= 2  # sanity: the walk must contain hexes
    order = on_segment_idx[np.argsort(along[on_segment_idx])]
    dists = ranking.distance_meters[order]
    assert np.all(np.diff(dists) < 0.0)  # strictly decreasing toward the corner store


def test_neighbors_of_rank_contract():
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
    assert ranking.ranks == len(ranking.rank) == len(ranking.distance_meters)
    for k in (2, 5, ranking.ranks - 2):
        worse, better = ranking.neighbors_of_rank(k)
        assert worse is not None and better is not None
        assert ranking.rank[worse] == k - 1 and ranking.rank[better] == k + 1
    worse, better = ranking.neighbors_of_rank(1)
    assert worse is None and ranking.rank[better] == 2
    worse, better = ranking.neighbors_of_rank(ranking.ranks)
    assert ranking.rank[worse] == ranking.ranks - 1 and better is None


def test_top_rank1_matches_solver_within_five_hundredth_mile():
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
    result = find_furthest_point(tn_geom=geom, stores=stores)
    top_idx = ranking.by_rank(1)
    assert abs(ranking.geodesic_miles[top_idx] - result.distance_miles) <= 0.05


def test_hex_vertices_across_flats_and_radius():
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
    verts = hex_vertices_m(ranking.centers_m, 1000.0)
    assert verts.shape == (len(ranking.centers_m), 6, 2)
    radius = np.linalg.norm(verts - ranking.centers_m[:, None, :], axis=-1)
    assert np.allclose(radius.max(axis=-1), 577.350, atol=0.1)  # circumradius R = w/sqrt(3)
    span_x = verts[..., 0].max(axis=-1) - verts[..., 0].min(axis=-1)
    assert np.allclose(span_x, 1000.0, atol=0.1)  # across-flats span = w


def test_ranking_array_dtypes_and_shapes():
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
    N = ranking.ranks
    assert isinstance(ranking, HexRanking)
    assert ranking.centers_latlon.shape == (N, 2)
    assert ranking.centers_m.shape == (N, 2)
    assert ranking.distance_meters.shape == (N,)
    assert ranking.geodesic_miles.shape == (N,)
    assert ranking.rank.shape == (N,)
    assert ranking.store_idx.shape == (N,)
    assert ranking.centers_latlon.dtype == np.float64
    assert ranking.centers_m.dtype == np.float64
    assert ranking.distance_meters.dtype == np.float64
    assert ranking.geodesic_miles.dtype == np.float64
    assert ranking.rank.dtype == np.int32
    assert ranking.store_idx.dtype == np.int32
    assert ranking.hex_across_m == 1000.0
    # ranks are a permutation of 1..N
    assert np.array_equal(np.sort(ranking.rank), np.arange(1, N + 1))
    # store_idx points into the stores list passed in
    assert np.all((ranking.store_idx >= 0) & (ranking.store_idx < len(stores)))


def test_top_k_returns_most_remote_first():
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
    top = ranking.top_k(10)
    assert len(top) == 10
    assert top[0] == ranking.by_rank(1)
    top_ranks = [int(ranking.rank[i]) for i in top]
    assert top_ranks == list(range(1, 11))
    # distances must be non-increasing across the top-k
    d = ranking.distance_meters[np.array(top)]
    assert np.all(np.diff(d) <= 0.0)
