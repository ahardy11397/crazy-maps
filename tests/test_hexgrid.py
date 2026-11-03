import base64
import dataclasses
import json
import struct

import numpy as np
import pyproj
import pytest
import shapely
from shapely.geometry import box
from shapely.prepared import prep

from tn_dollar_general.data_fetcher import StoreRecord
from tn_dollar_general.geometry import find_furthest_point
from tn_dollar_general.hexgrid import (
    HexRanking,
    build_click_table,
    build_hex_ranking,
    encode_data_block,
    hex_vertices_m,
    render_remoteness_png,
    _ramp_color,
    _store_address,
)

_CLICK_SENTINEL = 0xFFFFFFFF  # pinned "no hex resolves here" bucket value

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


def _tn_stores() -> list[StoreRecord]:
    """5 stores spread across the full TN-scale extent (see ``_TN_STATE_POLY``)."""
    return [
        StoreRecord(
            id=str(idx),
            name=f"TN Store #{idx}",
            lat=lat,
            lon=lon,
            city=None,
            street=None,
            state=None,
            in_tn=True,
        )
        for idx, (lat, lon) in enumerate(_TN_STORE_LATLONS, start=1)
    ]


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


def test_click_table_snaps_inside_and_sentinel_offstate():
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
    table = build_click_table(ranking)
    assert table.cell_deg == 0.005
    assert table.idx.dtype == np.uint32
    assert table.idx.shape == (table.lat_cells * table.lon_cells,)

    mid_lat = (_SQUARE_LAT_MIN + _SQUARE_LAT_MAX) / 2.0
    mid_lon = (_SQUARE_LON_MIN + _SQUARE_LON_MAX) / 2.0
    cl = ranking.centers_latlon

    # Interior click, offset half a cell from the square's center: resolve via
    # the same cell math the JS decoder uses (no helper function needed) and
    # check that the resolved hex center lies within 0.05 deg of the
    # smallest-distance (nearest) hex around the click point.
    p_lat = mid_lat + 0.5 * table.cell_deg
    p_lon = mid_lon - 0.5 * table.cell_deg
    i = int(np.floor((p_lat - table.lat_min) / table.cell_deg))
    j = int(np.floor((p_lon - table.lon_min) / table.cell_deg))
    assert 0 <= i < table.lat_cells and 0 <= j < table.lon_cells
    resolved = int(table.idx[i * table.lon_cells + j])
    assert resolved != _CLICK_SENTINEL
    d_click = np.hypot(cl[:, 0] - p_lat, cl[:, 1] - p_lon)
    near = np.flatnonzero(d_click < 0.1)
    ref = int(near[np.argmin(d_click[near])])  # smallest-distance hex near the point
    assert np.hypot(cl[resolved, 0] - cl[ref, 0], cl[resolved, 1] - cl[ref, 1]) <= 0.05

    # Off-state: a click 2 degrees beyond the clip bounds falls outside the
    # bucket grid entirely -> the pinned cell math resolves to the sentinel.
    far_lat = table.lat_min + (table.lat_cells * table.cell_deg) + 2.0
    far_lon = table.lon_min + (table.lon_cells * table.cell_deg) + 2.0
    i_far = int(np.floor((far_lat - table.lat_min) / table.cell_deg))
    j_far = int(np.floor((far_lon - table.lon_min) / table.cell_deg))
    outside = i_far < 0 or i_far >= table.lat_cells or j_far < 0 or j_far >= table.lon_cells
    assert outside
    idx_far = _CLICK_SENTINEL if outside else int(table.idx[i_far * table.lon_cells + j_far])
    assert idx_far == _CLICK_SENTINEL

    # Ragged edge cells whose own centers lie beyond the clip bounds carry the
    # sentinel, whole edge rows / columns of them. In this fixture the ragged
    # edge falls on the lat axis only.
    clip_lat_max = float(cl[:, 0].max())
    clip_lon_max = float(cl[:, 1].max())
    last_row_lat = table.lat_min + (table.lat_cells - 0.5) * table.cell_deg
    last_col_lon = table.lon_min + (table.lon_cells - 0.5) * table.cell_deg
    if last_row_lat > clip_lat_max:
        last_row = table.idx[
            (table.lat_cells - 1) * table.lon_cells : table.lat_cells * table.lon_cells
        ]
        assert np.all(last_row == _CLICK_SENTINEL)
    if last_col_lon > clip_lon_max:
        assert np.all(table.idx[table.lon_cells - 1 :: table.lon_cells] == _CLICK_SENTINEL)
    assert bool(np.any(table.idx == _CLICK_SENTINEL))


def test_click_table_self_consistency_for_all_hex_centers():
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
    table = build_click_table(ranking)
    cl = ranking.centers_latlon
    clip = {
        "lat": (float(cl[:, 0].min()), float(cl[:, 0].max())),
        "lon": (float(cl[:, 1].min()), float(cl[:, 1].max())),
    }

    rng = np.random.default_rng(4)
    n_sample = min(500, ranking.ranks)  # the square fixture holds < 500 hexes
    sample = rng.choice(ranking.ranks, size=n_sample, replace=False)

    resolved_count = 0
    for h in sample:
        lat, lon = cl[h]
        i = int(np.floor((lat - table.lat_min) / table.cell_deg))
        j = int(np.floor((lon - table.lon_min) / table.cell_deg))
        if not (0 <= i < table.lat_cells and 0 <= j < table.lon_cells):
            continue
        resolved = int(table.idx[i * table.lon_cells + j])
        if resolved == _CLICK_SENTINEL:
            # The pinned sentinel rule only allows skipped cells whose own
            # center is beyond the clip bounds (ragged edge rows / columns).
            cell_lat = table.lat_min + (i + 0.5) * table.cell_deg
            cell_lon = table.lon_min + (j + 0.5) * table.cell_deg
            beyond = (
                cell_lat > clip["lat"][1]
                or cell_lat < clip["lat"][0]
                or cell_lon > clip["lon"][1]
                or cell_lon < clip["lon"][0]
            )
            assert beyond
            continue
        resolved_count += 1
        d = float(np.hypot(cl[resolved, 0] - lat, cl[resolved, 1] - lon))
        assert d <= 3 * table.cell_deg
    # Essentially every sampled center must resolve; only the ragged-edge
    # sentinel cells may be skipped.
    assert resolved_count >= n_sample - 2 * table.lon_cells


def test_data_block_roundtrip_and_top10_alignment():
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
    table = build_click_table(ranking)
    block = encode_data_block(ranking, table, store_display={}, stores=stores)
    cl = ranking.centers_latlon
    N = ranking.ranks
    S = len(stores)

    loaded = json.loads(json.dumps(block))  # blocks must be JSON-serializable

    assert loaded["format"] == 1
    assert loaded["totals"] == {"hexes": N, "stores": S}
    clip = loaded["clip"]
    assert clip["proj"] == "EPSG:6575"
    assert clip["lat"] == [float(cl[:, 0].min()), float(cl[:, 0].max())]
    assert clip["lon"] == [float(cl[:, 1].min()), float(cl[:, 1].max())]

    dist = np.frombuffer(base64.b64decode(loaded["dist_mi_q"]), dtype="<u2")
    ro = np.frombuffer(base64.b64decode(loaded["rank_order"]), dtype="<u4")
    si = np.frombuffer(base64.b64decode(loaded["store_idx"]), dtype="<u4")
    centers = np.frombuffer(base64.b64decode(loaded["centers"]), dtype="<u2").reshape(N, 2)
    click_idx = np.frombuffer(base64.b64decode(loaded["click"]["idx"]), dtype="<u4")

    assert dist.shape == (N,) == ro.shape == si.shape
    assert centers.shape == (N, 2)
    assert click_idx.shape == (table.lat_cells * table.lon_cells,)

    # top10: fixed length 10, entries are hex indices, rank 1 first, carrying
    # the maximum quantized distance.
    assert len(loaded["top10"]) == 10
    assert loaded["top10"][0] == int(ranking.by_rank(1))
    assert int(dist[loaded["top10"][0]]) == int(round(float(ranking.geodesic_miles[loaded["top10"][0]]) * 1000))
    assert int(dist[loaded["top10"][0]]) == int(round(float(np.max(ranking.geodesic_miles)) * 1000))

    # rank_order: the hex index at rank k sits at slot k-1.
    assert int(ro[0]) == int(ranking.by_rank(1))
    assert int(ro[1]) == int(ranking.by_rank(2))
    assert int(ro[4]) == int(ranking.by_rank(5))
    assert int(ro[N - 1]) == int(ranking.by_rank(N))
    # store_idx mirrors the ranking's nearest-store indices.
    assert np.array_equal(si, ranking.store_idx.astype(np.uint32))

    # Quantized centers decode back onto the true latlon within quantization
    # error (half a quantization step).
    lat_min, lat_max = clip["lat"]
    lon_min, lon_max = clip["lon"]
    lat_rec = lat_min + centers[:, 0] / 65535.0 * (lat_max - lat_min)
    lon_rec = lon_min + centers[:, 1] / 65535.0 * (lon_max - lon_min)
    assert float(np.abs(lat_rec - cl[:, 0]).max()) <= 0.6 * (lat_max - lat_min) / 65535.0
    assert float(np.abs(lon_rec - cl[:, 1]).max()) <= 0.6 * (lon_max - lon_min) / 65535.0

    # Click metadata round-trips verbatim.
    c = loaded["click"]
    assert c["cell_deg"] == table.cell_deg
    assert c["lat_min"] == table.lat_min
    assert c["lon_min"] == table.lon_min
    assert c["lat_cells"] == table.lat_cells
    assert c["lon_cells"] == table.lon_cells
    assert np.array_equal(click_idx, table.idx.astype(np.uint32))

    # stores: one entry per store, each with a non-empty display address
    # (store_display={} => the Python-side fallback string).
    assert len(loaded["stores"]) == S
    for entry, store in zip(loaded["stores"], stores):
        assert entry["id"] == store.id
        assert entry["name"] == store.name
        assert isinstance(entry["address"], str) and entry["address"]
    # A provided display address wins over the fallback.
    block2 = encode_data_block(ranking, table, store_display={"1": "123 Main St, Somewhere"}, stores=stores)
    loaded2 = json.loads(json.dumps(block2))
    assert loaded2["stores"][0]["address"] == "123 Main St, Somewhere"


def test_ramp_is_monotone_darkening():
    rgb = _ramp_color(np.array([0.0, 0.5, 1.0]))
    assert rgb[0, :3].sum() > rgb[1, :3].sum() > rgb[2, :3].sum()


def test_render_remoteness_png_bounds_and_size(tmp_path):
    geom, stores = _square_state_and_corner_stores()
    raster = render_remoteness_png(build_hex_ranking(geom, stores), tmp_path / "r.png")
    data = raster.path.read_bytes()
    w, h = struct.unpack(">II", data[16:24])      # PNG IHDR
    # Fixture bounds are ~19,978 m across (~20 pinned 1000 m hexes), so strict
    # 6 px/hex could never satisfy the pinned width threshold; the renderer's
    # approved _MIN_RASTER_SIDE_PX floor raises the state scale here only.
    assert w > 400 and len(data) < 6 * 1024 * 1024


def test_raster_image_bounds_latlon_track_projected_vertices(tmp_path):
    geom, stores = _square_state_and_corner_stores()
    raster = render_remoteness_png(build_hex_ranking(geom, stores), tmp_path / "r.png")
    ranking = build_hex_ranking(geom, stores)
    verts = hex_vertices_m(ranking.centers_m, ranking.hex_across_m)
    # Leaflet draws ImageOverlay as an axis-aligned rectangle linear in
    # longitude and linear in *Mercator* latitude (EPSG:3857), so the declared
    # bounds must be the 4326 envelope of the vertices' 3857 bbox, not of the
    # 6575 bbox (different parameterizations -- see the state-scale test).
    to_3857 = pyproj.Transformer.from_crs("EPSG:6575", "EPSG:3857", always_xy=True)
    mx, my = to_3857.transform(verts[..., 0].ravel(), verts[..., 1].ravel())
    mx = np.asarray(mx, dtype=np.float64)
    my = np.asarray(my, dtype=np.float64)
    to_wgs84 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    corner_lons, corner_lats = to_wgs84.transform(
        [float(mx.min()), float(mx.min()), float(mx.max()), float(mx.max())],
        [float(my.min()), float(my.max()), float(my.max()), float(my.min())],
    )
    corner_lats = np.asarray(corner_lats, dtype=np.float64)
    corner_lons = np.asarray(corner_lons, dtype=np.float64)
    lat_min, lon_min, lat_max, lon_max = raster.image_bounds_latlon
    # Within half a hexagon across (≤ ~1 km ≈ 0.01 deg at TN latitudes) so the
    # folium ImageOverlay registers the PNG image itself, not the clip rect.
    assert abs(lat_min - float(corner_lats.min())) <= 0.01
    assert abs(lat_max - float(corner_lats.max())) <= 0.01
    assert abs(lon_min - float(corner_lons.min())) <= 0.01
    assert abs(lon_max - float(corner_lons.max())) <= 0.01
    # Ordering: (lat_min, lon_min) before (lat_max, lon_max).
    assert lat_min < lat_max and lon_min < lon_max


# ---------------------------------------------------------------------------
# State-scale overlay registration (Critical C1 regression guard)
# ---------------------------------------------------------------------------

# TN-shaped synthetic "state": full TN extent (35.0-36.6 N, -90.3 to -81.7 W)
# with the southeast corner cut off along a slanted border the way the real
# NC/GA mountain border runs (-81.7, 36.6) -> (-84.3, 35.0). Its area (~11.8
# deg^2 ~= 117k km^2) is within ~8% of real Tennessee (~109k km^2), so the
# ~136k-hex ranking built on it matches production scale for the off-state
# click mask and the data-block size budget; the raster registration test
# shares it. Asheville (35.6, -82.55) sits south of the slant — inside the
# clip bbox, outside the state.
_TN_STATE_POLY = shapely.Polygon(
    [
        (-90.3, 35.0),
        (-90.3, 36.6),
        (-81.7, 36.62),
        (-84.32, 35.0),
    ]
)

_TN_STORE_LATLONS = [
    (35.1, -90.0),
    (35.2, -82.9),
    (36.5, -88.1),
    (36.4, -82.0),
    (35.8, -86.5),
]

_TO_3857 = pyproj.Transformer.from_crs("EPSG:6575", "EPSG:3857", always_xy=True)
_WGS_TO_MERC = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_MERC_TO_WGS = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


@pytest.fixture(scope="module")
def tn_scale_ranking() -> HexRanking:
    """The ~136k-hex state-scale TN-shaped ranking, built once and shared.

    The pinned 1000 m lattice over the TN-shaped synthetic (within ~8% of
    real Tennessee's area, so the state-scale budget test matches production
    scale); a build takes ~0.3 s, so the raster registration, off-state click
    mask, and data-block size tests all reuse this one instance.
    """
    return build_hex_ranking(_TN_STATE_POLY, _tn_stores(), hex_across_m=1000.0)


def test_raster_registers_exactly_in_3857_at_state_scale(tmp_path, tn_scale_ranking):
    """The PNG is drawn in EPSG:3857 and its declared bounds register it exactly.

    Leaflet draws an ``ImageOverlay`` as an axis-aligned rectangle linear in
    longitude and linear in *Mercator* latitude. The renderer must therefore
    draw in EPSG:3857 and ``image_bounds_latlon`` must be the WGS84 corners of
    the drawn 3857 vertex bbox; the older EPSG:6575-linear rendering warped and
    shifted mid-state points by kilometres (Critical C1: the rank-1 hex drew
    ~8.8 km from its true center).

    Ground truth here is independent of the returned bounds: the PNG's edges
    sit at the pinned renderer contract's hex-vertex bounds, projected to
    3857. Each sampled hex center's *pixel fraction* is computed from those
    true 3857 image edges, and we then ask where the declared lat/lon
    rectangle (linear in x, linear in Mercator-y) draws that pixel fraction.
    """
    ranking = tn_scale_ranking
    raster = render_remoteness_png(ranking, tmp_path / "tn.png", px_per_hex=6)
    lat_min, lon_min, lat_max, lon_max = raster.image_bounds_latlon

    # True image edges in EPSG:3857: the vertices' 3857 bbox (what the figure
    # xlim/ylim span, per the pinned "image covers the hex-vertex bounds"
    # contract).
    verts = hex_vertices_m(ranking.centers_m, ranking.hex_across_m)
    vx, vy = _TO_3857.transform(verts[..., 0].ravel(), verts[..., 1].ravel())
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    xmin, xmax = float(vx.min()), float(vx.max())
    ymin, ymax = float(vy.min()), float(vy.max())

    # The 512 px floor must NOT engage at state scale (strict 6 px/hex), and
    # the PNG must stay under the 6 MB budget.
    data = raster.path.read_bytes()
    w, h = struct.unpack(">II", data[16:24])
    assert w > 4000 and h > 900, (w, h)
    assert len(data) < 6 * 1024 * 1024

    # >=50 sampled hex centers across the whole state.
    step = max(1, ranking.ranks // 60)
    sample_idx = np.arange(0, ranking.ranks, step)[:60]
    assert len(sample_idx) >= 50
    px, py = _TO_3857.transform(
        ranking.centers_m[sample_idx, 0], ranking.centers_m[sample_idx, 1]
    )
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    fx = (px - xmin) / (xmax - xmin)         # pixel fraction across the PNG
    fy = (py - ymin) / (ymax - ymin)         # pixel fraction down the PNG

    # Where Leaflet draws each pixel fraction: linear in x and linear in
    # Mercator-y through the *declared* corner lat/lon.
    drawn_lon = lon_min + fx * (lon_max - lon_min)
    y_lo = float(_WGS_TO_MERC.transform(lon_min, lat_min)[1])
    y_hi = float(_WGS_TO_MERC.transform(lon_max, lat_max)[1])
    drawn_y = y_lo + fy * (y_hi - y_lo)
    drawn_lat = _MERC_TO_WGS.transform(np.zeros_like(drawn_y), drawn_y)[1]
    drawn_x, drawn_y_check = _WGS_TO_MERC.transform(drawn_lon, drawn_lat)

    err_m = np.hypot(drawn_x - px, drawn_y_check - py)
    assert float(err_m.max()) <= 15.0, (
        f"state-scale 3857 registration error up to {float(err_m.max()):.1f} m "
        f"(median {float(np.median(err_m)):.1f} m) over {len(sample_idx)} hexes"
    )


# ---------------------------------------------------------------------------
# Off-state click mask (Important I1, spec §5.3)
# ---------------------------------------------------------------------------


def test_click_table_state_mask_sends_offstate_bbox_cells_to_sentinel():
    """``state_geom`` restricts snapping to bucket cells inside the state.

    An L-shaped synthetic state: bucket-cell centers in the cut-out notch lie
    inside the hex centers' clip bbox yet outside the state. With
    ``state_geom`` they must carry the sentinel (a click there leaves the rail
    untouched); with ``state_geom=None`` the old behavior — snap to the
    geometrically nearest hex, however far — is unchanged. A cell clearly
    inside the state still resolves to the geometrically nearest ranked hex
    (that resolution IS the spec §5.3 border-gap snap).
    """
    minx, miny, maxx, maxy = _SQUARE.bounds
    splitx = minx + 0.6 * (maxx - minx)   # notch edge (lon), full west arm kept
    splity = miny + 0.3 * (maxy - miny)   # notch edge (lat), full south arm kept
    l_state = shapely.Polygon(
        [
            (minx, miny),
            (maxx, miny),
            (maxx, splity),
            (splitx, splity),
            (splitx, maxy),
            (minx, maxy),
        ]
    )
    ranking = build_hex_ranking(
        l_state, _square_state_and_corner_stores()[1], hex_across_m=2000.0
    )
    cl = ranking.centers_latlon
    clip_lat_min, clip_lat_max = float(cl[:, 0].min()), float(cl[:, 0].max())
    clip_lon_min, clip_lon_max = float(cl[:, 1].min()), float(cl[:, 1].max())

    masked = build_click_table(ranking, state_geom=l_state)
    plain = build_click_table(ranking)
    assert (
        masked.lat_cells,
        masked.lon_cells,
        masked.cell_deg,
        masked.lat_min,
        masked.lon_min,
    ) == (plain.lat_cells, plain.lon_cells, plain.cell_deg, plain.lat_min, plain.lon_min)

    def cell_of(lat: float, lon: float) -> tuple[int, int, float, float]:
        i = int(np.floor((lat - masked.lat_min) / masked.cell_deg))
        j = int(np.floor((lon - masked.lon_min) / masked.cell_deg))
        return i, j, masked.lat_min + (i + 0.5) * masked.cell_deg, masked.lon_min + (j + 0.5) * masked.cell_deg

    # --- Off-state, in-bbox: a bucket cell in the notch ---------------------
    notch_lat = 0.5 * (splity + maxy)
    notch_lon = 0.5 * (splitx + maxx)
    i, j, cell_lat, cell_lon = cell_of(notch_lat, notch_lon)
    assert 0 <= i < masked.lat_cells and 0 <= j < masked.lon_cells
    assert clip_lat_min <= cell_lat <= clip_lat_max
    assert clip_lon_min <= cell_lon <= clip_lon_max
    assert not shapely.contains_xy(l_state, cell_lon, cell_lat)  # yes, off-state
    assert int(masked.idx[i * masked.lon_cells + j]) == _CLICK_SENTINEL
    # Old behavior (state_geom=None): the very same cell snapped to a real hex.
    plain_idx = int(plain.idx[i * plain.lon_cells + j])
    assert plain_idx != _CLICK_SENTINEL and plain_idx < ranking.ranks

    # --- In-state: a cell deep inside the west arm still snaps --------------
    in_lat = miny + 0.11 * (maxy - miny)
    in_lon = minx + 0.25 * (maxx - minx)
    i2, j2, cell_lat2, cell_lon2 = cell_of(in_lat, in_lon)
    assert shapely.contains_xy(l_state, cell_lon2, cell_lat2)
    resolved = int(masked.idx[i2 * masked.lon_cells + j2])
    assert resolved != _CLICK_SENTINEL and resolved < ranking.ranks
    # It is the geometrically nearest hex to the cell center.
    d = np.hypot(cl[:, 0] - cell_lat2, cl[:, 1] - cell_lon2)
    assert resolved == int(np.argmin(d))
    assert int(plain.idx[i2 * plain.lon_cells + j2]) == resolved  # mask is a no-op inside

    # --- Bulk invariant: exactly the off-state cells carry the sentinel ----
    state_cell_lats = masked.lat_min + (np.arange(masked.lat_cells) + 0.5) * masked.cell_deg
    state_cell_lons = masked.lon_min + (np.arange(masked.lon_cells) + 0.5) * masked.cell_deg
    lat_grid, lon_grid = np.meshgrid(state_cell_lats, state_cell_lons, indexing="ij")
    inside_state = shapely.contains_xy(l_state, lon_grid.ravel(), lat_grid.ravel())
    beyond = (
        (state_cell_lats > clip_lat_max) | (state_cell_lats < clip_lat_min)
    )[:, None] | ((state_cell_lons > clip_lon_max) | (state_cell_lons < clip_lon_min))[None, :]
    expected_sentinel = (~inside_state) | beyond.ravel()
    assert np.array_equal(masked.idx == _CLICK_SENTINEL, expected_sentinel)
    # Unmasked table is untouched: only the ragged beyond-clip cells.
    assert np.array_equal(plain.idx == _CLICK_SENTINEL, beyond.ravel())


def test_click_table_state_mask_offstate_cells_at_tn_scale(tn_scale_ranking):
    """Spec §5.3 at state scale: an Asheville-like corner click must NOT open
    a TN hex card tens of kilometres away.

    Clicking inside the state with no ranked hex underneath snaps to the
    nearest ranked hex (border-gap snap), but cells whose center is outside
    the state carry the sentinel. On the pre-fix code every cell inside the
    ~8.7 deg x 1.7 deg center bbox snapped (100% of ~40k off-state cells).
    """
    ranking = tn_scale_ranking
    cl = ranking.centers_latlon
    clip_lat_min, clip_lat_max = float(cl[:, 0].min()), float(cl[:, 0].max())
    clip_lon_min, clip_lon_max = float(cl[:, 1].min()), float(cl[:, 1].max())

    masked = build_click_table(ranking, state_geom=_TN_STATE_POLY)
    plain = build_click_table(ranking)

    # Asheville sits inside the clip bbox but clearly outside TN's slanted
    # eastern border -> sentinel with the mask, a (wrong) snap without it.
    asheville_lat, asheville_lon = 35.595, -82.551
    assert clip_lat_min < asheville_lat < clip_lat_max
    assert clip_lon_min < asheville_lon < clip_lon_max
    assert not shapely.contains_xy(_TN_STATE_POLY, asheville_lon, asheville_lat)
    i = int(np.floor((asheville_lat - masked.lat_min) / masked.cell_deg))
    j = int(np.floor((asheville_lon - masked.lon_min) / masked.cell_deg))
    assert masked.idx[i * masked.lon_cells + j] == _CLICK_SENTINEL
    plain_idx = int(plain.idx[i * plain.lon_cells + j])
    assert plain_idx != _CLICK_SENTINEL and plain_idx < ranking.ranks  # pre-fix bug

    # An in-state cell resolves identically with and without the mask.
    tn_lat, tn_lon = 35.86, -86.35
    assert shapely.contains_xy(_TN_STATE_POLY, tn_lon, tn_lat)
    i2 = int(np.floor((tn_lat - masked.lat_min) / masked.cell_deg))
    j2 = int(np.floor((tn_lon - masked.lon_min) / masked.cell_deg))
    resolved = int(masked.idx[i2 * masked.lon_cells + j2])
    assert resolved != _CLICK_SENTINEL and resolved < ranking.ranks
    assert resolved == int(plain.idx[i2 * plain.lon_cells + j2])

    # Bulk: exactly the off-state (or beyond-clip) cells carry the sentinel.
    state_cell_lats = masked.lat_min + (np.arange(masked.lat_cells) + 0.5) * masked.cell_deg
    state_cell_lons = masked.lon_min + (np.arange(masked.lon_cells) + 0.5) * masked.cell_deg
    lat_grid, lon_grid = np.meshgrid(state_cell_lats, state_cell_lons, indexing="ij")
    inside_state = shapely.contains_xy(_TN_STATE_POLY, lon_grid.ravel(), lat_grid.ravel())
    beyond = (
        (state_cell_lats > clip_lat_max) | (state_cell_lats < clip_lat_min)
    )[:, None] | ((state_cell_lons > clip_lon_max) | (state_cell_lons < clip_lon_min))[None, :]
    expected_sentinel = (~inside_state) | beyond.ravel()
    assert np.array_equal(masked.idx == _CLICK_SENTINEL, expected_sentinel)
    sent_fraction = float(expected_sentinel.mean())
    assert sent_fraction > 0.10  # a real chunk of the bbox is off-state
    assert sent_fraction < 0.90  # most of the state still snaps
    assert np.array_equal(plain.idx == _CLICK_SENTINEL, beyond.ravel())


# ---------------------------------------------------------------------------
# Data-block size budget at production scale (Important I2)
# ---------------------------------------------------------------------------


def test_state_scale_data_block_within_amended_size_budget(tn_scale_ranking):
    """Guard for the controller-amended size budget at state scale.

    The plan's <= 1.5 MB budget is arithmetically impossible under its own
    pinned schema: 14 B/hex per-hex arrays (~2.35 MB base64) plus the pinned
    0.005 deg click table (~3.1 MB base64) measure ~5.0-5.5 MB at state scale
    (~170k hexes). Ruling (binding, recorded in the ledger): the acceptance
    goal is fast open/click (measured 41 ms decode, 26 MB heap); the budgets
    were amended to data block <= 6.0 MB and HTML <= 7.5 MB. The data-block
    half is enforced here; compression / analytic lookup stay off the table
    unless a real perf regression appears.
    """
    ranking = tn_scale_ranking
    click_table = build_click_table(ranking, state_geom=_TN_STATE_POLY)
    block = encode_data_block(ranking, click_table, store_display={}, stores=_tn_stores())
    payload = json.dumps(block, separators=(",", ":"), sort_keys=True)
    assert len(payload) <= 6_000_000, (
        f"state-scale data block is {len(payload)} bytes, over the amended "
        "6.0 MB budget (controller ruling; see ledger)"
    )


# ---------------------------------------------------------------------------
# Equal-distance tie-break (Minor B, spec §5.1/§6.1)
# ---------------------------------------------------------------------------


def test_equal_distance_ties_break_by_ascending_hex_id_and_store_order():
    """Equal planar distances tie stably: ranks adjacent, ascending hex id,
    independent of input store order.

    The symmetric square fixture (4 corners + 4 edge-midpoint stores) exhibits
    a genuine bit-exact duplicate in its planar distances; grouping by exact
    duplicate value pins the ``np.lexsort((hex_ids, -distance))`` tie-break.
    """
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
    values, counts = np.unique(ranking.distance_meters, return_counts=True)
    tie_values = values[counts > 1]
    assert len(tie_values) >= 1, "fixture must contain an exact distance tie"
    for value in tie_values:
        hex_ids = np.flatnonzero(ranking.distance_meters == value)
        ranks = ranking.rank[hex_ids]
        # Ranks adjacent (one contiguous integer range) ...
        assert np.array_equal(np.sort(ranks), np.arange(ranks.min(), ranks.min() + len(hex_ids)))
        # ... ordered ascending by hex index (stable tie, spec §5.1/§6.1).
        assert np.array_equal(ranks, np.sort(ranks))

    # The order must not depend on the input store order: same distances over
    # the same hexes, same tie-break -> identical rank arrays.
    reversed_ranking = build_hex_ranking(geom, list(reversed(stores)))
    assert np.array_equal(ranking.rank, reversed_ranking.rank)
    assert np.allclose(ranking.distance_meters, reversed_ranking.distance_meters)


# ---------------------------------------------------------------------------
# Addressless-store display fallback (Minor A, spec §5.2)
# ---------------------------------------------------------------------------


def test_store_address_addressless_in_tn_uses_coordinate_fallback():
    """An in-TN store with no street, housenumber, city, or state must show
    the coordinate pair, not a bare "TN" (spec §5.2); a city-only store keeps
    the pinned composition when a city is present."""
    bare = StoreRecord(
        id="1", name="DG #1", lat=35.11, lon=-86.22,
        city=None, street=None, state=None, in_tn=True,
    )
    expected = f"{bare.lat:.5f}, {bare.lon:.5f}"
    assert _store_address(bare, {}) == expected
    assert _store_address(bare, None) == expected

    city_only = dataclasses.replace(bare, city="Franklin")
    assert _store_address(city_only, {}) == "Franklin, TN"
    # Street-only keeps the street + default-state composition.
    street_only = dataclasses.replace(bare, street="Main St")
    assert _store_address(street_only, {}) == "Main St, TN"
