"""Hex lattice over the projected state, planar remoteness ranking, and
geodesic display distances.

Pinned geometry (spec): flat-top hexes with face-to-face width ``hex_across_m``
(across-flats 1000 m by default). Circumradius ``R = hex_across_m / sqrt(3)``;
even-r layout with column pitch ``dx = hex_across_m`` (1000 m), row pitch
``dy = 3R/2`` (~866.03 m), odd rows shifted ``dx/2`` right; vertices at angles
30 + 60k degrees so left/right sides are the flats. Grid origin = SW corner of
the projected (EPSG:6575) bounds of ``tn_geom``; only hex centers strictly
inside ``tn_geom`` are kept.

Ranking: rank 1 = the most remote hex (max planar distance to the nearest store
center), ties broken by hex id ascending. ``neighbors_of_rank`` returns
``(hex at rank k-1, hex at rank k+1)`` with ``None`` at the ends — rank 1 has no
upper neighbor because it is the most remote, and rank N has no lower one.

All heavy work is vectorized: one projection pass for centers + stores, a single
``scipy.spatial.KDTree`` query for the planar pass, and one vectorized
``GEOD.inv`` pass for the WGS84 geodesic display distances.

Click lookup / data block: ``build_click_table`` snaps a coarse (0.005 deg)
bucket grid over the hex centers' clip extent to the geometrically nearest hex
center (0xFFFFFFFF sentinel for cells beyond the clip bounds), and
``encode_data_block`` packs the ranking, quantized centers, click table and
store display strings into the little-endian base64 JSON block the JS decoder
consumes.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)  # headless PNG rendering only, per spec
import matplotlib.collections
import matplotlib.pyplot

import numpy as np
import pyproj
import scipy.spatial
import shapely
from shapely.prepared import prep

from tn_dollar_general.data_fetcher import StoreRecord
from tn_dollar_general.geometry import GEOD, METERS_PER_MILE

_DEFAULT_HEX_ACROSS_M = 1000.0


@dataclass(frozen=True)
class HexRanking:
    """Hex centers inside ``tn_geom`` ranked by remoteness from the nearest store.

    ``rank[i]`` is the 1-based rank of hex ``i`` (rank 1 = most remote).
    ``store_idx[i]`` indexes the stores list passed to :func:`build_hex_ranking`.
    """

    centers_latlon: np.ndarray      # (N, 2) float64, lat/lon
    centers_m: np.ndarray           # (N, 2) float64, EPSG:6575 x/y
    distance_meters: np.ndarray     # (N,) float64 planar distance to nearest store
    geodesic_miles: np.ndarray      # (N,) float64 exact WGS84 miles, hex center -> nearest store
    rank: np.ndarray                # (N,) int32, 1-based; rank[i] = rank of hex i; rank 1 = max distance
    store_idx: np.ndarray           # (N,) int32 index into the *stores list passed in*
    hex_across_m: float
    ranks: int                      # N, number of ranked hexes

    def by_rank(self, k: int) -> int:
        """Hex index at rank ``k`` (``k`` in 1..ranks)."""
        if not 1 <= k <= self.ranks:
            raise ValueError(f"rank k must be in 1..{self.ranks}, got {k}")
        return int(np.flatnonzero(self.rank == k)[0])

    def neighbors_of_rank(self, k: int) -> tuple[int | None, int | None]:
        """``(hex at rank k-1, hex at rank k+1)``; a slot is ``None`` at the ends.

        Rank 1 (the most remote hex) has no upper neighbor and rank ``ranks``
        has no lower one.
        """
        if not 1 <= k <= self.ranks:
            raise ValueError(f"rank k must be in 1..{self.ranks}, got {k}")
        worse: int | None = self.by_rank(k - 1) if k > 1 else None
        better: int | None = self.by_rank(k + 1) if k < self.ranks else None
        return worse, better

    def top_k(self, k: int = 10) -> list[int]:
        """Hex indices for ranks 1..k, most remote first."""
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        return [self.by_rank(i) for i in range(1, min(k, self.ranks) + 1)]


def build_hex_ranking(
    tn_geom: shapely.Geometry,
    stores: list[StoreRecord],
    hex_across_m: float = _DEFAULT_HEX_ACROSS_M,
) -> HexRanking:
    """Build the hex lattice over ``tn_geom`` and rank hexes by remoteness.

    Planar work is done in EPSG:6575 (same projection as ``geometry.py``). The
    planar distance of a hex center is the distance to its nearest store center;
    ``geodesic_miles`` is the exact WGS84 geodesic distance between the hex
    center and that same nearest store (one vectorized pass at build time).
    """
    if hex_across_m <= 0:
        raise ValueError(f"hex_across_m must be positive, got {hex_across_m}")
    if not stores:
        raise ValueError("stores list cannot be empty")
    if tn_geom.is_empty:
        raise ValueError("tn_geom cannot be empty")

    to_proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6575", always_xy=True)
    to_wgs84 = pyproj.Transformer.from_crs("EPSG:6575", "EPSG:4326", always_xy=True)

    # --- Project stores and geometry to planar EPSG:6575 -------------------
    store_lons = np.array([s.lon for s in stores], dtype=float)
    store_lats = np.array([s.lat for s in stores], dtype=float)
    s_xs, s_ys = to_proj.transform(store_lons, store_lats)
    store_xy = np.column_stack([s_xs, s_ys])

    tn_geom_proj = shapely.transform(
        tn_geom,
        lambda coords: np.column_stack(to_proj.transform(coords[:, 0], coords[:, 1])),
    )

    # --- Pinned flat-top even-r lattice from the SW corner of the bounds ---
    minx, miny, maxx, maxy = tn_geom_proj.bounds
    dx = hex_across_m                      # column pitch (1000 m at default)
    R = hex_across_m / math.sqrt(3.0)      # circumradius
    dy = 1.5 * R                           # row pitch (~866.03 m at default)

    n_cols = int(math.ceil((maxx - minx) / dx)) + 2   # +2: odd-row right shift + slack
    n_rows = int(math.ceil((maxy - miny) / dy)) + 1
    i = np.arange(n_cols, dtype=float)
    j = np.arange(n_rows, dtype=float)

    X = minx + i[None, :] * dx + (dx / 2.0) * (j[:, None] % 2)  # odd rows shifted dx/2 right
    Y = miny + j[:, None] * dy + np.zeros_like(X)

    # Keep only centers strictly inside the projected geometry (prep == contains).
    prep_geom = prep(tn_geom_proj)
    inside = prep_geom.contains(shapely.points(X, Y))
    if not np.any(inside):
        raise ValueError("no hex centers fall inside tn_geom for the given hex size")

    centers_x = X[inside]
    centers_y = Y[inside]
    centers_m = np.column_stack([centers_x, centers_y])
    N = len(centers_m)

    # Centers back to lat/lon (single vectorized inverse pass).
    c_lons, c_lats = to_wgs84.transform(centers_x, centers_y)
    centers_latlon = np.column_stack([c_lats, c_lons])

    # --- Planar pass: KDTree over store centers -----------------------------
    tree = scipy.spatial.KDTree(store_xy)
    distance, nearest = tree.query(centers_m, k=1)

    # --- Geodesic display pass: hex center -> its nearest store -------------
    _, _, geo_m = GEOD.inv(c_lons, c_lats, store_lons[nearest], store_lats[nearest])
    geodesic_miles = geo_m / METERS_PER_MILE

    # --- Ranking: most remote first, ties by hex id ascending ---------------
    hex_ids = np.arange(N)
    order = np.lexsort((hex_ids, -distance))
    rank = np.empty(N, dtype=np.int32)
    rank[order] = np.arange(1, N + 1, dtype=np.int32)

    return HexRanking(
        centers_latlon=np.ascontiguousarray(centers_latlon, dtype=np.float64),
        centers_m=np.ascontiguousarray(centers_m, dtype=np.float64),
        distance_meters=np.ascontiguousarray(distance, dtype=np.float64),
        geodesic_miles=np.ascontiguousarray(geodesic_miles, dtype=np.float64),
        rank=rank,
        store_idx=np.ascontiguousarray(nearest, dtype=np.int32),
        hex_across_m=float(hex_across_m),
        ranks=int(N),
    )


def hex_vertices_m(centers: np.ndarray, hex_across_m: float) -> np.ndarray:
    """Corner vertices of flat-top hexes at ``centers`` in projected meters.

    Returns an ``(N, 6, 2)`` array of x/y vertices ordered counterclockwise
    starting at 30 degrees (30, 90, ..., 330), so left/right sides are the flats.
    """
    if hex_across_m <= 0:
        raise ValueError(f"hex_across_m must be positive, got {hex_across_m}")
    centers = np.asarray(centers, dtype=np.float64)
    R = hex_across_m / math.sqrt(3.0)
    angles_deg = 30.0 + 60.0 * np.arange(6)
    angles = np.radians(angles_deg)
    offsets = R * np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
    return centers[:, None, :] + offsets[None, :, :]


# ---------------------------------------------------------------------------
# Click-lookup bucket grid + encoded data block
# ---------------------------------------------------------------------------

_CLICK_SENTINEL = 0xFFFFFFFF  # uint32 sentinel: no ranked hex resolves to this cell

_CLIP_PROJ = "EPSG:6575"


@dataclass(frozen=True)
class ClickTable:
    """Bucket grid over the hex centers' lat/lon (clip) extent for O(1) clicks.

    ``idx`` is row-major over ``(lat_cells, lon_cells)``: cell ``(i, j)`` spans
    ``[lat_min + i*cell_deg, lat_min + (i+1)*cell_deg)`` x
    ``[lon_min + j*cell_deg, lon_min + (j+1)*cell_deg)`` and its value is
    ``idx[i * lon_cells + j]`` — the hex index of the geometrically nearest hex
    center, or :data:`_CLICK_SENTINEL` (0xFFFFFFFF) when the cell's own center
    lies beyond the clip bounds. A click outside the grid resolves to the
    sentinel too (checked by the JS decoder's cell math).
    """

    cell_deg: float                 # 0.005 (pinned default)
    lat_min: float
    lon_min: float
    lat_cells: int
    lon_cells: int
    idx: np.ndarray                 # uint32 (lat_cells*lon_cells,)


def build_click_table(hex_ranking: HexRanking, cell_deg: float = 0.005) -> ClickTable:
    """Snap every bucket cell to the geometrically nearest hex center.

    The grid covers the clip extent (the hex centers' min/max lat/lon) in
    ``cell_deg`` steps; the ragged edge cells produced by the ceil are kept in
    the array so any click inside the grid has a slot, but cells whose center
    is beyond the clip bounds carry the sentinel instead of a hex index.
    Distance is the WGS84 small-angle approximation: degrees with longitude
    scaled by ``cos(mid-lat)``, accurate enough at ~0.5 km cells.
    """
    if cell_deg <= 0:
        raise ValueError(f"cell_deg must be positive, got {cell_deg}")

    centers = np.asarray(hex_ranking.centers_latlon, dtype=np.float64)
    lats = centers[:, 0]
    lons = centers[:, 1]
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())

    # Ceil with an epsilon guard so spans that are (numerically) a whole number
    # of cells do not grow one spurious extra row/column.
    lat_cells = max(1, int(math.ceil((lat_max - lat_min) / cell_deg - 1e-9)))
    lon_cells = max(1, int(math.ceil((lon_max - lon_min) / cell_deg - 1e-9)))

    # Cell centers, row-major: i (lat) outer, j (lon) inner.
    cell_lats = lat_min + (np.arange(lat_cells) + 0.5) * cell_deg
    cell_lons = lon_min + (np.arange(lon_cells) + 0.5) * cell_deg
    q_lat = np.repeat(cell_lats, lon_cells)
    q_lon = np.tile(cell_lons, lat_cells)

    cos_mid = math.cos(math.radians(0.5 * (lat_min + lat_max)))
    tree = scipy.spatial.KDTree(np.column_stack([lats, lons * cos_mid]))
    _, nearest = tree.query(np.column_stack([q_lat, q_lon * cos_mid]), k=1)

    idx = nearest.astype(np.uint32)
    beyond = (
        (q_lat > lat_max)
        | (q_lat < lat_min)
        | (q_lon > lon_max)
        | (q_lon < lon_min)
    )
    idx[beyond] = _CLICK_SENTINEL

    return ClickTable(
        cell_deg=float(cell_deg),
        lat_min=lat_min,
        lon_min=lon_min,
        lat_cells=lat_cells,
        lon_cells=lon_cells,
        idx=np.ascontiguousarray(idx, dtype=np.uint32),
    )


def _b64_le(array: np.ndarray) -> str:
    """Little-endian base64 encoding of a numpy array's raw bytes."""
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _quantize_u16(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Quantize ``values`` over ``[lo, hi]`` into uint16 in 0..65535."""
    span = hi - lo
    if span <= 0:
        return np.zeros(len(values), dtype="<u2")
    q = np.round((np.asarray(values, dtype=np.float64) - lo) / span * 65535.0)
    return np.clip(q, 0.0, 65535.0).astype("<u2")


def _store_address(store: StoreRecord, store_display: dict[str, str]) -> str:
    """Display address for one store, mirroring the pinned composition order.

    Task 2's enrichment value wins when present and non-empty; otherwise the
    store's own housenumber+street, city, state triple is joined ("TN" for
    in-TN stores without a state); otherwise the plain lat/lon coordinate
    pair is the fallback for store-less/empty entries.
    """
    display = store_display.get(store.id, "") if store_display else ""
    if not display:
        first_part = " ".join(part for part in (store.housenumber, store.street) if part)
        state = store.state or ("TN" if store.in_tn else None)
        display = ", ".join(part for part in (first_part, store.city, state) if part)
    if not display:
        display = f"{store.lat:.5f}, {store.lon:.5f}"
    return display


def encode_data_block(
    hex_ranking: HexRanking,
    click_table: ClickTable,
    store_display: dict[str, str],
    stores: list[StoreRecord],
) -> dict:
    """Encode the pinned JSON data block consumed by the map's JS decoder.

    All arrays are little-endian base64 over the raw bytes: ``dist_mi_q``
    uint16 = ``round(geodesic_miles * 1000)``; ``rank_order`` uint32 = hex
    index at rank 1..N; ``store_idx`` uint32; ``centers`` uint16 (lat, lon)
    pairs quantized over the clip lat/lon ranges. ``top10`` is a plain list of
    hex indices, most remote first; it always has length 10 and is padded with
    the sentinel (0xFFFFFFFF) when the ranking holds fewer than 10 hexes.
    ``click`` carries the bucket grid verbatim plus its base64 uint32 ``idx``.
    Everything in the returned dict is JSON-serializable (no numpy types);
    ``clip`` is derived here from the hex centers' min/max lat/lon and must
    agree with the ``click_table`` built from the same ranking.
    """
    centers = np.asarray(hex_ranking.centers_latlon, dtype=np.float64)
    lats = centers[:, 0]
    lons = centers[:, 1]
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())
    n_hexes = int(hex_ranking.ranks)
    n_stores = int(len(stores))

    dist_mi_q = np.clip(
        np.round(np.asarray(hex_ranking.geodesic_miles, dtype=np.float64) * 1000.0),
        0.0,
        65535.0,
    ).astype("<u2")

    # ``rank`` is a permutation of 1..N, so the stable argsort yields exactly
    # the hex index at rank 1..N in slot order.
    rank_order = np.argsort(np.asarray(hex_ranking.rank), kind="stable").astype("<u4")
    store_idx = np.asarray(hex_ranking.store_idx, dtype=np.int64).astype("<u4")
    centers_q = np.column_stack(
        [
            _quantize_u16(lats, lat_min, lat_max),
            _quantize_u16(lons, lon_min, lon_max),
        ]
    ).astype("<u2")

    top10 = list(hex_ranking.top_k(10))
    top10 = top10 + [_CLICK_SENTINEL] * (10 - len(top10))

    stores_out = [
        {
            "id": str(store.id),
            "name": str(store.name),
            "address": _store_address(store, store_display),
        }
        for store in stores
    ]

    return {
        "format": 1,
        "totals": {"hexes": n_hexes, "stores": n_stores},
        "clip": {
            "lat": [lat_min, lat_max],
            "lon": [lon_min, lon_max],
            "proj": _CLIP_PROJ,
        },
        "dist_mi_q": _b64_le(dist_mi_q),
        "rank_order": _b64_le(rank_order),
        "store_idx": _b64_le(store_idx),
        "centers": _b64_le(centers_q),
        "click": {
            "cell_deg": float(click_table.cell_deg),
            "lat_min": float(click_table.lat_min),
            "lon_min": float(click_table.lon_min),
            "lat_cells": int(click_table.lat_cells),
            "lon_cells": int(click_table.lon_cells),
            "idx": _b64_le(click_table.idx),
        },
        "top10": [int(i) for i in top10],
        "stores": stores_out,
    }


# ---------------------------------------------------------------------------
# Remoteness raster overlay: one transparent-background PNG of the hex field
# ---------------------------------------------------------------------------

# Pinned blue ramp (spec): componentwise linear blend of the two endpoint RGB
# triples only — no intermediate palette stops. ``distance_norm`` 0 (nearest a
# store, lightest) → 1 (rank 1, deepest navy).
_RAMP_LO_RGB = np.array([0xE8, 0xF4, 0xFF], dtype=np.float64) / 255.0
_RAMP_HI_RGB = np.array([0x0F, 0x1F, 0x45], dtype=np.float64) / 255.0

# Small-extent resolution floor (see render_remoteness_png): the pinned default
# builder scale is px_per_hex px per hexagon-across (6 px per km at 1000 m
# hexes), which yields inspectable images at state scale; on small-area builds
# that scale would fall to a few dozen pixels wide, so the shorter image side is
# floored at this tile-scale minimum instead.
_MIN_RASTER_SIDE_PX = 512

_SAVE_DPI = 100.0


def _ramp_color(distance_norm: np.ndarray) -> np.ndarray:
    """Pinned remoteness RGBA colors: componentwise blend of the two endpoints.

    ``distance_norm`` is clamped to 0..1; the RGB triple lerps linearly and
    componentwise from :data:`_RAMP_LO_RGB` (0.0, lightest, near a store) to
    :data:`_RAMP_HI_RGB` (1.0, deepest navy, rank 1). Alpha is always 1.0 —
    hexes are fully opaque and the transparent mask comes from the figure
    background (``savefig(transparent=True)``), not from per-hex alpha.

    Returns an ``(..., 4)`` float64 array shaped like the input plus a trailing
    RGBA axis.
    """
    t = np.clip(np.asarray(distance_norm, dtype=np.float64), 0.0, 1.0)[..., None]
    rgb = _RAMP_LO_RGB + t * (_RAMP_HI_RGB - _RAMP_LO_RGB)
    alpha = np.ones_like(rgb[..., :1])
    return np.concatenate([rgb, alpha], axis=-1)


def render_remoteness_png(
    hex_ranking: HexRanking,
    output_path: Path | str,
    px_per_hex: int = 6,
) -> Path:
    """Render the whole ranked hex field to ONE transparent-background PNG.

    Anti-goal honored: the browser consumes a single image — never thousands of
    vector hex shapes. Exactly one ``matplotlib`` ``PolyCollection`` carries
    every hexagon from ``hex_vertices_m`` with facecolors from the pinned ramp
    (:func:`_ramp_color`) over ``distance / max(distance)``.

    Geometry: the image spans the bounds of the hex vertices themselves (a
    superset of the projected EPSG:6575 bounds, so no hexagon is clipped at an
    image edge) at the true projected-meters aspect; ``axes`` fill the figure
    edge to edge with no ticks or frame. Edges are drawn un-antialiased in each
    hexagon's own face color, so adjacent hexes share exact boundary pixels and
    pixel alpha is exactly 1.0 inside the hex field and 0 outside it.

    Pixel size: ``px_per_hex`` pixels per across-flats hexagon span
    (``px_per_hex`` px per km at the pinned 1000 m default), brokered through a
    fixed 100 dpi canvas. When that scale would floor the shorter image side
    below :data:`_MIN_RASTER_SIDE_PX` (tiny build extents), the scale is raised
    to the floor instead — never lowered at state scale.
    """
    if px_per_hex <= 0:
        raise ValueError(f"px_per_hex must be positive, got {px_per_hex}")

    verts = hex_vertices_m(hex_ranking.centers_m, hex_ranking.hex_across_m)
    minx = float(verts[..., 0].min())
    maxx = float(verts[..., 0].max())
    miny = float(verts[..., 1].min())
    maxy = float(verts[..., 1].max())

    distance = np.asarray(hex_ranking.distance_meters, dtype=np.float64)
    dmax = float(distance.max()) if distance.size else 0.0
    distance_norm = distance / dmax if dmax > 0.0 else np.zeros_like(distance)

    # Builder scale: px_per_hex px per hexagon-across, floored on the shorter
    # side for very small extents (test fixtures); never lowered at state scale.
    min_span = min(maxx - minx, maxy - miny)
    scale = max(px_per_hex / hex_ranking.hex_across_m, _MIN_RASTER_SIDE_PX / min_span)
    width_px = int(round((maxx - minx) * scale))
    height_px = int(round((maxy - miny) * scale))

    fig = matplotlib.pyplot.figure(
        figsize=(width_px / _SAVE_DPI, height_px / _SAVE_DPI), dpi=_SAVE_DPI
    )
    try:
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_axis_off()
        ax.add_collection(
            matplotlib.collections.PolyCollection(
                verts,
                facecolors=_ramp_color(distance_norm),
                edgecolors=_ramp_color(distance_norm),
                antialiaseds=False,  # exact coverage: no seams, alpha strictly 0/1
                linewidths=0.0,
            )
        )
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=_SAVE_DPI, transparent=True)
    finally:
        matplotlib.pyplot.close(fig)
    return output_path
