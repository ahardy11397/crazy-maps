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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
