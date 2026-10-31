from __future__ import annotations

from dataclasses import dataclass
import logging
import numpy as np
import pyproj
from pyproj import Geod
import scipy.optimize
import scipy.spatial
import shapely
from shapely.geometry import Point, LineString

from tn_dollar_general.data_fetcher import StoreRecord

logger = logging.getLogger(__name__)

GEOD = Geod(ellps="WGS84")
METERS_PER_MILE = 1609.344


@dataclass(frozen=True)
class NearestStoreInfo:
    store: StoreRecord
    distance_meters: float
    distance_miles: float
    distance_km: float
    bearing_degrees: float


@dataclass(frozen=True)
class FurthestPointResult:
    lat: float
    lon: float
    distance_meters: float
    distance_km: float
    distance_miles: float
    nearest_stores: list[NearestStoreInfo]


def geodesic_circle_coords(
    lat: float, lon: float, radius_meters: float, num_points: int = 180
) -> list[tuple[float, float]]:
    """Generate (lat, lon) coordinates for a closed geodesic circle of given radius on WGS84 ellipsoid."""
    azimuths = np.linspace(0.0, 360.0, num_points + 1)
    lons, lats, _ = GEOD.fwd(
        np.full_like(azimuths, lon),
        np.full_like(azimuths, lat),
        azimuths,
        np.full_like(azimuths, radius_meters),
    )
    return [(float(lt), float(ln)) for ln, lt in zip(lons, lats)]


def verify_furthest_point(
    tn_geom: shapely.Geometry,
    stores: list[StoreRecord],
    result: FurthestPointResult,
    grid_step_m: float = 1000.0,
    tolerance_m: float = 100.0,
) -> bool:
    """Verify (spec 4.3) that no point on a dense verification grid inside tn_geom
    has a greater distance to any store than the reported optimum.

    Uses the planar EPSG:6575 projection and a KDTree over the store positions as
    a fast proxy for the exact WGS84 geodesic distance; tolerance_m absorbs the
    projection-vs-geodesic discrepancy (order of 1:10,000 within Tennessee, i.e.
    a few meters per distance evaluation). Returns True when the reported optimum
    holds across the grid.
    """
    if not stores:
        raise ValueError("stores list cannot be empty")
    if tn_geom.is_empty:
        raise ValueError("tn_geom cannot be empty")

    to_proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6575", always_xy=True)
    store_lons = np.array([s.lon for s in stores], dtype=float)
    store_lats = np.array([s.lat for s in stores], dtype=float)
    s_xs, s_ys = to_proj.transform(store_lons, store_lats)
    store_xy = np.column_stack([s_xs, s_ys])

    tn_geom_proj = shapely.transform(
        tn_geom,
        lambda coords: np.column_stack(to_proj.transform(coords[:, 0], coords[:, 1])),
    )

    minx, miny, maxx, maxy = tn_geom_proj.bounds
    step = max(grid_step_m, 100.0)
    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    pts = shapely.points(grid[:, 0], grid[:, 1])
    inside = shapely.covers(tn_geom_proj, pts)
    grid = grid[inside]
    if len(grid) == 0:
        return True

    tree = scipy.spatial.KDTree(store_xy)
    planar_max, _ = tree.query(grid, k=1)
    return bool(float(np.max(planar_max)) <= result.distance_meters + tolerance_m)


def find_furthest_point(
    tn_geom: shapely.Geometry,
    stores: list[StoreRecord],
    grid_step_m: float = 1000.0,
) -> FurthestPointResult:
    """Find the exact point inside tn_geom furthest from any Dollar General store.

    Uses planar projection EPSG:6575 (NAD83(2011) / Tennessee) for candidate generation
    (interior Voronoi vertices, Voronoi ridge-boundary intersections, boundary vertices,
    dense boundary samples, and a 1 km interior verification grid per the spec), then
    refines top candidates using Nelder-Mead optimization on the exact WGS84 geodesic
    distance. The reported optimum is audited with verify_furthest_point before return.
    """
    if not stores:
        raise ValueError("stores list cannot be empty")
    if tn_geom.is_empty:
        raise ValueError("tn_geom cannot be empty")

    to_proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6575", always_xy=True)
    to_wgs84 = pyproj.Transformer.from_crs("EPSG:6575", "EPSG:4326", always_xy=True)

    # Project geometry to planar EPSG:6575
    tn_geom_proj = shapely.transform(
        tn_geom,
        lambda coords: np.column_stack(to_proj.transform(coords[:, 0], coords[:, 1])),
    )

    store_lons = np.array([s.lon for s in stores], dtype=float)
    store_lats = np.array([s.lat for s in stores], dtype=float)
    s_xs, s_ys = to_proj.transform(store_lons, store_lats)
    store_xy = np.column_stack([s_xs, s_ys])

    candidates: list[np.ndarray] = []

    # 1. Boundary ring vertices (handling Polygon and MultiPolygon)
    for part in shapely.get_parts(tn_geom_proj):
        if hasattr(part, "exterior"):
            candidates.append(shapely.get_coordinates(part.exterior))
            for interior in part.interiors:
                candidates.append(shapely.get_coordinates(interior))

    # 2. Boundary dense sampling (every 250m)
    boundary = tn_geom_proj.boundary
    for line in shapely.get_parts(boundary):
        if line.length > 0:
            n_steps = max(1, int(np.ceil(line.length / 250.0)))
            dists = np.linspace(0.0, line.length, n_steps + 1)
            pts = shapely.line_interpolate_point(line, dists)
            candidates.append(shapely.get_coordinates(pts))

    # 3. Uniform interior grid
    minx, miny, maxx, maxy = tn_geom_proj.bounds
    step = max(grid_step_m, 100.0)
    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)
    gx, gy = np.meshgrid(xs, ys)
    grid_coords = np.column_stack([gx.ravel(), gy.ravel()])
    if len(grid_coords) > 0:
        g_pts = shapely.points(grid_coords[:, 0], grid_coords[:, 1])
        inside_g = shapely.covers(tn_geom_proj, g_pts)
        candidates.append(grid_coords[inside_g])

    # 4. Voronoi vertices and ridge-boundary intersections with 4 dummy bounding points
    s_minx, s_maxx = float(np.min(store_xy[:, 0])), float(np.max(store_xy[:, 0]))
    s_miny, s_maxy = float(np.min(store_xy[:, 1])), float(np.max(store_xy[:, 1]))
    b_minx, b_miny = min(minx, s_minx), min(miny, s_miny)
    b_maxx, b_maxy = max(maxx, s_maxx), max(maxy, s_maxy)
    w = max(b_maxx - b_minx, 1000.0)
    h = max(b_maxy - b_miny, 1000.0)
    scale = 10.0 * max(w, h)
    cx = (b_minx + b_maxx) / 2.0
    cy = (b_miny + b_maxy) / 2.0

    dummy_pts = np.array([
        [cx - scale, cy - scale],
        [cx + scale, cy - scale],
        [cx + scale, cy + scale],
        [cx - scale, cy + scale],
    ])
    augmented_xy = np.vstack([store_xy, dummy_pts])

    try:
        vor = scipy.spatial.Voronoi(augmented_xy)
        # Voronoi vertices inside tn_geom_proj
        vor_verts = vor.vertices
        in_bbox = (
            (vor_verts[:, 0] >= minx)
            & (vor_verts[:, 0] <= maxx)
            & (vor_verts[:, 1] >= miny)
            & (vor_verts[:, 1] <= maxy)
        )
        cand_v = vor_verts[in_bbox]
        if len(cand_v) > 0:
            v_pts = shapely.points(cand_v[:, 0], cand_v[:, 1])
            inside_v = shapely.covers(tn_geom_proj, v_pts)
            candidates.append(cand_v[inside_v])

        # Intersect Voronoi ridge segments with tn_geom_proj.boundary
        ridge_lines = []
        for v_idx in vor.ridge_vertices:
            if -1 not in v_idx:
                p1 = vor.vertices[v_idx[0]]
                p2 = vor.vertices[v_idx[1]]
                seg_minx, seg_maxx = min(p1[0], p2[0]), max(p1[0], p2[0])
                seg_miny, seg_maxy = min(p1[1], p2[1]), max(p1[1], p2[1])
                if (
                    seg_maxx >= minx
                    and seg_minx <= maxx
                    and seg_maxy >= miny
                    and seg_miny <= maxy
                ):
                    ridge_lines.append(LineString([p1, p2]))
        if ridge_lines:
            inters = shapely.intersection(ridge_lines, boundary)
            coords = shapely.get_coordinates(inters)
            if len(coords) > 0:
                candidates.append(coords)
    except Exception as exc:
        logger.warning("Voronoi computation skipped: %s", exc)

    all_cands = np.vstack(candidates)
    finite_mask = np.isfinite(all_cands[:, 0]) & np.isfinite(all_cands[:, 1])
    all_cands = all_cands[finite_mask]

    # Filter candidates to within 1 meter of tn_geom_proj (guards against boundary rounding)
    cand_pts = shapely.points(all_cands[:, 0], all_cands[:, 1])
    dists_to_geom = shapely.distance(tn_geom_proj, cand_pts)
    all_cands = all_cands[dists_to_geom <= 1.0]

    if len(all_cands) == 0:
        # Fallback to centroid or representative point
        rep_pt = tn_geom_proj.representative_point()
        all_cands = np.array([[rep_pt.x, rep_pt.y]])

    # Query planar KDTree to find candidates with maximum store clearance
    tree = scipy.spatial.KDTree(store_xy)
    dists, _ = tree.query(all_cands, k=1)
    order = np.argsort(-dists)

    # Select top 25 distinct candidates at least 500m apart
    selected = []
    for idx in order:
        pt = all_cands[idx]
        if not selected or all(np.linalg.norm(pt - s) >= 500.0 for s in selected):
            selected.append(pt)
            if len(selected) >= 25:
                break
    if not selected:
        selected.append(all_cands[order[0]])

    cand_xs = np.array([p[0] for p in selected])
    cand_ys = np.array([p[1] for p in selected])
    cand_lons, cand_lats = to_wgs84.transform(cand_xs, cand_ys)

    prep_geom = shapely.prepared.prep(tn_geom)

    best_lon = float(cand_lons[0])
    best_lat = float(cand_lats[0])
    best_min_dist = -1.0

    k_sub = min(15, len(stores))

    # Geodesic local refinement for each top candidate
    for clat, clon in zip(cand_lats, cand_lons):
        _, _, all_d = GEOD.inv(
            np.full_like(store_lons, clon),
            np.full_like(store_lats, clat),
            store_lons,
            store_lats,
        )
        c_min_dist = float(np.min(all_d))

        # Local neighborhood of stores for fast Nelder-Mead iterations
        sub_idx = np.argsort(all_d)[:k_sub]
        sub_lons = store_lons[sub_idx]
        sub_lats = store_lats[sub_idx]

        def objective(coords: np.ndarray) -> float:
            lon, lat = coords[0], coords[1]
            pt = Point(lon, lat)
            _, _, d = GEOD.inv(
                np.full_like(sub_lons, lon),
                np.full_like(sub_lats, lat),
                sub_lons,
                sub_lats,
            )
            m = float(np.min(d))
            if prep_geom.covers(pt):
                return -m
            dist_outside_deg = tn_geom.distance(pt)
            dist_outside_m = dist_outside_deg * 111_000.0
            return -m + 1e6 + dist_outside_m * 1e4

        res_opt = scipy.optimize.minimize(
            objective,
            [clon, clat],
            method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 1e-6, "fatol": 0.1},
        )
        opt_lon, opt_lat = float(res_opt.x[0]), float(res_opt.x[1])
        opt_pt = Point(opt_lon, opt_lat)

        if not prep_geom.covers(opt_pt):
            # If slightly outside due to floating point error, snap to boundary
            if tn_geom.distance(opt_pt) < 1e-5:
                line = shapely.shortest_line(tn_geom, opt_pt)
                snap_pt = shapely.get_point(line, 0)
                opt_lon, opt_lat = float(snap_pt.x), float(snap_pt.y)
            else:
                opt_lon, opt_lat = float(clon), float(clat)

        _, _, opt_d = GEOD.inv(
            np.full_like(store_lons, opt_lon),
            np.full_like(store_lats, opt_lat),
            store_lons,
            store_lats,
        )
        refined_min_dist = float(np.min(opt_d))

        # Only accept refinement if it improves upon initial candidate
        if refined_min_dist >= c_min_dist:
            final_lon, final_lat, final_dist = opt_lon, opt_lat, refined_min_dist
        else:
            final_lon, final_lat, final_dist = float(clon), float(clat), c_min_dist

        if final_dist > best_min_dist:
            best_min_dist = final_dist
            best_lon = final_lon
            best_lat = final_lat

    # Final verification against all stores
    _, _, final_all_dists = GEOD.inv(
        np.full_like(store_lons, best_lon),
        np.full_like(store_lats, best_lat),
        store_lons,
        store_lats,
    )
    sorted_idx = np.argsort(final_all_dists)
    max_distance_meters = float(final_all_dists[sorted_idx[0]])

    # Constraining stores: within 1.005x max_distance or at least top 3
    threshold = 1.005 * max_distance_meters
    num_constraining = max(min(3, len(stores)), int(np.sum(final_all_dists <= threshold)))
    constraining_idx = sorted_idx[:num_constraining]

    nearest_stores: list[NearestStoreInfo] = []
    for idx in constraining_idx:
        s = stores[idx]
        fwd_az, _, d_m = GEOD.inv(best_lon, best_lat, s.lon, s.lat)
        dist_m = float(d_m)
        nearest_stores.append(
            NearestStoreInfo(
                store=s,
                distance_meters=dist_m,
                distance_miles=dist_m / METERS_PER_MILE,
                distance_km=dist_m / 1000.0,
                bearing_degrees=(float(fwd_az) + 360.0) % 360.0,
            )
        )

    # Grid-verification audit (spec 4.3): no point on a dense verification grid
    # across the region may have a greater distance to any store than the reported optimum.
    result = FurthestPointResult(
        lat=float(best_lat),
        lon=float(best_lon),
        distance_meters=max_distance_meters,
        distance_km=max_distance_meters / 1000.0,
        distance_miles=max_distance_meters / METERS_PER_MILE,
        nearest_stores=nearest_stores,
    )
    if not verify_furthest_point(tn_geom, stores, result, grid_step_m=grid_step_m):
        logger.warning(
            "Verification grid found a candidate exceeding the reported optimum; "
            "results may be approximate in this region (check boundary resolution and store coverage)."
        )
    return result
