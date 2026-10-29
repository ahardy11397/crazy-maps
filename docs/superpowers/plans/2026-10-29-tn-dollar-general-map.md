# Tennessee Dollar General Map & Furthest Point Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python geospatial pipeline that maps every Dollar General in Tennessee and computes the exact point inside Tennessee furthest from any Dollar General store (including border stores in neighboring states).

**Architecture:** Three modular Python components under `tn_dollar_general/` (`data_fetcher.py`, `geometry.py`, `map_builder.py`) driven by a `main.py` CLI. Store and boundary data are fetched from OpenStreetMap Overpass and US Census GeoJSON with local caching, solved geometrically in NAD83 / Tennessee (`EPSG:6576`) via bounded Voronoi + boundary intersections with WGS84 ellipsoid geodesic refinement, and rendered into a self-contained Folium HTML map.

**Tech Stack:** Python 3.11+, `shapely`, `scipy`, `pyproj`, `folium`, `requests`, `pytest` (managed via `uv` or `venv`).

**Spec:** `docs/superpowers/specs/2026-10-29-tn-dollar-general-map-design.md`

## Global Constraints

- Store bounding box query must cover `(34.5, -90.9, 37.2, -81.0)` (Tennessee + >=50 km border buffer).
- Include retail stores matching `name~"^Dollar General"`, `brand="Dollar General"`, or `brand:wikidata="Q145168"`; exclude warehouses/distribution centers (`building=warehouse`, `industrial=*`, or `"Distribution Center"` in name).
- Deduplicate stores within 50 meters of each other.
- Project to `EPSG:6576` (NAD83(2011) / Tennessee, meters) for planar candidate generation; refine and report final distances on the `WGS84` ellipsoid (`pyproj.Geod(ellps="WGS84")`).
- Output interactive map to `output/tn_dollar_general_map.html` and summary JSON to `output/summary.json`.

## Review Focus

1. **Infinite/unbounded Voronoi ridges near polygon borders:** `scipy.spatial.Voronoi` produces `-1` vertex indices for unbounded ridges on the convex hull of points; if those semi-infinite rays are not reconstructed and intersected with the polygon boundary, boundary-constrained maxima can be missed.
2. **MultiPolygon state boundaries:** Census state boundaries are often stored as a `MultiPolygon` (or a `GeoJSON` `FeatureCollection` containing a `MultiPolygon`); assuming a single `Polygon` with `.exterior` will raise `AttributeError`.
3. **Collinear or <=3 store inputs to Voronoi:** `scipy.spatial.Voronoi` requires >=4 non-coplanar points; small synthetic test inputs or degenerate clusters must still work via boundary + grid fallback without crashing.
4. **Overpass API timeouts / HTTP 429 / 504:** A single Overpass server frequently fails or rate-limits; `fetch_dollar_generals` must iterate through fallback endpoints when a request fails or returns non-JSON.
5. **OSM Ways/Relations vs. Nodes:** Many Dollar General stores in OSM are mapped as building footprints (`way` with `center: {lat, lon}`) rather than `node`s (`lat, lon` at root); parser must extract coordinates from either root or `center` and collapse node+way duplicates within 50m.

---

### Task 1: Project Setup & Data Fetcher (`tn_dollar_general/data_fetcher.py`)

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `tn_dollar_general/__init__.py`
- Create: `tn_dollar_general/data_fetcher.py`
- Test: `tests/test_data_fetcher.py`

**Interfaces:**
- Consumes: External Overpass JSON format and GeoJSON `FeatureCollection` / `Feature` / `Polygon` / `MultiPolygon`.
- Produces:
  - `@dataclass(frozen=True) class StoreRecord`: fields `id: str`, `name: str`, `lat: float`, `lon: float`, `city: str | None`, `street: str | None`, `state: str | None`, `in_tn: bool`
  - `parse_boundary_geojson(geojson_data: dict) -> shapely.Geometry` (returns `Polygon` or `MultiPolygon`)
  - `fetch_tn_boundary(cache_path: Path, refresh: bool = False) -> shapely.Geometry`
  - `parse_and_deduplicate_overpass(elements: list[dict], tn_geom: shapely.Geometry, dedup_distance_m: float = 50.0) -> list[StoreRecord]`
  - `fetch_dollar_generals(tn_geom: shapely.Geometry, cache_path: Path, refresh: bool = False) -> list[StoreRecord]`

- [ ] **Step 1: Initialize `pyproject.toml`, `.gitignore`, and environment, then write failing tests in `tests/test_data_fetcher.py`**

Create `pyproject.toml` with dependencies `shapely>=2.0`, `scipy>=1.11`, `pyproj>=3.6`, `folium>=0.15`, `requests>=2.31`, and dev dependency `pytest>=8.0`. Ignore `.venv/`, `__pycache__/`, `.pytest_cache/` in `.gitignore`.

Write `tests/test_data_fetcher.py`:
```python
from pathlib import Path
from unittest.mock import patch, MagicMock
from shapely.geometry import box, MultiPolygon
from tn_dollar_general.data_fetcher import (
    StoreRecord,
    parse_boundary_geojson,
    parse_and_deduplicate_overpass,
    fetch_dollar_generals,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_fetcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tn_dollar_general.data_fetcher'`

- [ ] **Step 3: Implement `tn_dollar_general/data_fetcher.py`**

Implement `StoreRecord`, `parse_boundary_geojson`, `fetch_tn_boundary`, `parse_and_deduplicate_overpass`, and `fetch_dollar_generals`.
- For `fetch_tn_boundary`, fetch Tennessee state polygon from `https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json` (filtering feature with `properties.name == "Tennessee"`) or Census TIGERweb/Cartographic GeoJSON fallback, saving only Tennessee's geometry to `cache_path`.
- For `parse_and_deduplicate_overpass`: project candidate `(lon, lat)` to `EPSG:6576` with `pyproj.Transformer` and use `scipy.spatial.KDTree` to cluster/deduplicate points within `dedup_distance_m` (preferring entries with richer address tags).
- Use `shapely.prepared.prep(tn_geom).intersects(Point(lon, lat))` to set `in_tn`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_fetcher.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore tn_dollar_general/__init__.py tn_dollar_general/data_fetcher.py tests/test_data_fetcher.py
git commit -m "feat: add boundary and Overpass Dollar General data fetcher with deduplication"
```

---

### Task 2: Exact Furthest Point Geometry Solver (`tn_dollar_general/geometry.py`)

**Files:**
- Create: `tn_dollar_general/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: `StoreRecord` from `tn_dollar_general.data_fetcher`, `shapely.Geometry` (`Polygon | MultiPolygon`).
- Produces:
  - `@dataclass(frozen=True) class NearestStoreInfo`: `store: StoreRecord`, `distance_meters: float`, `distance_miles: float`, `distance_km: float`, `bearing_degrees: float`
  - `@dataclass(frozen=True) class FurthestPointResult`: `lat: float`, `lon: float`, `distance_meters: float`, `distance_km: float`, `distance_miles: float`, `nearest_stores: list[NearestStoreInfo]`
  - `find_furthest_point(tn_geom: shapely.Geometry, stores: list[StoreRecord], grid_step_m: float = 2000.0) -> FurthestPointResult`
  - `geodesic_circle_coords(lat: float, lon: float, radius_meters: float, num_points: int = 180) -> list[tuple[float, float]]` (returns `(lat, lon)` pairs for rendering on Leaflet)

- [ ] **Step 1: Write failing tests in `tests/test_geometry.py`**

```python
import math
from shapely.geometry import box, MultiPolygon
from pyproj import Geod
from tn_dollar_general.data_fetcher import StoreRecord
from tn_dollar_general.geometry import find_furthest_point, geodesic_circle_coords

GEOD = Geod(ellps="WGS84")

def _make_store(idx: int, lat: float, lon: float, in_tn: bool = True) -> StoreRecord:
    return StoreRecord(id=str(idx), name=f"DG #{idx}", lat=lat, lon=lon, city=None, street=None, state="TN" if in_tn else "KY", in_tn=in_tn)

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tn_dollar_general.geometry'`

- [ ] **Step 3: Implement `tn_dollar_general/geometry.py`**

Implement `find_furthest_point` and `geodesic_circle_coords`:
- Project `tn_geom` and `stores` to `EPSG:6576` using `pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6576", always_xy=True)`.
- Generate candidate planar points `(x, y)` inside or on `tn_geom_proj`:
  1. All vertices of `tn_geom_proj` boundary rings (handling both `Polygon` and `MultiPolygon` via `shapely.get_parts`).
  2. Dense points sampled along `tn_geom_proj.boundary` every 250m (`line.interpolate`) plus a uniform interior grid spaced by `grid_step_m`.
  3. If `len(stores) >= 4`, compute `scipy.spatial.Voronoi(store_xy)` (adding 4 far-away dummy bounding box points at 10x bbox width so all Voronoi ridges in the region of interest are finite!), keep all Voronoi vertices inside `tn_geom_proj`, and intersect all Voronoi ridge segments with `tn_geom_proj.boundary`.
- Query `scipy.spatial.KDTree(store_xy)` on all candidates, pick the top 25 distinct candidates (at least 500m apart), convert to `(lon, lat)`, and run `scipy.optimize.minimize(..., method="Nelder-Mead")` on each candidate maximizing the minimum exact WGS84 geodesic distance (`Geod(ellps="WGS84").inv`) to its 15 nearest stores while penalizing points outside `tn_geom`.
- Identify all constraining stores within `1.005 * max_distance_m` (or at least the top 3 nearest stores) sorted by distance.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_geometry.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add tn_dollar_general/geometry.py tests/test_geometry.py
git commit -m "feat: implement exact Voronoi + geodesic furthest point solver"
```

---

### Task 3: Interactive Folium Map Builder & CLI (`tn_dollar_general/map_builder.py`, `main.py`)

**Files:**
- Create: `tn_dollar_general/map_builder.py`
- Create: `main.py`
- Test: `tests/test_map_builder.py`

**Interfaces:**
- Consumes: `StoreRecord` from `tn_dollar_general.data_fetcher`, `FurthestPointResult` and `geodesic_circle_coords` from `tn_dollar_general.geometry`.
- Produces:
  - `build_map(tn_geom: shapely.Geometry, stores: list[StoreRecord], result: FurthestPointResult, output_path: Path) -> Path`
  - `run_pipeline(data_dir: Path, output_dir: Path, refresh: bool = False) -> FurthestPointResult` in `main.py`

- [ ] **Step 1: Write failing tests in `tests/test_map_builder.py`**

```python
from pathlib import Path
from shapely.geometry import box
from tn_dollar_general.data_fetcher import StoreRecord
from tn_dollar_general.geometry import find_furthest_point
from tn_dollar_general.map_builder import build_map

def test_build_map_creates_self_contained_html_with_all_layers(tmp_path: Path):
    poly = box(-86.1, 35.7, -85.9, 35.9)
    stores = [
        StoreRecord(id="1", name="Dollar General #1", lat=35.72, lon=-86.08, city="Franklin", street="Main St", state="TN", in_tn=True),
        StoreRecord(id="2", name="Dollar General #2", lat=35.88, lon=-85.92, city="Smyrna", street="Oak Rd", state="TN", in_tn=True),
        StoreRecord(id="3", name="Dollar General Border", lat=35.95, lon=-86.0, city="Franklin", street="State Line", state="KY", in_tn=False),
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_map_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tn_dollar_general.map_builder'`

- [ ] **Step 3: Implement `tn_dollar_general/map_builder.py` and `main.py`**

In `tn_dollar_general/map_builder.py`:
- Create `folium.Map(location=[35.86, -86.35], zoom_start=7, tiles="CartoDB positron", prefer_canvas=True)` plus `OpenStreetMap` and `Esri.WorldImagery` tile layers.
- Add `folium.GeoJson` for Tennessee's boundary.
- Add a `FeatureGroup(name="Tennessee Dollar Generals (All Markers)", show=True)` with `CircleMarker`s (`#f5c518` fill, `#111111` border, radius 4) for every `in_tn` store with HTML popup (name, street/city, coordinates).
- Add a `FeatureGroup(name="Border State Dollar Generals (Buffer)", show=False)` with gray `CircleMarker`s for `not in_tn` stores.
- Add a `FeatureGroup(name="Furthest Point from a Dollar General", show=True)` containing:
  - Geodesic empty circle (`folium.Polygon` from `geodesic_circle_coords(result.lat, result.lon, result.distance_meters)`).
  - Dashed `folium.PolyLine` lines from `(result.lat, result.lon)` to each store in `result.nearest_stores[:3]`, plus highlighted markers for those constraining stores.
  - Prominent marker at `(result.lat, result.lon)` with `"Dollar General Pole of Inaccessibility"` popup and distance in miles/km.
- Inject a clean floating summary card (`position: fixed; bottom: 24px; left: 24px; z-index: 9999`) showing total TN store count, exact coordinates `(lat, lon)`, distance in miles and km, and the nearest stores.

In `main.py`:
- Implement `run_pipeline(data_dir: Path, output_dir: Path, refresh: bool = False) -> FurthestPointResult` and CLI (`--refresh`, `--data-dir`, `--output-dir`).
- Write `output/summary.json` with total TN stores, total border buffer stores, furthest point coordinates, distances (`distance_miles`, `distance_km`, `distance_meters`), and nearest store details.

- [ ] **Step 4: Run all unit tests and execute the live pipeline**

Run: `pytest -v && python main.py`
Expected: All unit tests PASS, `data/tn_boundary.geojson`, `data/dollar_generals.geojson`, `output/tn_dollar_general_map.html`, and `output/summary.json` are generated, and terminal prints the exact coordinates and distance of the point in Tennessee furthest from a Dollar General.

- [ ] **Step 5: Commit**

```bash
git add tn_dollar_general/map_builder.py main.py tests/test_map_builder.py
git commit -m "feat: add interactive Folium map builder and CLI pipeline"
```
