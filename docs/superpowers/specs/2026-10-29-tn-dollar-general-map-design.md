# Tennessee Dollar General Map & Furthest Point Finder — Design Spec

**Date:** 2026-10-29  
**Project:** `crazy-maps`  
**Status:** Draft for User Review

---

## 1. Overview & Goals

Build a reproducible Python geospatial pipeline that:
1. Maps every **Dollar General** store in the state of Tennessee on an interactive HTML map.
2. Computes the exact point inside Tennessee's borders that is furthest from any Dollar General store (the "Dollar General Pole of Inaccessibility"), accounting for stores both inside Tennessee and just across the state border in neighboring states (`KY, VA, NC, GA, AL, MS, AR, MO`).
3. Visualizes the result in a self-contained Leaflet/Folium HTML map (`output/tn_dollar_general_map.html`) with store markers, the state boundary, the furthest point, the empty geodesic circle of radius $R$, lines to the nearest constraining stores, and a summary statistics overlay.

---

## 2. Requirements & Constraints

- **Data Source:** Public OpenStreetMap data via the Overpass API (with multi-endpoint fallback) and US Census Bureau cartographic boundary GeoJSON for Tennessee. No proprietary API keys required.
- **Border Store Handling:** Store queries cover a bounding box buffered by at least 50 km outside Tennessee's boundaries (`34.5°N to 37.2°N`, `-90.9°W to -81.0°W`). Only points inside Tennessee are eligible as the "furthest point", but all Dollar General stores (in-state and out-of-state) act as distance constraints.
- **Store Filtering & Deduplication:**
  - Include retail stores tagged with `name~"^Dollar General"`, `brand="Dollar General"`, or `brand:wikidata="Q145168"` (including *Dollar General* and *Dollar General Market*).
  - Exclude warehouses/distribution centers (`building=warehouse`, `industrial=*`, or names matching `"Distribution Center"`).
  - Convert OSM ways/relations to their centroids and deduplicate any points within 50 meters of each other (handling cases where both a building polygon and an amenity node exist for the same store).
- **Geometric Accuracy:**
  - Use planar projection **NAD83(2011) / Tennessee (`EPSG:6576`, meters)** for candidate generation via bounded Voronoi vertices, Voronoi ridge–boundary intersections, and state polygon boundary vertices.
  - Refine the top candidate(s) directly on the **WGS84 ellipsoid** (`pyproj.Geod(ellps="WGS84")`) using constrained local optimization (`scipy.optimize.minimize`) and verify against a dense grid across Tennessee.

---

## 3. Architecture & Module Design

```
crazy-maps/
├── data/                        # Cached GeoJSON datasets (gitignored or tracked as cache)
│   ├── tn_boundary.geojson
│   └── dollar_generals.geojson
├── output/
│   ├── tn_dollar_general_map.html
│   └── summary.json
├── tn_dollar_general/
│   ├── __init__.py
│   ├── data_fetcher.py          # Boundary & Overpass store fetching + deduplication
│   ├── geometry.py              # Voronoi + boundary intersection solver & geodesic refinement
│   └── map_builder.py           # Folium interactive HTML map generator
├── tests/
│   ├── test_data_fetcher.py     # Deduplication & filtering unit tests
│   ├── test_geometry.py         # Synthetic polygon exact solver & border constraint tests
│   └── test_map_builder.py      # HTML output & layer verification tests
├── main.py                      # CLI entry point
└── pyproject.toml               # Dependencies & pytest config
```

### 3.1 `tn_dollar_general/data_fetcher.py`
- **`fetch_tn_boundary(cache_path: Path, refresh: bool = False) -> shapely.Polygon | shapely.MultiPolygon`**
  - Downloads the high-resolution Tennessee state boundary GeoJSON (from US Census Cartographic Boundary files / public GitHub mirror) and caches it locally.
- **`fetch_dollar_generals(tn_polygon, cache_path: Path, refresh: bool = False) -> list[StoreRecord]`**
  - Queries Overpass API endpoints (`https://overpass-api.de/api/interpreter`, `https://overpass.kumi.systems/api/interpreter`, `https://lz4.overpass-api.de/api/interpreter`) using `out center tags;` within bbox `(34.5, -90.9, 37.2, -81.0)`.
  - Parses nodes, ways, and relations into `StoreRecord(id, name, lat, lon, city, street, state, in_tn)`.
  - Filters out distribution centers/warehouses, deduplicates locations within 50 meters using a spatial `KDTree`, labels `in_tn` using `shapely.prepared.prep(tn_polygon).contains(Point(lon, lat))`, and caches to `data/dollar_generals.geojson`.

### 3.2 `tn_dollar_general/geometry.py`
- **`find_furthest_point(tn_polygon, stores: list[StoreRecord]) -> FurthestPointResult`**
  1. **Projection:** Transforms `tn_polygon` and all `stores` from `EPSG:4326` (lon/lat) to `EPSG:6576` (NAD83 / Tennessee meters).
  2. **Candidate Generation:**
     - Computes `scipy.spatial.Voronoi` over the projected store coordinates.
     - **Interior Voronoi vertices:** Keeps all finite Voronoi vertices strictly inside `tn_polygon`.
     - **Voronoi ridge–boundary intersections:** For every Voronoi ridge (finite segments and semi-infinite rays clipped to a bounding box surrounding TN), computes geometric intersection points with `tn_polygon.boundary`.
     - **Polygon boundary vertices:** Includes all vertices of `tn_polygon.boundary`.
     - **Dense fallback grid:** Also includes a uniform 1 km verification grid inside `tn_polygon` to guarantee global coverage.
  3. **Candidate Evaluation:**
     - Uses `scipy.spatial.KDTree` on projected store coordinates to compute planar nearest-store distance for all candidates.
     - Selects the top 25 distinct candidates and evaluates their exact **WGS84 geodesic distance** (`pyproj.Geod(ellps="WGS84")`) to the nearest stores.
  4. **Geodesic Refinement:**
     - Runs a local Nelder-Mead optimization around the best candidate(s), maximizing the minimum WGS84 geodesic distance to the nearest stores subject to staying inside `tn_polygon`.
  5. **Return Value (`FurthestPointResult`):**
     - `lat`, `lon`: Exact coordinates of the furthest point in Tennessee.
     - `distance_meters`, `distance_km`, `distance_miles`: Exact WGS84 geodesic distance to the nearest Dollar General.
     - `nearest_stores`: List of the constraining Dollar General stores (typically 3 equidistant stores for an interior Voronoi vertex, or 2 equidistant stores if constrained by the state boundary) with their individual geodesic distances and bearings.

### 3.3 `tn_dollar_general/map_builder.py`
- **`build_map(tn_polygon, stores: list[StoreRecord], result: FurthestPointResult, output_path: Path) -> Path`**
  - Creates a `folium.Map` centered on Tennessee (`[35.86, -86.35]`, `zoom_start=7`) with clean basemap tiles (`CartoDB positron` and `OpenStreetMap` plus `Esri WorldImagery` satellite toggle).
  - **Layers:**
    1. **Tennessee State Boundary:** GeoJSON outline in crisp dark slate/navy.
    2. **Tennessee Dollar Generals (All Markers):** Individual yellow/gold & black themed `CircleMarker` points for every Dollar General in Tennessee, with popups showing store name, address, coordinates, and OSM link.
    3. **Tennessee Dollar Generals (Clustered View, optional toggle):** Marker cluster layer for overview exploration.
    4. **Neighboring State Dollar Generals (Buffer Layer):** Toggleable gray/muted markers showing out-of-state stores near the border.
    5. **Furthest Point & Empty Circle:**
       - Red/crimson target marker at `(result.lat, result.lon)` with detailed popup (coordinates, distance in miles/km, nearest stores).
       - Geodesic circle polygon of radius `result.distance_meters` centered on the furthest point, demonstrating zero Dollar Generals inside the circle.
       - Dashed lines connecting the furthest point to each of its nearest constraining Dollar General stores, with highlighted markers on those nearest stores.
  - **Floating Info Card:** Fixed HTML/CSS overlay panel in the bottom-left/top-right displaying total Tennessee Dollar General count, furthest point coordinates, distance in miles and kilometers, and names/locations of the nearest stores.

---

## 4. Verification & Testing Strategy

1. **Synthetic Geometry Tests (`tests/test_geometry.py`):**
   - **Interior Voronoi Vertex Case:** Square polygon `[0, 1]^2` with 4 stores near the corners — verify the solver finds the exact center `(0.5, 0.5)` and identifies the equidistant stores.
   - **Boundary Intersection Case:** Square polygon where stores are placed near the center — verify the solver finds the corner/edge of the polygon.
   - **Out-of-State Border Store Constraint:** Verify that a store placed just outside the polygon boundary prevents a boundary point adjacent to it from being falsely chosen as the furthest point.
2. **Data Filtering & Deduplication Tests (`tests/test_data_fetcher.py`):**
   - Verify duplicate OSM node + way representations within 50m collapse into a single store.
   - Verify distribution centers / warehouses are excluded.
3. **End-to-End Pipeline & Grid Verification:**
   - Run `main.py` to generate `output/tn_dollar_general_map.html` and `output/summary.json`, and assert that no point on a dense verification grid across Tennessee has a greater distance to a Dollar General than the reported optimum.
