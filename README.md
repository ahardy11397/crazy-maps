# crazy-maps — Tennessee Dollar General Pole of Inaccessibility

Finds the exact point in Tennessee that is **furthest from any Dollar General store**
(the geodesic pole of inaccessibility with respect to Dollar General coverage) and
renders it as a self-contained interactive map.

Latest run: **658** stores inside Tennessee (+ 792 border-buffer stores analyzed),
optimum ≈ **17.3 miles** from the nearest Dollar General. Store coverage changes on
OpenStreetMap constantly, so results will drift over time.

## How it works

1. **Boundary** — fetches the U.S. Census 1:500k state polygon (Tennessee GeoJSON),
   with fallback source URLs, cached locally in `data/tn_boundary.geojson`.
2. **Stores** — queries the Overpass API (with endpoint fallback) for nodes, ways and
   relations matching Dollar General within Tennessee buffered by 50 km;
   spatially deduplicates co-located node/way representations within 50 m
   (EPSG:6575) and filters out warehouses/distribution centers. Cached in
   `data/dollar_generals.geojson`.
3. **Solver** — candidate generation in EPSG:6575 (Voronoi vertices and
   ridge-boundary intersections with dummy points to make ridges finite, dense
   boundary samples, and a 1 km interior grid), then local refinement of the top
   candidates on the exact **WGS84 ellipsoid** via Nelder-Mead. The reported optimum
   is audited against a dense verification grid per spec §4.3.
4. **Map** — a self-contained Folium HTML file with layers for the state boundary,
   in-TN / border stores, the largest empty circle, dashed lines to the three
   constraining stores, and popups linking each store to OpenStreetMap. A summary
   JSON (`output/summary.json`) is written alongside.

## Install

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

```bash
uv run python main.py            # use cached data if present
uv run python main.py --refresh  # re-fetch boundary and stores from live APIs
```

Optional flags: `--data-dir DIR` (default `data/`) and `--output-dir DIR`
(default `output/`).

Outputs:

- `output/tn_dollar_general_map.html` — open in any browser, no server needed
- `output/summary.json` — machine-readable result (coordinates, distances, nearest stores)

## Development

```bash
uv run pytest -q   # 22 tests: parsing, deduplication, solver, map output, pipeline
```

## Data sources & attribution

- Map data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors,
  available under the Open Database License (ODbL); queried via
  [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API).
- State boundaries from the [U.S. Census Bureau](https://www.census.gov/) 2010
  1:500k cartographic boundary files.

This project is an independent geographic analysis and is not affiliated with,
endorsed by, or sponsored by Dollar General or its affiliates.
