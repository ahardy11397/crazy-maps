# Hexagon Remoteness Ranking & Side-Rail Explorer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rank every 1 km hexagon in Tennessee by straight-line distance to the nearest Dollar General, and let a click on any hexagon open a side rail showing its rank, its miles, its nearest DG's address, plus the two hexagons ranked immediately above and below.

**Architecture:** All heavy work happens at build time in Python — a flat-top hex lattice inside `tn_geom` (~125k centers), a KDTree planar nearest-store pass in EPSG:6575, a WGS84 geodesic display-distance pass, ranking, a precomputed click-lookup bucket grid, and a single raster PNG of the shaded hex field. The Folium map gains one `ImageOverlay`, a right-hand rail (Podium + Top 10 tabs), and one ephemeral click-highlight; it never draws vector hexes.

**Tech Stack:** Python 3 (existing deps), scipy `KDTree`, pyproj (EPSG:6575 planar + WGS84 geodesic), shapely `prep`, matplotlib (new dep, Agg PNG only), folium `raster_layers.ImageOverlay`, Nominatim reverse geocoding (free, 1 req/s, cached), pytest.

**Spec:** `docs/superpowers/specs/2026-10-31-hex-remoteness-ranking-design.md` (execute together with its prior-dependency spec `2026-10-29-tn-dollar-general-map-design.md`)

## Global Constraints

- Rank **#1 = most remote** (farthest from any DG); equal distances tie-break deterministically by hex-id ascending.
- Hexagons: **flat-top, across-flats = 1000 m exactly**, centers kept only where `shapely prep(tn_geom).contains(center)`; expected order of magnitude ~125,000.
- Planar metric/ranking in **EPSG:6575** (`pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6575", always_xy=True)` — same projection `geometry.py` already uses; **not** 6576).
- Card **display distance** = exact WGS84 geodesic miles (`geometry.GEOD`), one build-time pass per hex→nearest-store pair; tolerance vs solver output: ≤ 0.05 mi in tests.
- Store set includes **border-buffer stores** (`in_tn=False` still constrains); a hexagon's nearest DG may be out-of-state.
- The generated HTML stays **self-contained**; raster embedded as a data-URI in `ImageOverlay`; data block embedded as typed-array base64 in JSON, ≤ ~1.5 MB; raster PNG ≤ ~6 MB.
- **One** baked raster image; **zero** baked vector hexes; exactly one ephemeral click-highlight shape allowed (a plain `L.circle`/`CircleMarker` at the hex center is the owner-approved highlight).
- Side rail **reserves its own column beside the map** (map viewport shrinks; nothing obscured); exact CSS selector for folium's map container is builder latitude, the reserved-width outcome is not.
- Existing features must be preserved: store markers, state boundary, focal point + empty circle + dashed lines, floating summary card. All existing tests must keep passing.
- No place/search box (decided non-goal).
- New dependencies limited to `matplotlib` (no Node toolchain, no server-side rendering).

## Review Focus

- Store with **no address at all** → card reads `coords — no address on record`, never an empty/garbled line. Pinned by `test_card_display_without_any_address_fields` in Task 6.
- **Nominatim unreachable / per-store failure** → enrichment is graceful, never raises, failed stores are simply missing from the result and NOT cached. Pinned by `test_enrich_failure_is_graceful_and_not_cached` in Task 2.
- `--refresh` re-fetches stores → previously geocoded addresses **survive** (separate id-keyed cache; only new ids get geocoded). Pinned by `test_enrich_uses_cache_and_only_geocodes_incomplete` in Task 2.
- **Clicks near the border / off-state** → inside TN with no ranked hex underneath: snap to nearest ranked hex; entirely off-state: rail untouched. Pinned by `test_click_table_snaps_inside_and_sentinel_offstate` in Task 4.
- **Hostile store names** (`<script>`, backticks, `${...}`) flowed into the rail's JSON → must pass through the same escaping policy as `_safe_tooltip`. Pinned by `test_store_display_is_escaped_for_rail_json` in Task 6.

---

### Task 1: Capture housenumber & postcode in `StoreRecord`

**Files:**
- Modify: `tn_dollar_general/data_fetcher.py` (dataclass `StoreRecord` ~L38-47; tag capture in `parse_and_deduplicate_overpass` ~L178-193 and dedup-merge ~L236-241; cache load ~L277-297 and save ~L330-352 of `fetch_dollar_generals`)
- Modify: `tests/test_data_fetcher.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `StoreRecord` gains trailing fields `housenumber: str | None = None`, `postcode: str | None = None` (defaults keep every existing constructor call and test working). Cache `data/dollar_generals.geojson` feature properties gain `"housenumber"` and `"postcode"` keys; the loader returns them; the dedup-merge fills them from suppressed co-located candidates, same as `city`/`street`. Task 2 and Task 6 consume these fields.

- [ ] **Step 1: Write the failing tests** in `tests/test_data_fetcher.py`, reusing the fixture style of `test_parse_overpass_extracts_nodes_ways_and_filters_warehouses` (a tiny square `tn_geometry` and raw Overpass element dicts):

```python
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
    # monkeypatch requests.post to a canned Overpass response as in the
    # existing fetch tests; build a store with housenumber/postcode, then
    stores = fetch_dollar_generals(tn_geom, tmp_path / "cache.geojson")
    reloaded = fetch_dollar_generals(tn_geom, tmp_path / "cache.geojson")
    assert reloaded[0].housenumber == stores[0].housenumber
    assert reloaded[0].postcode == stores[0].postcode

def test_dedup_merge_fills_housenumber_from_colocated_duplicate():
    # node + way for the same store within 50 m: node has addr:street, the
    # way also has addr:housenumber; kept record must gain the housenumber.
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_data_fetcher.py -k housenumber -v`
Expected: FAIL (`StoreRecord` has no attribute `housenumber` / properties missing)

- [ ] **Step 3: Implement** — capture `addr:housenumber`/`addr:postcode` where `city`/`street` are captured; extend `_tag_richness_score` keys list with the two new `addr:*` keys (same +10 scoring); add the two merge-fill lines mirroring `city`/`street`; add the two property keys in both cache branches.

- [ ] **Step 4: Run the full fetcher suite**

Run: `pytest tests/test_data_fetcher.py -v`
Expected: PASS (all pre-existing tests too)

- [ ] **Step 5: Commit**

```bash
git add tn_dollar_general/data_fetcher.py tests/test_data_fetcher.py
git commit -m "feat: capture addr housenumber and postcode in StoreRecord and cache"
```

---

### Task 2: Address enrichment via Nominatim + separate id-keyed cache

**Files:**
- Modify: `tn_dollar_general/data_fetcher.py` (append new client + `enrich_store_addresses`; add `import time` if absent)
- Modify: `tests/test_data_fetcher.py`

**Interfaces:**
- Consumes: `StoreRecord.housenumber/.street/.city/.state` (Task 1).
- Produces:
```python
class HTTPNominatimClient:
    """Polite Nominatim reverse client: exactly one request message then time.sleep(delay_seconds)."""
    def __init__(self, endpoint: str = "https://nominatim.openstreetmap.org/reverse",
                 delay_seconds: float = 1.05) -> None: ...
    def reverse(self, lat: float, lon: float) -> dict | None: ...

def enrich_store_addresses(
    stores: list[StoreRecord],
    address_cache_path: Path | str,
    refresh: bool = False,
    client: HTTPNominatimClient | None = None,   # None => HTTPNominatimClient(); inject a fake in tests
) -> dict[str, str]
```
Returns `{store_id: display_address}` where `display_address` flows straight into rail cards (e.g. `"707 New Life Ln, Cedar Grove, TN"`). Cache file format: `{"version": 1, "addresses": {osm_id: display_address}}`. Only stores lacking the complete triple `housenumber+street+city` (state defaults `"TN"` inside TN) get geocoded; per-store failures (client returns `None` or raises) are skipped, **not cached**, and never raise. Written to Task 3-independent use by Task 6/7.

- [ ] **Step 1: Write the failing tests**

```python
def test_enrich_geocodes_only_incomplete_and_uses_cache(tmp_path):
    # fake client records calls; 3 stores: store A complete (housenumber+street+city),
    # store B complete but pre-seeded in cache file, store C missing fields.
    # fake client returns {"address": {"road": "…", "house_number": "…", "city": "…", "state": "TN"}}
    assert fake.calls == [store C id]            # only C geocoded live
    assert result[A_id] uses A's own fields       # composed without any call
    assert result[B_id] == seeded value           # cache honored
    cache = json.loads((tmp_path / "store_addresses.json").read_text())
    assert cache["version"] == 1 and set(cache["addresses"]) == {C_id}

def test_enrich_failure_is_graceful_and_not_cached(tmp_path):
    # fake client raises for store C; return value lacks C; cache lacks C; no exception.

def test_enrich_display_composition_order():
    # fakes supply house_number="707", road="New Life Ln", city="Cedar Grove", state="TN"
    # -> "707 New Life Ln, Cedar Grove, TN"
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_data_fetcher.py -k enrich -v`
Expected: FAIL (`cannot import name 'enrich_store_addresses'`)

- [ ] **Step 3: Implement** — composition: parts joined `", "` in order house_number+road, city, state, dropping empties. `refresh=True` ignores existing cache and re-geocodes everything incomplete (spec §5.4). Live client uses `requests.get` with the crawler-style `DEFAULT_USER_AGENT`-suffixed UA and its own 1 req/s `time.sleep`.

- [ ] **Step 4: Run the fetcher suite**

Run: `pytest tests/test_data_fetcher.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tn_dollar_general/data_fetcher.py tests/test_data_fetcher.py
git commit -m "feat: reverse-geocode missing store addresses with 1rps Nominatim client and id-keyed cache"
```

---

### Task 3: Hex lattice, planar ranking, geodesic display pass (`hexgrid.py`)

**Files:**
- Create: `tn_dollar_general/hexgrid.py`
- Create: `tests/test_hexgrid.py`

**Interfaces:**
- Consumes: `StoreRecord` (existing), `geometry.GEOD`/projection conventions.
- Produces:
```python
@dataclass(frozen=True)
class HexRanking:
    centers_latlon: np.ndarray      # (N, 2) float64, lat/lon
    centers_m: np.ndarray           # (N, 2) float64, EPSG:6575 x/y
    distance_meters: np.ndarray     # (N,) float64 planar distance to nearest store
    geodesic_miles: np.ndarray      # (N,) float64 exact WGS84 miles, hex center -> nearest store
    rank: np.ndarray                # (N,) int32, 1-based; rank[i] = rank of hex i; rank 1 = max distance
    store_idx: np.ndarray           # (N,) int32 index into the *stores list passed in*
    hex_across_m: float
    ranks: int                      # N, number of ranked hexes
    def by_rank(self, k: int) -> int: ...       # hex index at rank k (k in 1..ranks)
    def neighbors_of_rank(self, k: int) -> tuple[int | None, int | None]: ...
    def top_k(self, k: int = 10) -> list[int]: ...

def build_hex_ranking(tn_geom: shapely.Geometry, stores: list[StoreRecord],
                      hex_across_m: float = 1000.0) -> HexRanking

def hex_vertices_m(centers: np.ndarray, hex_across_m: float) -> np.ndarray  # (N, 6, 2) x/y
```
Pinned geometry: flat-top hexes, circumradius `R = hex_across_m / sqrt(3)`; even-r layout: column pitch `dx = hex_across_m` (1000 m), row pitch `dy = 3R/2` (≈ 866.03 m), odd rows shifted `dx/2` right; vertices at angles 30°, 90°, …, 330° (so left/right sides are the flats). Grid origin = SW corner of the projected `tn_geom` bounds; only centers `prep(tn_geom).contains(center)` kept. Ranking: `np.lexsort((hex_id_array, -distance))` so ties break by hex id ascending.

- [ ] **Step 1: Write the failing tests** (synthetic square state: a `shapely` box ~20 km wide built *around* a real TN-region lon/lat so EPSG:6575 coordinates come out sane; stores at the 4 corners + 4 edge midpoints)

```python
def _square_state_and_corner_stores() -> tuple[shapely.Geometry, list[StoreRecord]]: ...

def test_rank_one_is_the_center_of_four_corner_stores():
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
    midpoint_xy = ...  # the square's midpoint, projected once with the same EPSG:6575 transformer
    center_hex = int(np.argmin(np.linalg.norm(ranking.centers_m - midpoint_xy, axis=1)))
    assert ranking.rank[center_hex] == 1

def test_rank_descends_monotonically_toward_a_store():
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
    # take the hex indices whose centers lie within 100 m of the straight segment
    # from the square midpoint to one corner store; order them by distance along
    # the segment: ranking.distance_meters must be strictly decreasing in that order.

def test_neighbors_of_rank_contract():
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
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
    radius = np.linalg.norm(verts - ranking.centers_m[:, None, :], axis=-1)
    assert np.allclose(radius.max(axis=-1), 577.350, atol=0.1)   # circumradius R = w/sqrt(3)
    span_x = verts[..., 0].max(axis=-1) - verts[..., 0].min(axis=-1)
    assert np.allclose(span_x, 1000.0, atol=0.1)                 # across-flats span = w
```

> Contract pinned: `neighbors_of_rank(k)` returns the tuple `(hex at rank k-1, hex at rank k+1)`, each slot `None` at the ends (rank 1 has no *upper* neighbor — it is the most remote — and rank N has no *lower* one).

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_hexgrid.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Implement** — vectorized: one `pyproj.Transformer` for centers + store coords, `scipy.spatial.cKDTree(stores_proj).query(centers)` for the planar pass, `geometry.GEOD.inv(lons1, lats1, lons2, lats2)` vectorized for the display pass (`stores` index from the same KDTree).

- [ ] **Step 4: Run to verify they pass; then run the whole suite**

Run: `pytest tests/test_hexgrid.py -v && pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tn_dollar_general/hexgrid.py tests/test_hexgrid.py
git commit -m "feat: hex lattice, planar remoteness ranking, and geodesic display distances"
```

---

### Task 4: Click-lookup bucket grid + encoded data block

**Files:**
- Modify: `tn_dollar_general/hexgrid.py` (append)
- Modify: `tests/test_hexgrid.py`

**Interfaces:**
- Consumes: `HexRanking` (Task 3).
- Produces:
```python
@dataclass(frozen=True)
class ClickTable:
    cell_deg: float                 # 0.005 (pinned default)
    lat_min: float; lon_min: float; lat_cells: int; lon_cells: int
    idx: np.ndarray                 # uint32 (lat_cells*lon_cells,), hex index or 0xFFFFFFFF sentinel

def build_click_table(hex_ranking: HexRanking, cell_deg: float = 0.005) -> ClickTable

def encode_data_block(hex_ranking: HexRanking, click_table: ClickTable,
                      store_display: dict[str, str], stores: list[StoreRecord]) -> dict
```
Data block layout (pinned, all arrays little-endian base64): `{"format": 1, "totals": {"hexes": N, "stores": S}, "clip": {"lat": [min, max], "lon": [min, max], "proj": "EPSG:6575"}, "dist_mi_q": uint16 = round(geodesic_miles * 1000), "rank_order": uint32 = hex index at rank 1..N, "store_idx": uint32, "centers": uint16 pairs quantized over clip lat/lon ranges, "click": {"cell_deg", "lat_min", "lon_min", "lat_cells", "lon_cells", "idx": uint32 base64}, "top10": [hex index, …] (len 10, padded to N when N < 10), "stores": [{id, name, address} …]}` — `store_display` comes from Task 2's `enrich_store_addresses` plus a Python-side fallback string for store-less/empty entries.

- [ ] **Step 1: Write the failing tests**

```python
def test_click_table_snaps_inside_and_sentinel_offstate():
    geom, stores = _square_state_and_corner_stores()
    ranking = build_hex_ranking(geom, stores)
    table = build_click_table(ranking)
    click in the square's interior, offset half a cell from the center:
        resolve via cell math (no helper function needed in test) ->
        the resolved hex center lies within 0.05 deg of the smallest-distance hex
        near that point.
    click 2 degrees beyond the clip bounds -> idx resolves to 0xFFFFFFFF sentinel.

def test_click_table_self_consistency_for_all_hex_centers():
    for at least 500 sampled hexes: cell lookup of a center's own coordinates
    returns a hex whose center is within `3 * cell_deg` degrees of the sample.

def test_data_block_roundtrip_and_top10_alignment():
    block = encode_data_block(ranking, table, store_display={}, stores=stores)
    loads json, np.frombuffer on each base64 field with dtype and shapes:
    dist_mi_q.shape == (N,) == rank_order.shape; top10 length 10;
    dist_mi_q[block top10[0]] == round(max(geodesic_miles)*1000).
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_hexgrid.py -k "click_table or data_block" -v`
Expected: FAIL

- [ ] **Step 3: Implement** — bucket each hex center into its cell; for empty cells carry the sentinel and leave resolution to the JS "nearest at cell-fill" pass folded into the same uint32 array: every cell's value = geometrically nearest hex center **including** over the boundary rows; build from a KDTree over all hex centers in lat/lon (degrees, small-angle approximation fine at ~9 km cells; assert max angular displacement in the self-consistency test, not geodesic miles).

- [ ] **Step 4: Run to verify they pass; then full suite**

Run: `pytest tests/test_hexgrid.py -v && pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tn_dollar_general/hexgrid.py tests/test_hexgrid.py
git commit -m "feat: click lookup bucket table and encoded hex data block"
```

---

### Task 5: Raster overlay PNG (single image)

**Files:**
- Modify: `tn_dollar_general/hexgrid.py` (append renderer)
- Modify: `pyproject.toml` (add `matplotlib>=3.8`; use `Agg` backend via `matplotlib.use("Agg", force=True)` at module import)
- Modify: `tests/test_hexgrid.py`

**Interfaces:**
- Consumes: `hex_vertices_m`, `HexRanking` (Task 3).
- Produces:
```python
def render_remoteness_png(hex_ranking: HexRanking, output_path: Path | str,
                          px_per_hex: int = 6) -> Path
def _ramp_color(distance_norm: np.ndarray) -> np.ndarray  # (..., 4) float RGBA, internal
```
Color ramp pinned: `distance_norm = distance / max(distance)`; `0.0 → "#E8F4FF"` (close to a DG, lightest), `1.0 → "#0F1F45"` (rank 1, deepest navy) — plain componentwise linear interpolation of the two endpoint RGB triples `(0xE8, 0xF4, 0xFF) → (0x0F, 0x1F, 0x45)`, no intermediate palette stops. Alpha = 1.0 inside hexes, 0 outside (figure background transparent). Filename written to the given path, pixel size = `px_per_hex` km-to-pixels scaled to the projected bounds extent (builder computes dpi; PNG bound: ≤ ~6 MB).

- [ ] **Step 1: Write the failing tests**

```python
def test_ramp_is_monotone_darkening():
    rgb = _ramp_color(np.array([0.0, 0.5, 1.0]))
    assert rgb[0, :3].sum() > rgb[1, :3].sum() > rgb[2, :3].sum()

def test_render_remoteness_png_bounds_and_size(tmp_path):
    geom, stores = _square_state_and_corner_stores()
    png = render_remoteness_png(build_hex_ranking(geom, stores), tmp_path / "r.png")
    data = png.read_bytes()
    w, h = struct.unpack(">II", data[16:24])      # PNG IHDR
    assert w > 400 and len(data) < 6 * 1024 * 1024
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_hexgrid.py -k "ramp or png" -v`
Expected: FAIL

- [ ] **Step 3: Implement** — `matplotlib.figure` sized from bounds/px_per_hex, one `PolyCollection` of `hex_vertices_m` with facecolors from `_ramp_color`, `Axes` with no ticks, `savefig(..., transparent=True)`.

- [ ] **Step 4: Run to verify they pass; then full suite**

Run: `pytest tests/test_hexgrid.py -v && pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tn_dollar_general/hexgrid.py tests/test_hexgrid.py pyproject.toml
git commit -m "feat: single-image remoteness raster PNG generator with blue ramp"
```

---

### Task 6: Side rail UI, raster overlay wiring, click handling (`map_builder.py`)

**Files:**
- Modify: `tn_dollar_general/map_builder.py` (extend `build_map` and add helpers)
- Modify: `tests/test_map_builder.py`

**Interfaces:**
- Consumes: `HexRanking`, `build_click_table`, `encode_data_block`, `render_remoteness_png` (Tasks 3–5), `StoreRecord` enrichment fields (Tasks 1–2).
- Produces: new signature
```python
def build_map(tn_geom, stores, result, output_path,
              hex_ranking: HexRanking | None = None,
              raster_png: Path | str | None = None,
              store_display: dict[str, str] | None = None) -> Path
```
When `hex_ranking is None`, behavior is byte-for-byte today's map (old tests keep passing untouched). Helper the plan pins:
```python
def _store_display_string(store: StoreRecord, address: str | None) -> str
```
Composition order: from `address` (Task 2 value) when not None/empty; else from `housenumber+street, city, state`; else the plain lat/lon coordinate pair. The rail: fixed right column, ~340 px reserved *beside* the map (map container gets a builder-chosen margin — outcome pinned: no map area is hidden under the rail; builder picks the exact CSS selector for folium's map container). Tabs pinned copy: `"Remoteness Podium"` and `"Top 10 Most Remote"`. Card per §2: rank (`#k of N hexes`), miles with 2 decimals, DG display string prefixed `"DG — "`. On map click the rail re-renders for the resolved hexagon; clicks resolving to the sentinel do nothing. The single click-highlight is a dashed `L.circle`-style pulse at the hex center (built as one `folium.PolyLine`/JS circle — one shape, replaced per click). Top 10 entries fly to their quantized center, then pulse the same highlight.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_map_without_ranking_is_unchanged(tmp_path):
    # no new test body: build_map called without the new kwargs must leave the
    # existing map tests passing untouched — Task gate is "pytest tests/test_map_builder.py" green.

def test_build_map_embeds_data_block_and_rail(tmp_path, tiny_hex_ranking):
    html_out = build_map(geom, stores, result, path, hex_ranking=tiny_hex_ranking,
                         raster_png=tmp_path / "r.png", store_display={}).read_text()
    assert html_out.count('id="dg-hex-data"') == 1
    assert "Remoteness Podium" in html_out and "Top 10 Most Remote" in html_out
    assert html_out.count("ImageOverlay") == 1      # exactly one raster layer

def test_no_large_hex_vector_layers_in_output(tmp_path, tiny_hex_ranking):
    plain_html = build_map(geom, stores, result, p1).read_text()
    ranked_html = build_map(geom, stores, result, p2, hex_ranking=tiny_hex_ranking,
                            raster_png=tmp_path / "r.png", store_display={}).read_text()
    for needle in ("L.geoJSON", "L.geoJson"):
        assert ranked_html.count(needle) == plain_html.count(needle)

def test_card_display_without_any_address_fields(tmp_path, tiny_hex_ranking):
    addressless = dataclasses.replace(stores[0], housenumber=None, street=None, city=None)
    html_out = build_map(geom, [addressless, *stores[1:]], result, path,
                         hex_ranking=tiny_hex_ranking, raster_png=tmp_path / "r.png",
                         store_display={}).read_text()
    # Python bakes no cards; the JSON payload must carry the coordinate fallback,
    # and the JS template carries the pinned copy verbatim:
    assert f"{addressless.lat:.4f}" in html_out
    assert "no address on record" in html_out

def test_store_display_is_escaped_for_rail_json(tmp_path, tiny_hex_ranking):
    hostile = dataclasses.replace(stores[0], name="<script>alert(1)</script>")
    html_out = build_map(geom, [hostile, *stores[1:]], result, path,
                         hex_ranking=tiny_hex_ranking, raster_png=tmp_path / "r.png",
                         store_display={}).read_text()
    assert "<script>alert(1)</script>" not in html_out
```
(`tiny_hex_ranking` = a fixture built on the square fixture with ~50 hexes — builder wires it once in `tests/test_map_builder.py`; `stores`/`geom`/`result` fixtures already exist there.)

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_map_builder.py -k "data_block or rail or escaped or address_fields" -v`
Expected: FAIL

- [ ] **Step 3: Implement** — read the raster PNG → base64 data URI for `folium.raster_layers.ImageOverlay(bounds=[…clip bounds of Task 3 clip…], interactive=False)`; serialize `encode_data_block(...)` output into `<script type="application/json" id="dg-hex-data">…</script>`; rail skeletons as `branca.element.Element` on `m.get_root()`; the JS click→rank→cards resolver consumes the block. All dynamic strings inserted into the JSON payload pass through `_store_display_string` + escaping (mirror `_safe_tooltip`'s `back gatherings`-style `html.escape` + backtick/`$` neutralization for anything text).

- [ ] **Step 4: Run to verify they pass; then full suite including untouched old tests**

Run: `pytest tests/test_map_builder.py -v && pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tn_dollar_general/map_builder.py tests/test_map_builder.py
git commit -m "feat: side rail podium tabs, top 10 jump, click-driven rank cards, single raster overlay"
```

---

### Task 7: Pipeline wiring in `main.py` + docs

**Files:**
- Modify: `main.py` (`run_pipeline` after the existing solver step, before `build_map`)
- Modify: `tests/test_map_builder.py` (extend the existing end-to-end test)
- Modify: `README.md` (extend "How it works" with the new hex/rail step)

**Interfaces:**
- Consumes: `build_hex_ranking`, `enrich_store_addresses`, `render_remoteness_png`, new `build_map` kwargs.
- Produces: pipeline behavior only — CLI surface unchanged (`--refresh`, `--data-dir`, `--output-dir` untouched).

- [ ] **Step 1: Extend the existing end-to-end test** (`test_run_pipeline_creates_outputs_and_summary_json`) — after the existing assertions, add:

```python
# extend test_run_pipeline_creates_outputs_and_summary_json after the existing asserts:
assert (output_dir / "remoteness.png").exists()
html_out = map_output_path.read_text()
assert 'id="dg-hex-data"' in html_out
assert "Top 10 Most Remote" in html_out
# summary.json's top-level contract stays exactly as it is today:
assert set(summary_data) == {
    "state", "total_tn_stores", "total_border_buffer_stores",
    "total_stores_analyzed", "coordinates", "distance_miles",
    "distance_km", "distance_meters", "nearest_stores",
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests -k run_pipeline -v`
Expected: FAIL (remoteness.png missing / data block absent)

- [ ] **Step 3: Implement** in `run_pipeline` between step 3 (solver) and step 4 (map):
```python
hex_ranking = build_hex_ranking(tn_geom=tn_geom, stores=stores)
store_display = enrich_store_addresses(stores, data_dir / "store_addresses.json", refresh=refresh)
raster_png = render_remoteness_png(hex_ranking, output_dir / "remoteness.png")
build_map(..., hex_ranking=hex_ranking, raster_png=raster_png, store_display=store_display)
```
Console summary unchanged; README's "How it works" gains one bullet naming the three new steps and the rail/tags behavior in a sentence (point to the spec path for details).

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_map_builder.py README.md
git commit -m "feat: wire hex ranking, address enrichment, and remoteness raster into the pipeline"
```
