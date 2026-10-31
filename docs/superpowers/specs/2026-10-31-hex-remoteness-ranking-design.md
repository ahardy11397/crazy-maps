# Hexagon Remoteness Ranking & Side-Rail Explorer — Design Spec

**Date:** 2026-10-31
**Project:** `crazy-maps`
**Status:** Draft for User Review
**Depends on:** `2026-10-29-tn-dollar-general-map-design.md` (existing pipeline kept intact)

---

## 1. Overview & Intent

The existing map answers one question: *which single point in Tennessee is farthest from any Dollar General?* This expansion turns the map into a **statewide remoteness ranking**:

1. Tile Tennessee with small hexagons; compute each hexagon's distance to the nearest Dollar General store (in-state and border-buffer stores alike).
2. Rank all hexagons state-wide. **Rank #1 = the most remote hexagon (farthest from any DG)**; store-topped hexagons rank last.
3. On the map, clicking any hexagon opens a **side rail** with three cards: the selected hexagon plus its two rank-neighbors (one spot better, one spot worse). Each card shows rank, distance in miles, and the nearest DG's street address.
4. A **Top 10 tab** in the side rail shows the ten most-remote hexagons as a browsable leaderboard, each linking to and highlighting its hexagon on the map.

The user's core need: follow the news about a town / be curious about a spot, click it, and learn "where does this neighborhood stand statewide — and who beats me by one spot?"

## 2. Decided-by-User Choices (do not re-litigate)

| Choice | Value |
|---|---|
| Hexagon size | **1 km across**, yielding roughly 42,000 hexes whose centers lie inside Tennessee |
| Rank direction | **#1 = most remote**; ranks descend toward the store-dense areas |
| Metric | Straight-line distance, in **miles**, from hexagon center to nearest DG center |
| Palette | **Single-hue blue scale** — pale blue where stores crowd in, deep navy in emptiest regions |
| Click UI | **Side rail** (fixed panel on the right), never covering the map |
| Cards | Exactly 3: selected hexagon (emphasized) + rank-1 and rank+1 neighbors |
| Card fields | Rank (out of ~42,000), distance in miles, nearest DG name + street address + city, state |
| Leaderboard | Yes — Top 10 tab, browsable, jump-to-highlight on the map |
| Store addresses | **Option 3 (enrich)**: capture house numbers where OSM has them AND reverse-geocode the gaps; addresses cached, one-time build cost ok (~9 min at 1 req/s) |
| Place search | **No** — map browsing + Top 10 tab is enough for now; a search box is deliberately out of scope |

## 3. Goals, Non-Goals, Anti-Goals

**Goals**
- Click anywhere → instant, accurate rank (`< 50 ms` perceived) with no lag from the graphic layer.
- The generated `output/tn_dollar_general_map.html` remains **self-contained**: works offline except for the existing tile-CDN/cached basemap behavior it already has.
- The existing features — store markers, state boundary, focal point + its popup, floating info card — remain present and unchanged in behavior.
- Repeatable, tested pipeline: same caches → same rank list.

**Non-goals (explicitly out of scope)**
- Driving distances / routing times. Metric stays straight-line.
- Per-hexagon store *count* density (a different choropleth).
- Mobile/responsive layout polish.
- Changing the solver, dedup rules, or data sources.
- A place/search box (decided 2026-10-31: browsing + Top 10 suffice; revisit later if wanted).

**Anti-goals (would make this a failure)**
- A map that stutters or takes > 2 s to open because of the 42k hexagons.
- Any per-hex graphics drawn as thousands of SVG/canvas shapes — the raster-overlay rule in §5 must hold.
- Ranks that disagree with the existing focal point (the #1-ranked hexagon must sit at/near `FurthestPointResult`, within grid-resolution tolerance).

## 4. Constraints

- Public data only, **no API keys** (Overpass, Census boundary). Reverse geocoding uses Nominatim (OpenStreetMap), which is free and key-less; usage policy caps polite clients at 1 req/s — hence one-time enrichment cached into `data/dollar_generals.geojson`.
- Build stack: keep the existing dependency set plus nothing heavier than matplotlib (for the overlay raster) — no server-side rendering service, no Node toolchain.
- The code already computes planar nearest-store distances against store `KDTree`s in **EPSG:6575** (NAD83(2011)/Tennessee). The hex grid shall compute its metric in the *same projection* on the *same prepared store set*, so a `verify_furthest_point`-style audit can compare ranks with the existing solver exactly.
- Store set includes border-buffer stores (`in_tn=False` still constraints) — a hexagon's nearest DG may be out-of-state.
- pytest suite must grow with the feature; existing tests must keep passing.

## 5. Architecture (new/changed pieces)

```
tn_dollar_general/
├── hexgrid.py          # NEW — hex grid, distances, ranks, raster, data block
├── data_fetcher.py     # EXT — housenumber/postcode capture + Nominatim enrichment
└── map_builder.py      # EXT — side rail UI, overlay layer, click wiring, leaderboard
tests/
└── test_hexgrid.py     # NEW
```

### 5.1 Hex grid & ranking (`hexgrid.py`)
- Anchored on the TN extent in EPSG:6575; flat-top hexagons with **1 km across-flats** (`spacing ≈ 1 km` center-to-center horizontally, ~1.155 km row pitch — builder picks exact axial scheme; not load-bearing).
- Keep only hexes whose **center** is inside `tn_geom` (shapely `contains`, prepared). Target order of magnitude: ~42,000.
- Distances: `cKDTree` on projected store coords, one `query(k=1)` over all centers (trivial compute).
- **Ranks:** sort by distance descending → rank 1..N. Equal distances broken deterministically (hex-index ascending). Nearest store id stored with each hex for card display.
- Exposed dataclass `HexRanking` carrying: arrays (center lat/lon, distance_meters, rank, store_idx), the axial-coordinate mapping (for click lookup), and Neighbor Table by rank (rank-1, rank+1).

### 5.2 Address enrichment (`data_fetcher.py`)
- Capture `addr:housenumber` and `addr:postcode` into `StoreRecord` alongside existing city/street/state; thread through the dedup-merge already present.
- Fallback tags for reverse geocoding coverage: only stores missing any of street/housenumber/city get Nominatim `reverse` lookups (lat/lon → display address), 1 req/s with a library-style helper (`time.sleep(1.05)`), results written back into `data/dollar_generals.geojson` cache so it happens once. Tolerate complete failure gracefully (address renders as `"coords — no address on record"`; never raise).
- All address fields flow to the front-end per store.

### 5.3 Map layer (`map_builder.py`)
- **Raster overlay:** build renders ONE PNG of the whole hex field (matplotlib `PolyCollection`/`pcolormesh`, ≤ ~2 MB), colors from the single-hue blue ramp (`0 mi → lightest, #1 → deepest navy`), projected bounds handed to `folium.raster_layers.ImageOverlay`. **Only one image, zero vector hexes in SVG/GeoJSON.**
- **Click math:** embedded compact data block — per hex: distance (uint16, 0.01 mi units), rank (delta-encoded uint16), nearest store index (uint16) ≈ 6 B/hex ≈ **~250 KB** — base64 in the HTML, decoded into typed arrays. Clicked `(lat, lon)` → EPSG:6575 → axial hex id → array lookup: O(1), pure math.
- **Side rail UI** (*owner call 2026-10-31*): the rail **reserves its own column beside the map** (viewport shrinks, map DOM resized in a flex wrapper) so the map is never *obscured* — it stays fully visible, and the existing floating info card keeps its home. Rail: (a) 3-card "remoteness podium" per §2, (b) **Top 10 tab** listing most-remote hexes with rank/distance/nearest DG and a `jump to map` action that pans/zooms and pulses a highlight overlay of that one hexagon (a vector heptagon for 10 hexes is fine).
- **Click highlight** (*owner call*): one ephemeral vector hexagon — outline drawn dynamically at click time on the clicked (or snapped) hex, removed on the next click. Baked anti-goal targets pre-rendered hexes only; this lives in a dedicated empty-at-build layer.
- **Border clicks** (*owner call*): a click inside TN with no ranked hex underneath (near-border gaps in the center-inside-TN set) **snaps to the nearest ranked hex**; clicks entirely off-state leave the rail untouched.
- **Displayed distance** (*owner call*): cards show the **exact WGS84 geodesic** miles — one build-time geodesic pass over each hex center → its nearest store (cheap, ~42k `pyproj.Geod` invocations) — so a card's miles match the existing info card to the decimal. **Ranking** uses the planar EPSG:6575 metric (§5.1); acceptance allows equality only to display precision (~0.05 mi tolerance in test §6.3).
- Card copy example: `#412 of ~42,000 · 15.4 mi · DG — 707 New Life Ln, Cedar Grove, TN`.
- The #1-ranked hexagon and the existing focal point must visibly coincide (grid tolerance), so the old and new stories agree.

### 5.4 Address cache durability (`data_fetcher.py`) (*owner call*)
- Geocoded addresses persist in a **separate cache** (`data/store_addresses.json`, keyed by stable OSM store id), written only on successful enrichment. Store re-fetch (`--refresh` from Overpass) does **not** discard them; after a refresh, enrichment runs only for store ids missing from the address cache (new stores only). A per-store failure is retried on the next build, not treated as fatal.

### 5.5 CLI (`main.py`)
- Unchanged interface; pipeline grows the hex step automatically after store fetch + solver, writing the PNG + embedding the data block in the same HTML build.

## 6. Verification & Testing Strategy

1. **test_hexgrid.py:** synthetic square state + 4 stores at corners: central hex = rank #1, ranks along a radial corner→center line decrease monotonically; equal-distance tie is stable; every rank-neighbor pairs are adjacent by ±1; hex math round-trip (`click→hex→center within half-across-flats`).
2. **Address enrichment:** mocked Nominatim transport with a fake turn-based sleep so tests run instantly; assert cache write-back, 1/s pacing, and graceful per-store failure fallback.
3. **Map/output:** assert exactly one raster overlay, no > 1k-element GeoJSON hex layer, side-rail markup present, Top-10 tab present, data block decodes and the top decoded distance matches the solver's `FurthestPointResult` ≤ grid-resolution tolerance.
4. **Manual acceptance:** open the file offline — map opens < 2 s, zoom/pan stays smooth with the overlay on, click near a known DG yields rank near the end of the list, click near TN's true pole-of-inaccessibility yields rank ≈ #1 with the same distance as the existing floating info card.

## 7. Open Items for the Builder

- Exact axial-coordinate scheme + hex-to-index encoding (only the O(1) requirement is hard).
- The precise raster pixel-to-hex alignment (edge picking between pixels) — reasonable defaults fine.
- UI details of the side rail (collapse button, small-screen behavior) — non-goal territory, keep simple.
