# Graph Report - crazy-maps  (2026-10-31)

## Corpus Check
- Corpus is ~9,791 words - fits in a single context window. You may not need a graph.

## Summary
- 127 nodes · 313 edges · 10 communities (8 shown, 2 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Furthest-Point Geometry Solver
- Folium Map Rendering
- Core Modules & Dependencies
- Pipeline CLI & Projections
- Overpass Parsing & Dedup
- Store Fetching & Caching
- Project Overview & Integration Tests
- Boundary Fetch & Parsing
- Cache Fallback Tests
- Repository Root

## God Nodes (most connected - your core abstractions)
1. `find_furthest_point()` - 31 edges
2. `build_map()` - 21 edges
3. `StoreRecord` - 19 edges
4. `fetch_dollar_generals()` - 19 edges
5. `parse_and_deduplicate_overpass()` - 17 edges
6. `run_pipeline()` - 16 edges
7. `fetch_tn_boundary()` - 13 edges
8. `FurthestPointResult` - 13 edges
9. `verify_furthest_point()` - 10 edges
10. `_make_store()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `How it works: Solver step` --references--> `find_furthest_point()`  [INFERRED]
  README.md → tn_dollar_general/geometry.py
- `How it works: Solver step` --semantically_similar_to--> `EPSG:6576 NAD83(2011)/Tennessee planar projection`  [AMBIGUOUS] [semantically similar]
  README.md → docs/superpowers/plans/2026-10-29-tn-dollar-general-map.md
- `TN Dollar General Map Implementation Plan` --references--> `run_pipeline()`  [EXTRACTED]
  docs/superpowers/plans/2026-10-29-tn-dollar-general-map.md → main.py
- `TN Dollar General Map Design Spec` --references--> `run_pipeline()`  [EXTRACTED]
  docs/superpowers/specs/2026-10-29-tn-dollar-general-map-design.md → main.py
- `How it works: Boundary fetch step` --references--> `fetch_tn_boundary()`  [INFERRED]
  README.md → tn_dollar_general/data_fetcher.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **End-to-end TN Dollar General pipeline** — main_run_pipeline, tn_dollar_general_data_fetcher_fetch_tn_boundary, tn_dollar_general_data_fetcher_fetch_dollar_generals, tn_dollar_general_data_fetcher_parse_and_deduplicate_overpass, tn_dollar_general_geometry_find_furthest_point, tn_dollar_general_map_builder_build_map [INFERRED 0.95]
- **Furthest-point solver geometry stack** — tn_dollar_general_geometry_find_furthest_point, docs_superpowers_specs_2026_10_29_tn_dollar_general_map_design_candidate_generation, docs_superpowers_specs_2026_10_29_tn_dollar_general_map_design_geodesic_refinement, docs_superpowers_plans_2026_10_29_tn_dollar_general_map_epsg_6576_nad83_tennessee, docs_superpowers_plans_2026_10_29_tn_dollar_general_map_wgs84_geodesic_refinement [INFERRED 0.85]
- **Overpass store ingestion & deduplication** — tn_dollar_general_data_fetcher_fetch_dollar_generals, tn_dollar_general_data_fetcher_parse_and_deduplicate_overpass, tn_dollar_general_data_fetcher_storerecord, docs_superpowers_specs_2026_10_29_tn_dollar_general_map_design_overpass_endpoints, docs_superpowers_plans_2026_10_29_tn_dollar_general_map_overpass_fallback_concern, docs_superpowers_specs_2026_10_29_tn_dollar_general_map_design_store_filtering_deduplication [INFERRED 0.85]

## Communities (10 total, 2 thin omitted)

### Community 0 - "Furthest-Point Geometry Solver"
Cohesion: 0.16
Nodes (15): WGS84 ellipsoid geodesic refinement, _make_store(), test_boundary_constrained_with_infinite_voronoi_ridges(), test_default_grid_step_is_1km_verification_grid(), test_empty_stores_or_geom_raises_value_error(), test_interior_voronoi_vertex_exact_center(), test_multipolygon_and_geodesic_circle(), test_out_of_state_border_store_constrains_result() (+7 more)

### Community 1 - "Folium Map Rendering"
Cohesion: 0.18
Nodes (8): output/summary.json, output/tn_dollar_general_map.html, _bearing_to_compass(), _build_furthest_popup(), build_map(), _build_store_popup(), _build_summary_card_html(), _safe_tooltip()

### Community 3 - "Pipeline CLI & Projections"
Cohesion: 0.17
Nodes (4): EPSG:6576 NAD83(2011)/Tennessee planar projection, main(), run_pipeline(), main.py CLI entry point

### Community 4 - "Overpass Parsing & Dedup"
Cohesion: 0.20
Nodes (7): test_dedup_radius_is_50_meters_not_feet(), test_parse_assigns_typed_osm_ids_for_links(), test_parse_overpass_empty_and_invalid(), test_parse_overpass_extracts_nodes_ways_and_filters_warehouses(), test_parse_overpass_relation_and_wikidata(), parse_and_deduplicate_overpass(), _tag_richness_score()

### Community 5 - "Store Fetching & Caching"
Cohesion: 0.20
Nodes (4): data/dollar_generals.geojson cache, OpenStreetMap data (ODbL), Overpass API, fetch_dollar_generals()

### Community 6 - "Project Overview & Integration Tests"
Cohesion: 0.33
Nodes (7): crazy-maps project, Tennessee Dollar General Pole of Inaccessibility, test_build_map_creates_self_contained_html_with_all_layers(), test_build_map_default_basemap_is_openstreetmap_without_keyed_tiles(), test_build_map_escapes_hostile_store_names_in_tooltips(), test_run_pipeline_creates_outputs_and_summary_json(), StoreRecord

### Community 7 - "Boundary Fetch & Parsing"
Cohesion: 0.25
Nodes (5): data/tn_boundary.geojson cache, U.S. Census Bureau cartographic boundary files, test_parse_boundary_handles_feature_collection_and_multipolygon(), fetch_tn_boundary(), parse_boundary_geojson()

### Community 8 - "Cache Fallback Tests"
Cohesion: 0.50
Nodes (3): test_fetch_dollar_generals_cache_hit(), test_fetch_dollar_generals_falls_back_Across_overpass_endpoints(), test_fetch_tn_boundary_caching_and_download()

## Ambiguous Edges - Review These
- `How it works: Solver step` → `EPSG:6576 NAD83(2011)/Tennessee planar projection`  [AMBIGUOUS]
  README.md · relation: semantically_similar_to

## Knowledge Gaps
- **8 isolated node(s):** `crazy-maps`, `main.py CLI entry point`, `output/summary.json`, `output/tn_dollar_general_map.html`, `data/tn_boundary.geojson cache` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 38 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `find_furthest_point()` connect `Furthest-Point Geometry Solver` to `Core Modules & Dependencies`, `Pipeline CLI & Projections`, `Project Overview & Integration Tests`?**
  _High betweenness centrality (0.193) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `find_furthest_point()` (e.g. with `How it works: Solver step` and `test_default_grid_step_is_1km_verification_grid()`) actually correct?**
  _`find_furthest_point()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `crazy-maps`, `main.py CLI entry point`, `output/summary.json` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Why does `fetch_dollar_generals()` connect `Store Fetching & Caching` to `Furthest-Point Geometry Solver`, `Core Modules & Dependencies`, `Pipeline CLI & Projections`, `Overpass Parsing & Dedup`, `Project Overview & Integration Tests`, `Boundary Fetch & Parsing`, `Cache Fallback Tests`?**
  _High betweenness centrality (0.169) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `StoreRecord` (e.g. with `find_furthest_point()` and `verify_furthest_point()`) actually correct?**
  _`StoreRecord` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Why does `StoreRecord` connect `Project Overview & Integration Tests` to `Furthest-Point Geometry Solver`, `Folium Map Rendering`, `Core Modules & Dependencies`, `Overpass Parsing & Dedup`, `Store Fetching & Caching`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `How it works: Solver step` and `EPSG:6576 NAD83(2011)/Tennessee planar projection`?**
  _Edge tagged AMBIGUOUS (relation: semantically_similar_to) - confidence is low._