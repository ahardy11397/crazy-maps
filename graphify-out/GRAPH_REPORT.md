# Graph Report - crazy-maps  (2026-11-02)

## Corpus Check
- 11 files · ~29,646 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 312 nodes · 775 edges · 8 communities (6 shown, 2 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 33 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Hex Grid & Remoteness Ranking
- Data Fetcher & Address Enrichment
- Map Builder & Side Rail UI
- Documentation, Specs & Plans
- Pipeline & Map Helpers
- Geometry Solver & Verification
- Test Fixtures (Nominatim Mocks)
- Package Root

## God Nodes (most connected - your core abstractions)
1. `StoreRecord` - 33 edges
2. `build_map()` - 32 edges
3. `build_hex_ranking()` - 28 edges
4. `find_furthest_point()` - 24 edges
5. `enrich_store_addresses()` - 21 edges
6. `parse_and_deduplicate_overpass()` - 19 edges
7. `fetch_dollar_generals()` - 19 edges
8. `build_click_table()` - 19 edges
9. `run_pipeline()` - 17 edges
10. `_square_state_and_corner_stores()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Spec §4.3 End-to-end pipeline & dense grid verification` --references--> `run_pipeline()`  [EXTRACTED]
  docs/superpowers/specs/2026-10-29-tn-dollar-general-map-design.md → main.py
- `Spec §1 Overview & Goals` --semantically_similar_to--> `crazy-maps project`  [INFERRED] [semantically similar]
  docs/superpowers/specs/2026-10-29-tn-dollar-general-map-design.md → README.md
- `Spec §3.2 Candidate generation (Voronoi + boundary + grid)` --rationale_for--> `find_furthest_point()`  [EXTRACTED]
  docs/superpowers/specs/2026-10-29-tn-dollar-general-map-design.md → tn_dollar_general/geometry.py
- `Spec §3.2 Geodesic refinement (Nelder-Mead)` --rationale_for--> `find_furthest_point()`  [EXTRACTED]
  docs/superpowers/specs/2026-10-29-tn-dollar-general-map-design.md → tn_dollar_general/geometry.py
- `Pole of Inaccessibility (Dollar General)` --conceptually_related_to--> `find_furthest_point()`  [EXTRACTED]
  README.md → tn_dollar_general/geometry.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Furthest-point solver geometry stack** — tn_dollar_general_geometry_find_furthest_point, docs_superpowers_specs_2026_10_29_tn_dollar_general_map_design_candidate_generation, docs_superpowers_specs_2026_10_29_tn_dollar_general_map_design_geodesic_refinement, docs_superpowers_plans_2026_10_29_tn_dollar_general_map_epsg_6576_nad83_tennessee, docs_superpowers_plans_2026_10_29_tn_dollar_general_map_wgs84_geodesic_refinement [INFERRED 0.85]
- **Overpass store ingestion & deduplication** — tn_dollar_general_data_fetcher_fetch_dollar_generals, tn_dollar_general_data_fetcher_parse_and_deduplicate_overpass, tn_dollar_general_data_fetcher_storerecord, docs_superpowers_specs_2026_10_29_tn_dollar_general_map_design_overpass_endpoints, docs_superpowers_plans_2026_10_29_tn_dollar_general_map_overpass_fallback_concern, docs_superpowers_specs_2026_10_29_tn_dollar_general_map_design_store_filtering_deduplication [INFERRED 0.85]
- **End-to-end TN Dollar General pipeline** — main_run_pipeline, tn_dollar_general_data_fetcher_fetch_tn_boundary, tn_dollar_general_data_fetcher_fetch_dollar_generals, tn_dollar_general_data_fetcher_parse_and_deduplicate_overpass, tn_dollar_general_geometry_find_furthest_point, tn_dollar_general_map_builder_build_map [INFERRED 0.95]
- **Hex Remoteness Ranking 7-Task Implementation Stack** — plan_task1_store_record, plan_task2_enrichment, plan_task3_hexgrid, plan_task4_click_table, plan_task5_raster, plan_task6_side_rail, plan_task7_pipeline [INFERRED 0.90]
- **User-Decided Design Constraints** — spec_decisions, spec_goals, spec_non_goals, spec_anti_goals, spec_constraints [INFERRED 0.90]
- **New/Modified Architecture Components** — spec_architecture, spec_hexgrid, spec_enrichment, spec_map_layer, spec_click_table, spec_data_block, spec_address_cache [INFERRED 0.90]

## Communities (8 total, 2 thin omitted)

### Community 0 - "Hex Grid & Remoteness Ranking"
Cohesion: 0.06
Nodes (34): _square_state_and_corner_stores(), test_click_table_self_consistency_for_all_hex_centers(), test_click_table_snaps_inside_and_sentinel_offstate(), test_click_table_state_mask_offstate_cells_at_tn_scale(), test_click_table_state_mask_sends_offstate_bbox_cells_to_sentinel(), test_data_block_roundtrip_and_top10_alignment(), test_equal_distance_ties_break_by_ascending_hex_id_and_store_order(), test_hex_vertices_across_flats_and_radius() (+26 more)

### Community 1 - "Data Fetcher & Address Enrichment"
Cohesion: 0.06
Nodes (34): _AlwaysFailingClient, _FakeNominatimClient, square_geom_fixture(), _store(), test_cache_roundtrip_persists_housenumber_and_postcode(), test_dedup_merge_fills_housenumber_from_colocated_duplicate(), test_dedup_radius_is_50_meters_not_feet(), test_enrich_breaker_stops_negatively_but_keeps_successes() (+26 more)

### Community 2 - "Map Builder & Side Rail UI"
Cohesion: 0.07
Nodes (28): test_store_address_addressless_in_tn_uses_coordinate_fallback(), _decode_block(), _extract_block(), _extract_rail_script(), geom(), result(), _run_rail_harness(), stores() (+20 more)

### Community 3 - "Documentation, Specs & Plans"
Cohesion: 0.06
Nodes (38): Architecture: build-time Python (KDTree, pyproj, matplotlib, folium ImageOverlay, Nominatim), Constraints: Rank 1 = most remote, flat-top 1km hex, EPSG:6575, WGS84 geodesic display, self-contained HTML, one raster PNG ≤6MB, Goal: Rank every 1 km hex in TN by distance to nearest DG, click → side rail with Podium + Top 10, Hex Remoteness Ranking & Side-Rail Explorer Implementation Plan, Review focus: no-address fallback, Nominatim graceful failure, cache survives --refresh, border clicks snap, hostile names escaped, Task 1: Capture housenumber & postcode in StoreRecord, Task 2: Address enrichment via Nominatim + id-keyed cache, Task 3: Hex lattice, planar ranking, geodesic display (hexgrid.py) (+30 more)

### Community 4 - "Pipeline & Map Helpers"
Cohesion: 0.10
Nodes (12): TN Dollar General Map Implementation Plan, TN Dollar General Map Design Spec, main(), run_pipeline(), _extract_display_address(), _join_address_parts(), _bearing_to_compass(), _build_furthest_popup() (+4 more)

### Community 5 - "Geometry Solver & Verification"
Cohesion: 0.13
Nodes (16): EPSG:6576 NAD83(2011)/Tennessee planar projection, WGS84 ellipsoid geodesic refinement, _make_store(), test_boundary_constrained_with_infinite_voronoi_ridges(), test_default_grid_step_is_1km_verification_grid(), test_empty_stores_or_geom_raises_value_error(), test_interior_voronoi_vertex_exact_center(), test_multipolygon_and_geodesic_circle() (+8 more)

## Knowledge Gaps
- **15 isolated node(s):** `crazy-maps`, `Boundary fetch (US Census 1:500k state polygon)`, `Stores fetch (Overpass API with fallback, deduplication)`, `Install via uv sync`, `Usage: uv run python main.py [--refresh]` (+10 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 109 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Pole of Inaccessibility (Dollar General)` connect `Documentation, Specs & Plans` to `Map Builder & Side Rail UI`, `Geometry Solver & Verification`?**
  _High betweenness centrality (0.183) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `StoreRecord` (e.g. with `stores()` and `find_furthest_point()`) actually correct?**
  _`StoreRecord` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `crazy-maps`, `Boundary fetch (US Census 1:500k state polygon)`, `Stores fetch (Overpass API with fallback, deduplication)` to the rest of the system?**
  _15 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Hex Grid & Remoteness Ranking` be split into smaller, more focused modules?**
  _Cohesion score 0.06493506493506493 - nodes in this community are weakly interconnected._
- **Are the 2 inferred relationships involving `build_map()` (e.g. with `StoreRecord` and `HexRanking`) actually correct?**
  _`build_map()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `Data Fetcher & Address Enrichment` be split into smaller, more focused modules?**
  _Cohesion score 0.05961538461538462 - nodes in this community are weakly interconnected._
- **Are the 3 inferred relationships involving `find_furthest_point()` (e.g. with `test_default_grid_step_is_1km_verification_grid()` and `StoreRecord`) actually correct?**
  _`find_furthest_point()` has 3 INFERRED edges - model-reasoned connections that need verification._