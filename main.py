from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from tn_dollar_general.data_fetcher import fetch_tn_boundary, fetch_dollar_generals
from tn_dollar_general.geometry import FurthestPointResult, find_furthest_point
from tn_dollar_general.map_builder import build_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def run_pipeline(
    data_dir: Path | str = Path("data"),
    output_dir: Path | str = Path("output"),
    refresh: bool = False,
) -> FurthestPointResult:
    """Run the complete end-to-end pipeline:

    1. Fetch / load Tennessee state boundary.
    2. Fetch / load Dollar General stores (Tennessee + 50km buffer).
    3. Compute the exact furthest point inside Tennessee using Voronoi + geodesic optimization.
    4. Render the interactive Folium HTML map.
    5. Write the summary JSON file.
    6. Print a formatted summary report.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch boundary
    boundary_cache = data_dir / "tn_boundary.geojson"
    logger.info("Step 1/5: Loading Tennessee state boundary (%s)...", boundary_cache)
    tn_geom = fetch_tn_boundary(cache_path=boundary_cache, refresh=refresh)

    # 2. Fetch Dollar General locations
    stores_cache = data_dir / "dollar_generals.geojson"
    logger.info("Step 2/5: Loading Dollar General store locations (%s)...", stores_cache)
    stores = fetch_dollar_generals(
        tn_geom=tn_geom,
        cache_path=stores_cache,
        refresh=refresh,
    )
    total_tn = sum(1 for s in stores if s.in_tn)
    total_border = sum(1 for s in stores if not s.in_tn)
    logger.info("Loaded %d Dollar Generals (%d in TN, %d in border buffer)", len(stores), total_tn, total_border)

    # 3. Compute furthest point
    logger.info("Step 3/5: Computing exact furthest point via Voronoi and WGS84 geodesic optimization...")
    result = find_furthest_point(tn_geom=tn_geom, stores=stores)

    # 4. Generate interactive map
    map_output_path = output_dir / "tn_dollar_general_map.html"
    logger.info("Step 4/5: Rendering interactive Folium map to %s...", map_output_path)
    build_map(tn_geom=tn_geom, stores=stores, result=result, output_path=map_output_path)

    # 5. Write summary JSON
    summary_path = output_dir / "summary.json"
    logger.info("Step 5/5: Writing summary JSON to %s...", summary_path)
    summary_data = {
        "state": "Tennessee",
        "total_tn_stores": total_tn,
        "total_border_buffer_stores": total_border,
        "total_stores_analyzed": len(stores),
        "coordinates": {
            "lat": result.lat,
            "lon": result.lon,
        },
        "distance_miles": result.distance_miles,
        "distance_km": result.distance_km,
        "distance_meters": result.distance_meters,
        "nearest_stores": [
            {
                "id": ns.store.id,
                "name": ns.store.name,
                "lat": ns.store.lat,
                "lon": ns.store.lon,
                "city": ns.store.city,
                "street": ns.store.street,
                "state": ns.store.state,
                "in_tn": ns.store.in_tn,
                "distance_meters": ns.distance_meters,
                "distance_miles": ns.distance_miles,
                "distance_km": ns.distance_km,
                "bearing_degrees": ns.bearing_degrees,
            }
            for ns in result.nearest_stores
        ],
    }
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # Console Summary Report
    separator = "=" * 78
    print(f"\n{separator}")
    print(" TENNESSEE DOLLAR GENERAL POLE OF INACCESSIBILITY")
    print(separator)
    print(f" Total Dollar Generals in Tennessee: {total_tn:,}")
    print(f" Out-of-State Buffer Stores Analyzed: {total_border:,}")
    print(f" Total Stores Analyzed:              {len(stores):,}")
    print("\n Furthest Point in Tennessee:")
    print(f"   Latitude:   {result.lat:.6f}° N")
    print(f"   Longitude:  {abs(result.lon):.6f}° W  ({result.lon:.6f})")
    print("\n Distance to Nearest Dollar General:")
    print(f"   Miles:      {result.distance_miles:.2f} miles")
    print(f"   Kilometers: {result.distance_km:.2f} km")
    print(f"   Meters:     {result.distance_meters:,.1f} meters")
    print("\n Nearest Constraining Stores:")
    for idx, ns in enumerate(result.nearest_stores[:5], 1):
        s = ns.store
        loc = f"{s.street or ''}, {s.city or ''}, {s.state or ''}".strip(", ")
        print(f"   {idx}. {s.name} ({loc})")
        print(f"      Distance: {ns.distance_miles:.2f} mi ({ns.distance_km:.2f} km) | Bearing: {ns.bearing_degrees:.1f}°")
    print(f"\n Artifacts Generated:")
    print(f"   - Interactive Map: {map_output_path.resolve()}")
    print(f"   - Summary JSON:    {summary_path.resolve()}")
    print(f"{separator}\n")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the point in Tennessee furthest from any Dollar General store and generate an interactive map."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh data from Census and Overpass APIs even if cached.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to cache downloaded GeoJSON and Overpass data (default: data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to write HTML map and summary JSON (default: output).",
    )
    args = parser.parse_args()

    try:
        run_pipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            refresh=args.refresh,
        )
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
