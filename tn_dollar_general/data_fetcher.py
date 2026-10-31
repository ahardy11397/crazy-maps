from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
import re
import time

import numpy as np
import pyproj
import requests
import scipy.spatial
import shapely
from shapely.geometry import Point, shape, mapping
from shapely.prepared import prep

logger = logging.getLogger(__name__)

DEFAULT_BBOX = (34.5, -90.9, 37.2, -81.0)

DEFAULT_USER_AGENT = "CrazyMapsDollarGeneralFinder/1.0 (https://github.com/crazy-maps)"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

TN_BOUNDARY_URLS = [
    # Census 1:500k state polygons — high resolution (spec 3.1), tried first
    "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_500k.json",
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json",
    "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json",
]


@dataclass(frozen=True)
class StoreRecord:
    id: str
    name: str
    lat: float
    lon: float
    city: str | None
    street: str | None
    state: str | None
    in_tn: bool
    housenumber: str | None = None
    postcode: str | None = None


def parse_boundary_geojson(geojson_data: dict) -> shapely.Geometry:
    """Parse GeoJSON data (FeatureCollection, Feature, or Geometry) into a Shapely Geometry."""
    geom_type = geojson_data.get("type")
    if geom_type == "FeatureCollection":
        features = geojson_data.get("features", [])
        if not features:
            raise ValueError("FeatureCollection contains no features.")
        tn_feature = None
        for feat in features:
            props = feat.get("properties") or {}
            for key in ("name", "NAME", "State_Name", "STATE_NAME", "STUSPS"):
                val = str(props.get(key, "")).strip().lower()
                if val in ("tennessee", "tn"):
                    tn_feature = feat
                    break
            if tn_feature:
                break
        if tn_feature:
            geom = shape(tn_feature["geometry"])
        elif len(features) == 1:
            geom = shape(features[0]["geometry"])
        else:
            geom = shapely.unary_union([shape(f["geometry"]) for f in features if "geometry" in f])
    elif geom_type == "Feature":
        geom = shape(geojson_data["geometry"])
    elif geom_type in ("Polygon", "MultiPolygon", "GeometryCollection"):
        geom = shape(geojson_data)
    else:
        raise ValueError(f"Unsupported GeoJSON structure: {geom_type}")

    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    return geom


def fetch_tn_boundary(cache_path: Path | str, refresh: bool = False) -> shapely.Geometry:
    """Fetch Tennessee state boundary geometry, using cache if available."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return parse_boundary_geojson(data)

    last_error = None
    for url in TN_BOUNDARY_URLS:
        try:
            logger.info("Fetching Tennessee boundary from %s", url)
            resp = requests.get(url, headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            geom = parse_boundary_geojson(data)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            feature = {
                "type": "Feature",
                "properties": {"name": "Tennessee"},
                "geometry": mapping(geom),
            }
            cache_path.write_text(json.dumps(feature, indent=2), encoding="utf-8")
            return geom
        except Exception as exc:
            logger.warning("Failed to fetch Tennessee boundary from %s: %s", url, exc)
            last_error = exc

    raise RuntimeError(f"Failed to fetch Tennessee state boundary from all candidate URLs: {last_error}")


def _tag_richness_score(tags: dict) -> int:
    score = 0
    for key in ("addr:street", "addr:city", "addr:state", "addr:postcode", "addr:housenumber"):
        if tags.get(key):
            score += 10
    name = tags.get("name", "")
    if name and name != "Dollar General":
        score += 5
    score += len(tags)
    return score


def parse_and_deduplicate_overpass(
    elements: list[dict],
    tn_geom: shapely.Geometry,
    dedup_distance_m: float = 50.0,
) -> list[StoreRecord]:
    """Parse raw Overpass elements, filter out warehouses/distribution centers,

    deduplicate co-located points within dedup_distance_m using EPSG:6575 and KDTree,
    and tag whether each store is inside TN.
    """
    candidates = []
    prepared_tn = prep(tn_geom)

    for el in elements:
        el_type = el.get("type", "node")
        if el_type == "node":
            lat = el.get("lat")
            lon = el.get("lon")
        else:
            center = el.get("center") or {}
            lat = center.get("lat", el.get("lat"))
            lon = center.get("lon", el.get("lon"))

        if lat is None or lon is None:
            continue
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            continue

        tags = el.get("tags") or {}
        name = tags.get("name", "")
        brand = tags.get("brand", "")
        wikidata = tags.get("brand:wikidata", "")

        matches_name = bool(re.search(r"^Dollar General", name, re.IGNORECASE))
        matches_brand = (brand == "Dollar General")
        matches_wikidata = (wikidata == "Q145168")
        if not (matches_name or matches_brand or matches_wikidata):
            continue

        building = str(tags.get("building", "")).lower()
        if (
            building == "warehouse"
            or "industrial" in tags
            or "distribution center" in name.lower()
        ):
            continue

        store_name = name or brand or "Dollar General"
        city = tags.get("addr:city")
        street = tags.get("addr:street")
        state = tags.get("addr:state")
        housenumber = tags.get("addr:housenumber")
        postcode = tags.get("addr:postcode")

        score = _tag_richness_score(tags)
        candidates.append({
            "id": f"{el_type}/{el.get('id', '')}",
            "name": store_name,
            "lat": lat,
            "lon": lon,
            "city": city,
            "street": street,
            "state": state,
            "housenumber": housenumber,
            "postcode": postcode,
            "score": score,
        })

    if not candidates:
        return []

    if len(candidates) == 1:
        c = candidates[0]
        in_tn = bool(prepared_tn.intersects(Point(c["lon"], c["lat"])))
        state = c["state"] or ("TN" if in_tn else None)
        return [
            StoreRecord(
                id=c["id"],
                name=c["name"],
                lat=c["lat"],
                lon=c["lon"],
                city=c["city"],
                street=c["street"],
                state=state,
                in_tn=in_tn,
                housenumber=c["housenumber"],
                postcode=c["postcode"],
            )
        ]

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6575", always_xy=True)
    lons = [c["lon"] for c in candidates]
    lats = [c["lat"] for c in candidates]
    xs, ys = transformer.transform(lons, lats)
    coords = np.column_stack([xs, ys])

    tree = scipy.spatial.KDTree(coords)
    sorted_indices = sorted(range(len(candidates)), key=lambda i: candidates[i]["score"], reverse=True)

    suppressed = set()
    kept_indices = []

    for idx in sorted_indices:
        if idx in suppressed:
            continue
        kept_indices.append(idx)
        neighbors = tree.query_ball_point(coords[idx], r=dedup_distance_m)
        for n in neighbors:
            suppressed.add(n)
            if n != idx:
                n_cand = candidates[n]
                if not candidates[idx]["city"] and n_cand["city"]:
                    candidates[idx]["city"] = n_cand["city"]
                if not candidates[idx]["street"] and n_cand["street"]:
                    candidates[idx]["street"] = n_cand["street"]
                if not candidates[idx]["state"] and n_cand["state"]:
                    candidates[idx]["state"] = n_cand["state"]
                if not candidates[idx]["housenumber"] and n_cand["housenumber"]:
                    candidates[idx]["housenumber"] = n_cand["housenumber"]
                if not candidates[idx]["postcode"] and n_cand["postcode"]:
                    candidates[idx]["postcode"] = n_cand["postcode"]

    kept_indices.sort()

    records = []
    for idx in kept_indices:
        c = candidates[idx]
        in_tn = bool(prepared_tn.intersects(Point(c["lon"], c["lat"])))
        state = c["state"] or ("TN" if in_tn else None)
        records.append(
            StoreRecord(
                id=c["id"],
                name=c["name"],
                lat=c["lat"],
                lon=c["lon"],
                city=c["city"],
                street=c["street"],
                state=state,
                in_tn=in_tn,
                housenumber=c["housenumber"],
                postcode=c["postcode"],
            )
        )

    return records


def fetch_dollar_generals(
    tn_geom: shapely.Geometry,
    cache_path: Path | str,
    refresh: bool = False,
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
) -> list[StoreRecord]:
    """Fetch Dollar General stores within bounding box from Overpass API, using local cache if available."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "features" in data:
            records = []
            for feat in data["features"]:
                props = feat.get("properties") or {}
                geom = feat.get("geometry") or {}
                coords = geom.get("coordinates") or [None, None]
                lon = float(props.get("lon", coords[0]))
                lat = float(props.get("lat", coords[1]))
                records.append(
                    StoreRecord(
                        id=str(props.get("id", "")),
                        name=str(props.get("name", "Dollar General")),
                        lat=lat,
                        lon=lon,
                        city=props.get("city"),
                        street=props.get("street"),
                        state=props.get("state"),
                        in_tn=bool(props.get("in_tn", False)),
                        housenumber=props.get("housenumber"),
                        postcode=props.get("postcode"),
                    )
                )
            return records
        elif "elements" in data:
            return parse_and_deduplicate_overpass(data["elements"], tn_geom)

    south, west, north, east = bbox
    query = (
        "[out:json][timeout:180];\n"
        "(\n"
        f'  nwr["name"~"^Dollar General"]({south},{west},{north},{east});\n'
        f'  nwr["brand"="Dollar General"]({south},{west},{north},{east});\n'
        f'  nwr["brand:wikidata"="Q145168"]({south},{west},{north},{east});\n'
        ");\n"
        "out center tags;\n"
    )

    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            logger.info("Querying Overpass API at %s", endpoint)
            resp = requests.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=180,
            )
            resp.raise_for_status()
            res_json = resp.json()
            if "elements" not in res_json:
                raise ValueError("Overpass response missing 'elements' key")
            elements = res_json["elements"]
            stores = parse_and_deduplicate_overpass(elements, tn_geom)

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            geojson_data = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [s.lon, s.lat],
                        },
                        "properties": {
                            "id": s.id,
                            "name": s.name,
                            "lat": s.lat,
                            "lon": s.lon,
                            "city": s.city,
                            "street": s.street,
                            "state": s.state,
                            "in_tn": s.in_tn,
                            "housenumber": s.housenumber,
                            "postcode": s.postcode,
                        },
                    }
                    for s in stores
                ],
            }
            cache_path.write_text(json.dumps(geojson_data, indent=2), encoding="utf-8")
            return stores
        except Exception as exc:
            logger.warning("Overpass endpoint %s failed: %s", endpoint, exc)
            last_error = exc

    raise RuntimeError(f"All Overpass endpoints failed: {last_error}")


class HTTPNominatimClient:
    """Polite Nominatim reverse client: exactly one request message then time.sleep(delay_seconds)."""

    def __init__(
        self,
        endpoint: str = "https://nominatim.openstreetmap.org/reverse",
        delay_seconds: float = 1.05,
    ) -> None:
        self.endpoint = endpoint
        self.delay_seconds = delay_seconds
        self.user_agent = f"{DEFAULT_USER_AGENT} address-enrichment"

    def reverse(self, lat: float, lon: float) -> dict | None:
        """Reverse-geocode one point; return the Nominatim JSON dict or None on any failure."""
        try:
            resp = requests.get(
                self.endpoint,
                params={"lat": lat, "lon": lon, "format": "json", "zoom": 18, "addressdetails": 1},
                headers={"User-Agent": self.user_agent},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.warning("Nominatim reverse geocode failed for (%s, %s): %s", lat, lon, exc)
            return None
        finally:
            time.sleep(self.delay_seconds)


def _load_address_cache(address_cache_path: Path) -> dict[str, str]:
    """Load the id-keyed address cache; a missing/malformed/wrong-version file yields {}."""
    if not address_cache_path.exists():
        return {}
    try:
        data = json.loads(address_cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Address cache %s unreadable, starting fresh: %s", address_cache_path, exc)
        return {}
    if not isinstance(data, dict) or data.get("version") != 1:
        logger.warning("Address cache %s has an unexpected format, starting fresh", address_cache_path)
        return {}
    addresses = data.get("addresses")
    return dict(addresses) if isinstance(addresses, dict) else {}


def _join_address_parts(first_part: str, city: str | None, state: str | None) -> str:
    # Display order: house_number+road, city, state — ", "-joined, empties dropped.
    parts = [part for part in (first_part, city, state) if part]
    return ", ".join(parts)


def enrich_store_addresses(
    stores: list[StoreRecord],
    address_cache_path: Path | str,
    refresh: bool = False,
    client: HTTPNominatimClient | None = None,
) -> dict[str, str]:
    """Return {store_id: display_address} for every store that has an address.

    Stores carrying the complete own-data triple (housenumber+street+city --
    state defaults to "TN" inside TN) are composed without any network call;
    incomplete stores are reverse-geocoded via ``client.reverse(lat, lon)``.
    Successful geocodes are cached id-keyed in ``address_cache_path``; a
    per-store failure (None return or raised exception) is skipped -- never
    raised and never cached. ``refresh=True`` ignores the existing cache and
    re-geocodes every incomplete store (previous entries still survive on disk).
    """
    if client is None:
        client = HTTPNominatimClient()
    cache_path = Path(address_cache_path)
    cached = {} if refresh else _load_address_cache(cache_path)

    display: dict[str, str] = {}
    new_entries: dict[str, str] = {}

    for store in stores:
        if store.id in cached:
            display[store.id] = cached[store.id]
            continue
        if store.housenumber and store.street and store.city:
            state = store.state or ("TN" if store.in_tn else None)
            first_part = " ".join(part for part in (store.housenumber, store.street) if part)
            display[store.id] = _join_address_parts(first_part, store.city, state)
            continue
        try:
            data = client.reverse(store.lat, store.lon)
        except Exception as exc:
            logger.warning("Address enrichment failed for %s: %s", store.id, exc)
            continue
        if not data:
            continue
        address = data.get("address") or {}
        first_part = " ".join(
            part for part in (address.get("house_number"), address.get("road")) if part
        )
        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("hamlet")
        )
        state = address.get("state") or ("TN" if store.in_tn else None)
        text = _join_address_parts(first_part, city, state)
        if not text:
            continue
        display[store.id] = text
        new_entries[store.id] = text

    if new_entries:
        # Verbatim brief semantics: a normal run writes only the newly
        # geocoded entries. A refresh run ignores those entries for lookups
        # but preserves previously geocoded addresses on disk (spec 5.4
        # durability: store re-fetch must not discard them).
        merged = dict(new_entries)
        if refresh:
            merged.update(_load_address_cache(cache_path))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"version": 1, "addresses": merged}, indent=2), encoding="utf-8"
        )

    return display
