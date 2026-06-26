"""
Real safe-zone discovery from OpenStreetMap for the Response & Communication Agent.

The agent's safe-zone logic (``agent/response_agent.py``) is provider-agnostic: it
operates on :class:`SafeZoneCandidate` objects. Originally those came from a fixed
Charsadda scenario in ``agent/mock_osm.py``, so every alert pointed citizens at the
same three hardcoded points regardless of the district under analysis.

This module replaces that mock with *real* shelters fetched live from OpenStreetMap
via the Overpass API, anchored on the actual area being assessed. The four safety
fields each candidate needs are derived with lightweight heuristics:

* ``in_flood_zone``      — sampled from the UNet flood mask at the shelter's pixel.
* ``route_crosses_flood``— True if the straight line origin→shelter touches a
                            flooded pixel in that same mask.
* ``elevation_m``        — supplied by the caller (e.g. sampled from GEE SRTM); a
                            safe default is used when elevation is unavailable so a
                            real shelter is not rejected purely for lack of data.
* ``road_accessible``    — assumed True for an OSM-mapped amenity (we no longer
                            model an explicit road graph; routing is straight-line).

The Overpass fetch is the only network call; the geometry helpers and candidate
builder are pure functions so they are fully unit-testable without a live service.
"""

import math
from typing import Optional, Sequence

import requests

from .response_agent import AVERAGE_EVACUATION_SPEED_KMH
from .response_schemas import (
    EvacuationRoute,
    LocationType,
    SafeZoneCandidate,
)

# Public Overpass endpoint. Override in tests or to use a mirror.
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Mirrors tried in order; the public Overpass instances reject clients that do not
# identify themselves, so a descriptive User-Agent is mandatory (a bare
# ``python-requests`` UA gets a 406 Not Acceptable).
OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)

_OVERPASS_HEADERS = {
    "User-Agent": "FloodSense-PK/1.0 (flood early-warning; contact: floodsense-pk)",
    "Accept": "application/json",
}

# OSM amenity tag -> our internal shelter classification.
_AMENITY_TO_TYPE = {
    "school": LocationType.SCHOOL,
    "college": LocationType.SCHOOL,
    "university": LocationType.SCHOOL,
    "hospital": LocationType.HOSPITAL,
    "clinic": LocationType.HOSPITAL,
    "shelter": LocationType.RELIEF_CAMP,
    "community_centre": LocationType.RELIEF_CAMP,
}

# Elevation assumed for a candidate when no measured value is available. Chosen so a
# real, mapped shelter is not rejected by the safe-elevation rule solely because we
# could not look its elevation up.
DEFAULT_ELEVATION_M = 20.0


# --------------------------------------------------------------------------- #
# Geometry helpers (pure)
# --------------------------------------------------------------------------- #
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points, in kilometres."""
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def _latlon_to_rowcol(
    lat: float, lon: float, bbox: Sequence[float], shape: tuple[int, int]
) -> Optional[tuple[int, int]]:
    """Map a lat/lon to a (row, col) in a north-up raster covering ``bbox``.

    ``bbox`` is ``[min_lon, min_lat, max_lon, max_lat]`` and matches the GeoTIFF
    region requested from Earth Engine, where row 0 is the northern (max-lat) edge.
    Returns ``None`` when the point falls outside the raster.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    height, width = shape
    if max_lon == min_lon or max_lat == min_lat:
        return None
    col = int((lon - min_lon) / (max_lon - min_lon) * width)
    row = int((max_lat - lat) / (max_lat - min_lat) * height)
    if 0 <= row < height and 0 <= col < width:
        return row, col
    return None


def point_in_flood(lat: float, lon: float, mask, bbox: Sequence[float]) -> bool:
    """True if ``(lat, lon)`` falls on a flooded pixel of the UNet ``mask``."""
    if mask is None or bbox is None:
        return False
    rowcol = _latlon_to_rowcol(lat, lon, bbox, mask.shape)
    if rowcol is None:
        return False
    return bool(mask[rowcol[0], rowcol[1]])


def line_crosses_flood(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    mask,
    bbox: Sequence[float],
    samples: int = 64,
) -> bool:
    """True if the straight segment between two points touches a flooded pixel.

    A coarse proxy for "the only obvious approach is blocked by water": we sample
    ``samples`` evenly-spaced points along the line and test each against the mask.
    """
    if mask is None or bbox is None:
        return False
    for i in range(samples + 1):
        t = i / samples
        lat = lat1 + (lat2 - lat1) * t
        lon = lon1 + (lon2 - lon1) * t
        if point_in_flood(lat, lon, mask, bbox):
            return True
    return False


# --------------------------------------------------------------------------- #
# Overpass fetch
# --------------------------------------------------------------------------- #
def _build_overpass_query(lat: float, lon: float, radius_m: int) -> str:
    """Overpass QL selecting candidate-shelter amenities around a point."""
    clauses = []
    for amenity in ("school", "college", "university", "hospital", "clinic", "shelter"):
        clauses.append(f'  node["amenity"="{amenity}"](around:{radius_m},{lat},{lon});')
        clauses.append(f'  way["amenity"="{amenity}"](around:{radius_m},{lat},{lon});')
    body = "\n".join(clauses)
    return f"[out:json][timeout:25];\n(\n{body}\n);\nout center tags;"


def _parse_overpass(data: dict) -> list[dict]:
    """Turn a raw Overpass JSON response into simple shelter dicts.

    Each dict has ``name``, ``location_type``, ``latitude`` and ``longitude``.
    Elements without a usable amenity tag or coordinates are skipped.
    """
    out: list[dict] = []
    for element in data.get("elements", []):
        tags = element.get("tags", {}) or {}
        location_type = _AMENITY_TO_TYPE.get(tags.get("amenity"))
        if location_type is None:
            continue

        if element.get("type") == "node":
            lat, lon = element.get("lat"), element.get("lon")
        else:  # way / relation -> Overpass "out center" gives a representative point
            center = element.get("center") or {}
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            continue

        name = tags.get("name") or f"Unnamed {tags.get('amenity')}"
        out.append(
            {
                "name": name,
                "location_type": location_type,
                "latitude": float(lat),
                "longitude": float(lon),
            }
        )
    return out


def fetch_osm_safe_zones(
    center_lat: float,
    center_lon: float,
    radius_m: int = 20000,
    timeout: int = 30,
    overpass_url: Optional[str] = None,
    session: Optional[requests.Session] = None,
) -> list[dict]:
    """Query Overpass for shelters within ``radius_m`` of a centre point.

    Tries each mirror in :data:`OVERPASS_MIRRORS` in turn (or just ``overpass_url``
    if given) so a single endpoint being slow or rate-limited does not break the
    feature. A descriptive ``User-Agent`` is always sent — public Overpass
    instances answer 406 Not Acceptable to unidentified clients. Returns a list of
    raw shelter dicts (see :func:`_parse_overpass`); the last error is raised only
    if every endpoint fails.
    """
    query = _build_overpass_query(center_lat, center_lon, radius_m)
    http = session or requests
    endpoints = [overpass_url] if overpass_url else list(OVERPASS_MIRRORS)

    last_error: Optional[Exception] = None
    for endpoint in endpoints:
        try:
            response = http.post(
                endpoint,
                data={"data": query},
                headers=_OVERPASS_HEADERS,
                timeout=timeout,
            )
            response.raise_for_status()
            return _parse_overpass(response.json())
        except Exception as e:  # try the next mirror before giving up
            last_error = e

    raise last_error if last_error else RuntimeError("No Overpass endpoint configured.")


# --------------------------------------------------------------------------- #
# Candidate assembly
# --------------------------------------------------------------------------- #
def build_safe_zone_candidates(
    raw_list: Sequence[dict],
    *,
    origin_lat: float,
    origin_lon: float,
    mask=None,
    bbox: Optional[Sequence[float]] = None,
    elevations: Optional[Sequence[Optional[float]]] = None,
    default_elevation_m: float = DEFAULT_ELEVATION_M,
) -> list[SafeZoneCandidate]:
    """Build validated :class:`SafeZoneCandidate` objects from raw OSM shelters.

    Flood-zone membership and route crossing are derived from ``mask``/``bbox``;
    ``elevations`` (parallel to ``raw_list``) supplies per-shelter elevation, with
    ``default_elevation_m`` used wherever a value is missing. ``distance_km`` is the
    straight-line distance from the origin, used to rank candidates.
    """
    candidates: list[SafeZoneCandidate] = []
    for i, raw in enumerate(raw_list):
        lat, lon = raw["latitude"], raw["longitude"]

        elevation = default_elevation_m
        if elevations is not None and i < len(elevations) and elevations[i] is not None:
            elevation = float(elevations[i])

        candidates.append(
            SafeZoneCandidate(
                name=raw["name"],
                location_type=raw["location_type"],
                latitude=lat,
                longitude=lon,
                in_flood_zone=point_in_flood(lat, lon, mask, bbox),
                elevation_m=elevation,
                road_accessible=True,
                route_crosses_flood=line_crosses_flood(
                    origin_lat, origin_lon, lat, lon, mask, bbox
                ),
                distance_km=round(haversine_km(origin_lat, origin_lon, lat, lon), 2),
            )
        )
    return candidates


def straight_line_route(
    origin_label: str,
    origin_lat: float,
    origin_lon: float,
    safe_zone: SafeZoneCandidate,
    speed_kmh: float = AVERAGE_EVACUATION_SPEED_KMH,
) -> EvacuationRoute:
    """A straight-line evacuation estimate from the origin to a safe zone.

    Without an explicit road graph we report the direct distance and a time derived
    from the average evacuation speed; turn-by-turn directions are provided to the
    citizen separately via the Google Maps link in the alert email.
    """
    distance_km = round(
        haversine_km(origin_lat, origin_lon, safe_zone.latitude, safe_zone.longitude),
        1,
    )
    travel_time_min = max(0, round(distance_km / speed_kmh * 60))
    return EvacuationRoute(
        path=[origin_label, safe_zone.name],
        distance_km=distance_km,
        estimated_travel_time_min=travel_time_min,
    )


__all__ = [
    "OVERPASS_URL",
    "DEFAULT_ELEVATION_M",
    "haversine_km",
    "point_in_flood",
    "line_crosses_flood",
    "fetch_osm_safe_zones",
    "build_safe_zone_candidates",
    "straight_line_route",
]
