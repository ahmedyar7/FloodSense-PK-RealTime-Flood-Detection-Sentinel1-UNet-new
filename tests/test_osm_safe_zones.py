"""
Tests for real OSM safe-zone discovery (agent/osm_safe_zones.py).

These cover the pure logic — distance, flood-mask sampling, candidate assembly,
and straight-line routing — plus Overpass response parsing with a stubbed HTTP
session, so no live network call is made.
"""

import json

import numpy as np
import pytest

from agent import LocationType, SafeZoneCandidate
from agent.osm_safe_zones import (
    build_safe_zone_candidates,
    fetch_osm_safe_zones,
    haversine_km,
    line_crosses_flood,
    point_in_flood,
    straight_line_route,
)

# A 10x10 raster covering 1°x1° centred on a Pakistan-like location. Row 0 is the
# northern (max-lat) edge, matching the GeoTIFF region requested from GEE.
BBOX = [70.0, 30.0, 71.0, 31.0]


def _mask_with_flooded_cell(row: int, col: int) -> np.ndarray:
    mask = np.zeros((10, 10), dtype=bool)
    mask[row, col] = True
    return mask


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def test_haversine_known_distance():
    # ~1° of latitude is ~111 km.
    d = haversine_km(30.0, 70.0, 31.0, 70.0)
    assert 110.0 < d < 112.0


def test_point_in_flood_hits_flooded_pixel():
    # lon 70.05, lat 30.95 -> col 0, row 0 (top-left).
    mask = _mask_with_flooded_cell(0, 0)
    assert point_in_flood(30.95, 70.05, mask, BBOX) is True
    # A dry pixel elsewhere reads False.
    assert point_in_flood(30.05, 70.95, mask, BBOX) is False


def test_point_outside_raster_is_not_flooded():
    mask = np.ones((10, 10), dtype=bool)
    assert point_in_flood(45.0, 90.0, mask, BBOX) is False


def test_line_crosses_flood_detects_intervening_water():
    # Flooded cell in the middle of the raster (row 5, col 5 ~ lon 70.55, lat 30.45).
    mask = _mask_with_flooded_cell(5, 5)
    # A line passing through the centre crosses it...
    assert line_crosses_flood(30.95, 70.05, 30.05, 70.95, mask, BBOX) is True
    # ...while a line hugging a dry edge does not.
    assert line_crosses_flood(30.95, 70.05, 30.95, 70.95, mask, BBOX) is False


# --------------------------------------------------------------------------- #
# Candidate assembly
# --------------------------------------------------------------------------- #
def test_build_candidates_sets_flood_elevation_and_distance():
    raw = [
        {
            "name": "City Hospital",
            "location_type": LocationType.HOSPITAL,
            "latitude": 30.05,
            "longitude": 70.95,
        },
        {
            "name": "Flooded School",
            "location_type": LocationType.SCHOOL,
            "latitude": 30.95,
            "longitude": 70.05,  # this is the flooded top-left pixel
        },
    ]
    mask = _mask_with_flooded_cell(0, 0)
    # Origin near the hospital so it ranks closest.
    candidates = build_safe_zone_candidates(
        raw,
        origin_lat=30.05,
        origin_lon=70.95,
        mask=mask,
        bbox=BBOX,
        elevations=[42.0, None],  # school elevation unknown -> default applied
    )

    hospital, school = candidates
    assert hospital.in_flood_zone is False
    assert hospital.elevation_m == 42.0
    assert school.in_flood_zone is True
    # Missing elevation falls back to the safe default (>= safe threshold).
    assert school.elevation_m >= 15.0
    # Distances are sane: hospital ~0, school far away.
    assert hospital.distance_km < school.distance_km


def test_build_candidates_without_mask_marks_nothing_flooded():
    raw = [
        {
            "name": "Clinic",
            "location_type": LocationType.HOSPITAL,
            "latitude": 30.5,
            "longitude": 70.5,
        }
    ]
    candidates = build_safe_zone_candidates(raw, origin_lat=30.4, origin_lon=70.4)
    assert candidates[0].in_flood_zone is False
    assert candidates[0].route_crosses_flood is False


# --------------------------------------------------------------------------- #
# Straight-line route
# --------------------------------------------------------------------------- #
def test_straight_line_route_distance_and_time():
    safe_zone = SafeZoneCandidate(
        name="District HQ Hospital",
        location_type=LocationType.HOSPITAL,
        latitude=31.0,
        longitude=70.0,
        in_flood_zone=False,
        elevation_m=40.0,
        road_accessible=True,
        route_crosses_flood=False,
    )
    route = straight_line_route("Origin", 30.0, 70.0, safe_zone)
    assert route.path == ["Origin", "District HQ Hospital"]
    # ~111 km at 18 km/h -> ~370 min.
    assert 110.0 < route.distance_km < 112.0
    assert route.estimated_travel_time_min == round(route.distance_km / 18.0 * 60)


# --------------------------------------------------------------------------- #
# Overpass parsing (no network)
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.last_query = None

    def post(self, url, data=None, headers=None, timeout=None):
        self.last_query = data["data"]
        self.last_headers = headers
        return _FakeResponse(self._payload)


def test_fetch_osm_safe_zones_parses_nodes_and_ways():
    payload = {
        "elements": [
            {
                "type": "node",
                "lat": 34.17,
                "lon": 71.74,
                "tags": {"amenity": "hospital", "name": "DHQ Hospital"},
            },
            {
                "type": "way",
                "center": {"lat": 34.15, "lon": 71.73},
                "tags": {"amenity": "school", "name": "Govt High School"},
            },
            # No usable amenity -> skipped.
            {"type": "node", "lat": 34.1, "lon": 71.7, "tags": {"shop": "bakery"}},
            # Missing coordinates -> skipped.
            {"type": "way", "tags": {"amenity": "shelter", "name": "No Center"}},
        ]
    }
    session = _FakeSession(payload)
    shelters = fetch_osm_safe_zones(34.16, 71.73, radius_m=15000, session=session)

    assert [s["name"] for s in shelters] == ["DHQ Hospital", "Govt High School"]
    assert shelters[0]["location_type"] == LocationType.HOSPITAL
    assert shelters[1]["location_type"] == LocationType.SCHOOL
    # The query was anchored on the requested point and radius.
    assert "around:15000,34.16,71.73" in session.last_query
