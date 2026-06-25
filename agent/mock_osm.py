"""
Mock OpenStreetMap data for the Response & Communication Agent.

These simulate the upstream OSM amenity query and road network so the agent can
be exercised end-to-end without a live Overpass / routing service. The scenario
mirrors a Charsadda evacuation: one flooded school, one safe hospital, and one
relief camp whose approach road is partially blocked by floodwater.
"""

from .response_schemas import GraphEdge, LocationType, SafeZoneCandidate

# Candidate shelters returned by a (mock) OSM amenity query.
MOCK_SAFE_ZONE_CANDIDATES = [
    SafeZoneCandidate(
        name="Government High School #1",
        location_type=LocationType.SCHOOL,
        latitude=34.1463,
        longitude=71.7308,
        in_flood_zone=True,  # inside the flood extent — rejected
        elevation_m=8.0,
        road_accessible=True,
        route_crosses_flood=True,
        distance_km=1.1,
    ),
    SafeZoneCandidate(
        name="District Headquarters Hospital",
        location_type=LocationType.HOSPITAL,
        latitude=34.1719,
        longitude=71.7440,
        in_flood_zone=False,  # safe and fully accessible — valid
        elevation_m=42.0,
        road_accessible=True,
        route_crosses_flood=False,
        distance_km=3.2,
    ),
    SafeZoneCandidate(
        name="Relief Camp C",
        location_type=LocationType.RELIEF_CAMP,
        latitude=34.1602,
        longitude=71.7195,
        in_flood_zone=False,
        elevation_m=21.0,
        road_accessible=True,
        route_crosses_flood=True,  # approach road partially blocked — rejected
        distance_km=2.4,
    ),
]

# Mock road network from the citizen's origin to the hospital. The direct
# segment is flooded, forcing the dry route Origin -> Junction A -> Hospital.
MOCK_ROAD_GRAPH = [
    GraphEdge(source="Origin", target="Hospital", distance_km=2.5, flooded=True),
    GraphEdge(source="Origin", target="Junction A", distance_km=1.4),
    GraphEdge(source="Junction A", target="Hospital", distance_km=1.8),
    GraphEdge(source="Origin", target="Junction C", distance_km=2.0),
    GraphEdge(source="Junction C", target="Hospital", distance_km=2.5),
]
