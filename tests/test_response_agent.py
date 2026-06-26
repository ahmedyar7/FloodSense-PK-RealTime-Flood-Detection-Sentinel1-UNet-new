"""
Tests for the Response & Communication Agent (PR 3).

The four mandated cases from pr3.md are implemented exactly:

* Test 1 — location filtering: reject a flooded school and a partially-blocked
  relief camp, recommend the safe, accessible hospital.
* Test 2 — routing: Dijkstra over the mock graph returns 3.2 km / 11 minutes.
* Test 3 — citizen alert: contains the mandated shelter and route phrases.
* Test 4 — authority alert: contains the mandated coverage and population phrases.
"""

import pytest

from agent import (
    EvacuationRoute,
    FloodState,
    GraphEdge,
    LocationType,
    RiskLevel,
    SafeZoneCandidate,
    evaluate_safe_zones,
    generate_authority_alert,
    generate_citizen_alert,
    plan_route,
    recommend_safe_zone,
)


# --------------------------------------------------------------------------- #
# Test Case 1: Location Filtering Evaluation
# --------------------------------------------------------------------------- #
def test_location_filtering_recommends_only_valid_safe_zone():
    school_a = SafeZoneCandidate(
        name="School A",
        location_type=LocationType.SCHOOL,
        latitude=34.1463,
        longitude=71.7308,
        in_flood_zone=True,  # flooded
        elevation_m=8.0,
        road_accessible=True,
        route_crosses_flood=True,
    )
    hospital_b = SafeZoneCandidate(
        name="Hospital B",
        location_type=LocationType.HOSPITAL,
        latitude=34.1719,
        longitude=71.7440,
        in_flood_zone=False,  # safe, road accessible
        elevation_m=40.0,
        road_accessible=True,
        route_crosses_flood=False,
    )
    relief_camp_c = SafeZoneCandidate(
        name="Relief Camp C",
        location_type=LocationType.RELIEF_CAMP,
        latitude=34.1602,
        longitude=71.7195,
        in_flood_zone=False,  # safe, but road partially blocked
        elevation_m=22.0,
        road_accessible=True,
        route_crosses_flood=True,
    )

    candidates = [school_a, hospital_b, relief_camp_c]
    valid = evaluate_safe_zones(candidates)

    # School A and Relief Camp C are rejected; only Hospital B survives.
    assert [c.name for c in valid] == ["Hospital B"]
    assert school_a not in valid
    assert relief_camp_c not in valid

    recommended = recommend_safe_zone(candidates)
    assert recommended is hospital_b


# --------------------------------------------------------------------------- #
# Test Case 2: Routing Calculation
# --------------------------------------------------------------------------- #
def test_dijkstra_route_distance_and_time():
    # Direct Origin->Hospital B segment is flooded, forcing the dry detour via
    # Junction A (1.4 + 1.8 = 3.2 km), which beats the Junction C path (4.5 km).
    graph = [
        GraphEdge(source="Origin", target="Hospital B", distance_km=2.5, flooded=True),
        GraphEdge(source="Origin", target="Junction A", distance_km=1.4),
        GraphEdge(source="Junction A", target="Hospital B", distance_km=1.8),
        GraphEdge(source="Origin", target="Junction C", distance_km=2.0),
        GraphEdge(source="Junction C", target="Hospital B", distance_km=2.5),
    ]

    route = plan_route(graph, origin="Origin", destination="Hospital B")

    assert isinstance(route, EvacuationRoute)
    assert route.path == ["Origin", "Junction A", "Hospital B"]
    assert route.distance_km == 3.2
    assert route.estimated_travel_time_min == 11


def test_route_raises_when_no_dry_path_exists():
    graph = [
        GraphEdge(source="Origin", target="Hospital B", distance_km=2.5, flooded=True),
    ]
    with pytest.raises(ValueError):
        plan_route(graph, origin="Origin", destination="Hospital B")


# --------------------------------------------------------------------------- #
# Test Case 3: Citizen Alert Generation
# --------------------------------------------------------------------------- #
def test_citizen_alert_contains_mandated_phrases():
    alert = generate_citizen_alert(
        district="Charsadda",
        risk_level=RiskLevel.HIGH_RISK,
        shelter_name="Government High School #2",
        distance_km=3.2,
        travel_time_min=11,
    )

    assert "Nearest Safe Shelter: Government High School #2" in alert
    assert "Please follow the highlighted evacuation route immediately." in alert


# --------------------------------------------------------------------------- #
# Test Case 4: Authority Alert Generation
# --------------------------------------------------------------------------- #
def test_authority_alert_contains_mandated_phrases():
    flood_state = FloodState(
        district="Charsadda",
        flood_coverage_percentage=31,
        affected_area_km2=140,
        population_at_risk=120_000,
        available_shelters=8,
    )

    alert = generate_authority_alert(flood_state)

    assert "Current Flood Coverage: 31%" in alert
    assert "Estimated Population at Risk: 120,000" in alert
